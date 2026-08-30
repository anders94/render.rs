//! Interactive progressive preview (roadmap Phase 11 deferral): a window
//! that shows the render accumulating sample by sample, re-rendering from
//! scratch whenever the RIB file changes on disk. Keys: q / Esc quit,
//! s saves the current accumulation next to the RIB.
//!
//! The winit event loop owns the main thread (a macOS requirement); the
//! renderer lives on a worker thread — Metal sessions are created and
//! used entirely inside it — and posts frames back through the event-loop
//! proxy.

use render_rs::output::{apply_tonemap, write_png, Image, Tonemap};
use render_rs::parser::{parse_rib_bytes, SceneBuilder};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

pub struct PreviewOptions {
    pub rib_file: PathBuf,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub use_metal: bool,
    pub tonemap: Tonemap,
    pub max_spp: u32,
}

enum UserEvent {
    Frame { image: Image, spp: u32 },
    Status(String),
}

struct App {
    window: Option<Rc<Window>>,
    surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,
    latest: Option<Image>,
    latest_spp: u32,
    tonemap: Tonemap,
    rib_file: PathBuf,
    size: (u32, u32),
    stop: Arc<AtomicBool>,
}

impl App {
    fn present(&mut self) {
        let (Some(surface), Some(img)) = (self.surface.as_mut(), self.latest.as_ref()) else {
            return;
        };
        let (w, h) = self.size;
        if surface
            .resize(NonZeroU32::new(w).unwrap(), NonZeroU32::new(h).unwrap())
            .is_err()
        {
            return;
        }
        let Ok(mut buffer) = surface.buffer_mut() else { return };
        let display = apply_tonemap(self.tonemap, img);
        for (y, row) in display.iter().enumerate() {
            for (x, p) in row.iter().enumerate() {
                let r = (p.x.clamp(0.0, 1.0) * 255.0) as u32;
                let g = (p.y.clamp(0.0, 1.0) * 255.0) as u32;
                let b = (p.z.clamp(0.0, 1.0) * 255.0) as u32;
                buffer[y * w as usize + x] = (r << 16) | (g << 8) | b;
            }
        }
        let _ = buffer.present();
    }

    fn save(&self) {
        let Some(img) = &self.latest else { return };
        let out = self.rib_file.with_extension("preview.png");
        let display = apply_tonemap(self.tonemap, img);
        match write_png(&display, out.to_str().unwrap_or("preview.png")) {
            Ok(()) => println!("saved {} ({} spp)", out.display(), self.latest_spp),
            Err(e) => eprintln!("save failed: {e}"),
        }
    }
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(format!("render.rs preview — {}", self.rib_file.display()))
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.size.0 as f64,
                self.size.1 as f64,
            ))
            .with_resizable(false);
        let window = Rc::new(event_loop.create_window(attrs).expect("window"));
        let context = softbuffer::Context::new(window.clone()).expect("context");
        let surface = softbuffer::Surface::new(&context, window.clone()).expect("surface");
        self.window = Some(window);
        self.surface = Some(surface);
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::Frame { image, spp } => {
                self.latest = Some(image);
                self.latest_spp = spp;
                if let Some(w) = &self.window {
                    w.set_title(&format!(
                        "render.rs preview — {} — {spp} spp",
                        self.rib_file.file_name().unwrap_or_default().to_string_lossy()
                    ));
                    w.request_redraw();
                }
            }
            UserEvent::Status(msg) => {
                if let Some(w) = &self.window {
                    w.set_title(&format!("render.rs preview — {msg}"));
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                self.stop.store(true, Ordering::Relaxed);
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => self.present(),
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => {
                match event.logical_key {
                    Key::Named(NamedKey::Escape) => {
                        self.stop.store(true, Ordering::Relaxed);
                        event_loop.exit();
                    }
                    Key::Character(ref c) if c == "q" => {
                        self.stop.store(true, Ordering::Relaxed);
                        event_loop.exit();
                    }
                    Key::Character(ref c) if c == "s" => self.save(),
                    _ => {}
                }
            }
            _ => {}
        }
    }
}

fn accumulate(sum: &mut Image, sample: &Image, n: u32) -> Image {
    for (rs, r) in sum.iter_mut().zip(sample) {
        for (ps, p) in rs.iter_mut().zip(r) {
            *ps = *ps + *p;
        }
    }
    sum.iter()
        .map(|row| row.iter().map(|p| *p / n as f64).collect())
        .collect()
}

fn render_thread(
    opts: &PreviewOptions,
    proxy: EventLoopProxy<UserEvent>,
    stop: Arc<AtomicBool>,
) {
    let mut last_mtime = std::time::SystemTime::UNIX_EPOCH;
    'outer: while !stop.load(Ordering::Relaxed) {
        // (Re)load the scene.
        let mtime = std::fs::metadata(&opts.rib_file)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        last_mtime = mtime;
        let scene = (|| {
            let bytes = std::fs::read(&opts.rib_file).ok()?;
            let requests = parse_rib_bytes(&bytes).ok()?;
            let mut builder = SceneBuilder::new();
            if let Some(dir) = opts.rib_file.parent() {
                builder = builder.with_base_dir(dir);
            }
            builder.build(&requests).ok()
        })();
        let Some(mut scene) = scene else {
            let _ = proxy.send_event(UserEvent::Status("RIB failed to load".into()));
            std::thread::sleep(std::time::Duration::from_millis(500));
            continue;
        };
        if let Some(w) = opts.width {
            scene.camera.width = w;
        }
        if let Some(h) = opts.height {
            scene.camera.height = h;
        }

        // Progressive accumulation, checking for edits between samples.
        #[cfg(target_os = "macos")]
        let session = if opts.use_metal {
            render_rs::raytracer::metal::PtSession::new(&scene).ok()
        } else {
            None
        };
        #[cfg(not(target_os = "macos"))]
        let session: Option<()> = None;

        let mut cpu_sum: Option<Image> = None;
        for sample in 0..opts.max_spp {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            let image = {
                #[cfg(target_os = "macos")]
                {
                    if let Some(session) = &session {
                        if session.render_samples(sample, 1).is_err() {
                            let _ = proxy.send_event(UserEvent::Status("GPU error".into()));
                            break;
                        }
                        session.image()
                    } else {
                        let s = render_rs::raytracer::pt::render_one(&scene, sample);
                        let sum = cpu_sum.get_or_insert_with(|| {
                            vec![
                                vec![render_rs::math::Vec3::zero(); s[0].len()];
                                s.len()
                            ]
                        });
                        accumulate(sum, &s, sample + 1)
                    }
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let s = render_rs::raytracer::pt::render_one(&scene, sample);
                    let sum = cpu_sum.get_or_insert_with(|| {
                        vec![vec![render_rs::math::Vec3::zero(); s[0].len()]; s.len()]
                    });
                    accumulate(sum, &s, sample + 1)
                }
            };
            let _ = proxy.send_event(UserEvent::Frame { image, spp: sample + 1 });

            // File changed? Restart from scratch.
            let now = std::fs::metadata(&opts.rib_file)
                .and_then(|m| m.modified())
                .unwrap_or(last_mtime);
            if now != last_mtime {
                let _ = proxy.send_event(UserEvent::Status("re-rendering (file changed)".into()));
                continue 'outer;
            }
        }
        // Converged: idle-poll for edits.
        let _ = proxy.send_event(UserEvent::Status(format!("{} spp (done)", opts.max_spp)));
        loop {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(300));
            let now = std::fs::metadata(&opts.rib_file)
                .and_then(|m| m.modified())
                .unwrap_or(last_mtime);
            if now != last_mtime {
                continue 'outer;
            }
        }
    }
}

pub fn run(opts: PreviewOptions) -> anyhow::Result<()> {
    // Peek at the scene once for the window size.
    let bytes = std::fs::read(&opts.rib_file)?;
    let requests = parse_rib_bytes(&bytes)?;
    let mut builder = SceneBuilder::new();
    if let Some(dir) = opts.rib_file.parent() {
        builder = builder.with_base_dir(dir);
    }
    let scene = builder.build(&requests)?;
    let size = (
        opts.width.unwrap_or(scene.camera.width),
        opts.height.unwrap_or(scene.camera.height),
    );
    drop(scene);

    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
    let proxy = event_loop.create_proxy();
    let stop = Arc::new(AtomicBool::new(false));
    let mut app = App {
        window: None,
        surface: None,
        latest: None,
        latest_spp: 0,
        tonemap: opts.tonemap,
        rib_file: opts.rib_file.clone(),
        size,
        stop: stop.clone(),
    };

    let thread_stop = stop.clone();
    std::thread::spawn(move || render_thread(&opts, proxy, thread_stop));

    event_loop.run_app(&mut app)?;
    Ok(())
}
