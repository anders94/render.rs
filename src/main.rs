mod preview;

use anyhow::Result;
use clap::Parser;
use render_rs::output::{write_exr, write_png, write_ppm, write_ppm_ascii};
use render_rs::parser::{parse_rib_bytes, SceneBuilder};
use render_rs::raytracer::renderer::render;
use std::fs;
use std::path::PathBuf;

#[derive(Copy, Clone, Debug, clap::ValueEnum)]
enum Backend {
    /// Multi-threaded CPU raytracer (f64)
    Cpu,
    /// Apple GPU via a native Metal compute kernel (f32, macOS only)
    Metal,
}

#[derive(Parser, Debug)]
#[command(name = "render")]
#[command(author, version, about = "A RenderMan RIB renderer", long_about = None)]
struct Args {
    /// Path to the RIB scene file
    #[arg(value_name = "RIB_FILE")]
    rib_file: PathBuf,

    /// Override image width
    #[arg(short, long)]
    width: Option<u32>,

    /// Override image height
    #[arg(short = 'H', long)]
    height: Option<u32>,

    /// Output file path
    #[arg(short, long, default_value = "output.ppm")]
    output: PathBuf,

    /// Output format: ppm (binary P6), ppm-ascii (P3), png, exr (linear)
    #[arg(short, long, default_value = "ppm")]
    format: String,

    /// Number of threads (default: auto-detect)
    #[arg(short, long)]
    threads: Option<usize>,

    /// Rendering backend
    #[arg(short, long, value_enum, default_value_t = Backend::Cpu)]
    backend: Backend,

    /// Integrator: whitted (fast, direct light + mirror reflections) or
    /// path (progressive Monte Carlo global illumination, CPU only)
    #[arg(short, long, value_enum, default_value_t = Integrator::Whitted)]
    integrator: Integrator,

    /// Samples per pixel for the path integrator (default: PixelSamples
    /// product from the RIB, or 64 if unset there)
    #[arg(long)]
    spp: Option<u32>,

    /// Adaptive sampling tolerance (CPU path integrator only): pixels stop
    /// once their 95% CI relative error drops below this; --spp caps the
    /// budget. Try 0.02.
    #[arg(long)]
    adaptive: Option<f64>,

    /// Checkpoint file for the Metal path tracer: progress is saved
    /// periodically and resumed from this file if it exists.
    #[arg(long)]
    checkpoint: Option<PathBuf>,

    /// Render the full AOV stack (path integrator only). With -f exr the
    /// output becomes a multilayer EXR (beauty/diffuse/specular/albedo/
    /// N/depth/id + manifest); other formats still write beauty.
    #[arg(long)]
    aovs: bool,

    /// AOV-guided denoise of the beauty layer (implies --aovs).
    #[arg(long)]
    denoise: bool,

    /// Display transform for png/ppm output: linear, srgb, or aces.
    #[arg(long, default_value = "srgb")]
    tonemap: String,

    /// Also dump each AOV layer as <prefix>_<layer>.png (with --aovs).
    #[arg(long)]
    aov_dump: Option<String>,

    /// Interactive progressive preview window (q/Esc quit, s save);
    /// re-renders when the RIB file changes on disk.
    #[arg(long)]
    preview: bool,

    /// GPU scheduler for the Metal path tracer: megakernel (default) or
    /// wavefront (queued stages with dead-path compaction).
    #[arg(long, default_value = "megakernel")]
    gpu_schedule: String,

    /// Render only samples [a, b) — for distributed rendering. Combine
    /// with `-f accum` on each node and `render merge` afterward.
    #[arg(long, value_name = "A:B")]
    sample_range: Option<String>,
}

#[derive(Copy, Clone, Debug, PartialEq, clap::ValueEnum)]
enum Integrator {
    Whitted,
    Path,
}

/// Merge accumulation EXRs from distributed nodes into one image.
#[derive(Parser, Debug)]
#[command(name = "render merge")]
struct MergeArgs {
    /// Accum EXR inputs (from `-f accum` runs)
    #[arg(value_name = "INPUT", required = true)]
    inputs: Vec<String>,
    #[arg(short, long, default_value = "merged.png")]
    output: PathBuf,
    /// Output display transform (png only): linear, srgb, aces
    #[arg(long, default_value = "srgb")]
    tonemap: String,
}

/// catrib-equivalent: decode/re-encode RIB between text and binary.
#[derive(Parser, Debug)]
#[command(name = "render catrib")]
struct CatribArgs {
    #[arg(value_name = "INPUT")]
    input: PathBuf,
    #[arg(value_name = "OUTPUT")]
    output: PathBuf,
    /// Emit binary RIB (default: canonical text)
    #[arg(long)]
    binary: bool,
}

/// txmake-equivalent: convert any image to the renderer's tiled-mip .tex.
#[derive(Parser, Debug)]
#[command(name = "render txmake")]
struct TxmakeArgs {
    #[arg(value_name = "INPUT")]
    input: PathBuf,
    #[arg(value_name = "OUTPUT")]
    output: PathBuf,
}

fn main() -> Result<()> {
    // Subcommand dispatch that keeps `render scene.rib` working unchanged.
    if std::env::args().nth(1).as_deref() == Some("catrib") {
        let cr = CatribArgs::parse_from(std::env::args().skip(1));
        let data = fs::read(&cr.input)?;
        let requests = parse_rib_bytes(&data)?;
        if cr.binary {
            fs::write(&cr.output, render_rs::parser::binary::encode_binary(&requests))?;
        } else {
            fs::write(&cr.output, render_rs::parser::binary::encode_text(&requests))?;
        }
        let out_len = fs::metadata(&cr.output)?.len();
        println!(
            "{} ({} bytes) -> {} ({} bytes, {})",
            cr.input.display(),
            data.len(),
            cr.output.display(),
            out_len,
            if cr.binary { "binary" } else { "text" }
        );
        return Ok(());
    }
    if std::env::args().nth(1).as_deref() == Some("merge") {
        let m = MergeArgs::parse_from(std::env::args().skip(1));
        let tonemap = render_rs::output::Tonemap::from_name(&m.tonemap)
            .ok_or_else(|| anyhow::anyhow!("unknown tonemap {:?}", m.tonemap))?;
        let image = render_rs::output::accum::merge(&m.inputs)?;
        let out = m.output.to_str().unwrap();
        if out.ends_with(".exr") {
            render_rs::output::write_exr(&image, out)?;
        } else {
            write_png(&render_rs::output::apply_tonemap(tonemap, &image), out)?;
        }
        println!("merged {} accum files -> {}", m.inputs.len(), out);
        return Ok(());
    }
    if std::env::args().nth(1).as_deref() == Some("txmake") {
        let tx = TxmakeArgs::parse_from(std::env::args().skip(1));
        let header = render_rs::texture::tex::txmake(&tx.input, &tx.output)
            .map_err(|e| anyhow::anyhow!(e))?;
        println!(
            "{} -> {} ({}x{}, {} mip levels, {}px tiles)",
            tx.input.display(),
            tx.output.display(),
            header.width,
            header.height,
            header.mips.len(),
            header.tile_size
        );
        return Ok(());
    }

    let args = Args::parse();

    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
    }

    println!("render.rs - RenderMan RIB Renderer");
    println!("Reading RIB file: {}", args.rib_file.display());

    let rib_content = fs::read(&args.rib_file)?;

    println!("Parsing RIB file...");
    let commands = parse_rib_bytes(&rib_content)?;

    println!("Building scene...");
    let mut builder = SceneBuilder::new();
    if let Some(dir) = args.rib_file.parent() {
        builder = builder.with_base_dir(dir);
    }
    let mut scene = builder.build(&commands)?;

    println!(
        "Scene has {} objects, {} meshes / {} instances ({} triangles), {} lights, {} materials",
        scene.objects.len(),
        scene.meshes.len(),
        scene.instances.len(),
        scene.triangle_count() + scene.curve_segment_count(),
        scene.lights.len(),
        scene.materials.len()
    );

    if let Some(width) = args.width {
        scene.camera.width = width;
    }
    if let Some(height) = args.height {
        scene.camera.height = height;
    }

    println!(
        "Rendering {}x{} image ({}x{} samples/pixel)...",
        scene.camera.width, scene.camera.height,
        scene.pixel_samples.0, scene.pixel_samples.1
    );
    let tonemap = render_rs::output::Tonemap::from_name(&args.tonemap)
        .ok_or_else(|| anyhow::anyhow!("unknown tonemap {:?}", args.tonemap))?;

    if args.preview {
        return preview::run(preview::PreviewOptions {
            rib_file: args.rib_file.clone(),
            width: args.width,
            height: args.height,
            use_metal: matches!(args.backend, Backend::Metal),
            tonemap,
            max_spp: args.spp.unwrap_or(512),
        });
    }

    // AOV path: render the film, optionally denoise, then write.
    if (args.aovs || args.denoise) && args.integrator == Integrator::Path {
        let spp = args.spp.unwrap_or_else(|| {
            let (sx, sy) = scene.pixel_samples;
            (sx * sy).max(64)
        });
        let film = match args.backend {
            Backend::Cpu => {
                println!("Path tracing AOVs at {spp} spp...");
                render_rs::raytracer::pt::render_film(&scene, spp)
            }
            Backend::Metal => {
                println!("Path tracing AOVs on Metal at {spp} spp...");
                #[cfg(target_os = "macos")]
                {
                    render_rs::raytracer::metal::render_pt_film(&scene, spp)?
                }
                #[cfg(not(target_os = "macos"))]
                anyhow::bail!("the metal backend requires macOS")
            }
        };
        let beauty = if args.denoise {
            if let Some(out) = render_rs::output::oidn::denoise_oidn(&film) {
                println!("Denoising (OIDN, albedo/normal-guided)...");
                out
            } else {
                println!("Denoising (albedo/normal-guided a-trous)...");
                render_rs::output::denoise(&film)
            }
        } else {
            film.beauty.clone()
        };
        if let Some(prefix) = &args.aov_dump {
            use render_rs::math::Vec3;
            let tm = |img: &render_rs::output::Image| {
                render_rs::output::apply_tonemap(tonemap, img)
            };
            write_png(&tm(&beauty), &format!("{prefix}_beauty.png"))?;
            write_png(&tm(&film.diffuse), &format!("{prefix}_diffuse.png"))?;
            write_png(&tm(&film.specular), &format!("{prefix}_specular.png"))?;
            write_png(&tm(&film.albedo), &format!("{prefix}_albedo.png"))?;
            let nrm: render_rs::output::Image = film
                .normal
                .iter()
                .map(|r| r.iter().map(|n| (*n + Vec3::one()) * 0.5).collect())
                .collect();
            write_png(&nrm, &format!("{prefix}_normal.png"))?;
            let dmax = film
                .depth
                .iter()
                .flatten()
                .map(|d| d.x)
                .fold(0.0f64, f64::max)
                .max(1e-6);
            let dep: render_rs::output::Image = film
                .depth
                .iter()
                .map(|r| {
                    r.iter()
                        .map(|d| Vec3::one() * (1.0 - (d.x / dmax)).max(0.0))
                        .collect()
                })
                .collect();
            write_png(&dep, &format!("{prefix}_depth.png"))?;
            let idc: render_rs::output::Image = film
                .id
                .iter()
                .map(|r| {
                    r.iter()
                        .map(|v| {
                            let id = v.x as u32;
                            if id == 0 {
                                return Vec3::zero();
                            }
                            let h = id.wrapping_mul(2654435761);
                            Vec3::new(
                                (h & 0xFF) as f64 / 255.0,
                                ((h >> 8) & 0xFF) as f64 / 255.0,
                                ((h >> 16) & 0xFF) as f64 / 255.0,
                            )
                        })
                        .collect()
                })
                .collect();
            write_png(&idc, &format!("{prefix}_id.png"))?;
            println!("AOV layers dumped to {prefix}_*.png");
        }
        println!("Writing output to {}...", args.output.display());
        match args.format.as_str() {
            "exr" => render_rs::output::write_multilayer_exr(&film, args.output.to_str().unwrap())?,
            "png" => write_png(
                &render_rs::output::apply_tonemap(tonemap, &beauty),
                args.output.to_str().unwrap(),
            )?,
            "ppm-ascii" => write_ppm_ascii(&beauty, args.output.to_str().unwrap())?,
            _ => write_ppm(&beauty, args.output.to_str().unwrap())?,
        }
        println!("Done!");
        return Ok(());
    }

    // Distributed node: render a sample range, write raw accumulation.
    if let Some(range) = &args.sample_range {
        let (a, b) = range
            .split_once(':')
            .and_then(|(a, b)| Some((a.parse::<u32>().ok()?, b.parse::<u32>().ok()?)))
            .ok_or_else(|| anyhow::anyhow!("--sample-range wants A:B, got {range:?}"))?;
        if b <= a {
            anyhow::bail!("--sample-range: empty range {a}:{b}");
        }
        if args.integrator != Integrator::Path {
            anyhow::bail!("--sample-range requires --integrator path");
        }
        let (sum, count) = match args.backend {
            Backend::Cpu => {
                println!("Path tracing samples [{a}, {b}) on CPU...");
                let sum = render_rs::raytracer::pt::render_sum(&scene, a, b);
                let n = (b - a) as f64;
                let count =
                    vec![vec![n; scene.camera.width as usize]; scene.camera.height as usize];
                (sum, count)
            }
            Backend::Metal => {
                #[cfg(target_os = "macos")]
                {
                    println!("Path tracing samples [{a}, {b}) on Metal...");
                    let session = render_rs::raytracer::metal::PtSession::new(&scene)?;
                    session.render_samples(a, b - a)?;
                    session.sum_and_weight()
                }
                #[cfg(not(target_os = "macos"))]
                anyhow::bail!("the metal backend requires macOS")
            }
        };
        println!("Writing accumulation to {}...", args.output.display());
        render_rs::output::accum::write_accum_exr(
            args.output.to_str().unwrap(),
            &sum,
            &count,
        )?;
        println!("Done! Merge nodes with: render merge -o out.png <accum files>");
        return Ok(());
    }

    let image = match (args.integrator, args.backend) {
        (Integrator::Path, Backend::Cpu) => {
            let spp = args.spp.unwrap_or_else(|| {
                let (sx, sy) = scene.pixel_samples;
                (sx * sy).max(64)
            });
            if let Some(tol) = args.adaptive {
                println!("Adaptive path tracing (tol {tol}, max {spp} spp)...");
                let (image, avg) = render_rs::raytracer::pt::render_adaptive(&scene, spp, tol);
                println!("Adaptive sampling averaged {avg:.1} spp");
                image
            } else {
                println!("Path tracing at {spp} spp...");
                render_rs::raytracer::pt::render(&scene, spp)
            }
        }
        (Integrator::Path, Backend::Metal) => {
            let spp = args.spp.unwrap_or_else(|| {
                let (sx, sy) = scene.pixel_samples;
                (sx * sy).max(64)
            });
            #[cfg(target_os = "macos")]
            {
                if args.gpu_schedule == "wavefront" {
                    let mut session = render_rs::raytracer::metal::WfSession::new(&scene)?;
                    if let Some(tol) = args.adaptive {
                        println!(
                            "Adaptive path tracing on Metal (wavefront, tol {tol}, max {spp} spp)..."
                        );
                        session.set_adaptive(tol);
                        session.render_samples(0, spp)?;
                        println!("Adaptive sampling averaged {:.1} spp", session.average_spp());
                    } else {
                        println!("Path tracing on Metal (wavefront) at {spp} spp...");
                        session.render_samples(0, spp)?;
                    }
                    session.image()
                } else {
                    println!("Path tracing on Metal at {spp} spp...");
                    render_rs::raytracer::metal::render_pt_checkpointed(
                        &scene,
                        spp,
                        args.checkpoint.as_deref(),
                    )?
                }
            }
            #[cfg(not(target_os = "macos"))]
            anyhow::bail!("the metal backend requires macOS")
        }
        (Integrator::Whitted, backend) => match backend {
        Backend::Cpu => render(&scene),
        Backend::Metal => {
            #[cfg(target_os = "macos")]
            {
                render_rs::raytracer::metal::render(&scene)?
            }
            #[cfg(not(target_os = "macos"))]
            anyhow::bail!("the metal backend requires macOS")
        }
        },
    };

    if !scene.patterns.is_empty() {
        println!("{}", render_rs::texture::global_cache().stats_line());
    }

    println!("Writing output to {}...", args.output.display());
    match args.format.as_str() {
        "png" => write_png(
            &render_rs::output::apply_tonemap(tonemap, &image),
            args.output.to_str().unwrap(),
        )?,
        "exr" => write_exr(&image, args.output.to_str().unwrap())?,
        "ppm-ascii" => write_ppm_ascii(&image, args.output.to_str().unwrap())?,
        _ => write_ppm(&image, args.output.to_str().unwrap())?,
    }

    println!("Done!");

    Ok(())
}
