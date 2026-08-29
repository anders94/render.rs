use anyhow::Result;
use clap::Parser;
use render_rs::output::{write_exr, write_png, write_ppm, write_ppm_ascii};
use render_rs::parser::{parse_rib, SceneBuilder};
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
}

#[derive(Copy, Clone, Debug, PartialEq, clap::ValueEnum)]
enum Integrator {
    Whitted,
    Path,
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

    let rib_content = fs::read_to_string(&args.rib_file)?;

    println!("Parsing RIB file...");
    let commands = parse_rib(&rib_content)?;

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
            println!("Path tracing on Metal at {spp} spp...");
            #[cfg(target_os = "macos")]
            {
                render_rs::raytracer::metal::render_pt(&scene, spp)?
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
        "png" => write_png(&image, args.output.to_str().unwrap())?,
        "exr" => write_exr(&image, args.output.to_str().unwrap())?,
        "ppm-ascii" => write_ppm_ascii(&image, args.output.to_str().unwrap())?,
        _ => write_ppm(&image, args.output.to_str().unwrap())?,
    }

    println!("Done!");

    Ok(())
}
