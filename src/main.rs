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
    /// Apple Silicon GPU via MLX (f32, requires the `mlx` build feature)
    Mlx,
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
}

#[derive(Copy, Clone, Debug, PartialEq, clap::ValueEnum)]
enum Integrator {
    Whitted,
    Path,
}

fn main() -> Result<()> {
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
    let mut scene = SceneBuilder::new().build(&commands)?;

    println!("Scene has {} objects, {} lights, {} materials",
             scene.objects.len(), scene.lights.len(), scene.materials.len());

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
            println!("Path tracing at {spp} spp...");
            render_rs::raytracer::pt::render(&scene, spp)
        }
        (Integrator::Path, _) => {
            anyhow::bail!("the path integrator is CPU-only until roadmap Phase 3; use --backend cpu")
        }
        (Integrator::Whitted, backend) => match backend {
        Backend::Cpu => render(&scene),
        Backend::Mlx => {
            #[cfg(feature = "mlx")]
            {
                render_rs::raytracer::mlx::render(&scene)?
            }
            #[cfg(not(feature = "mlx"))]
            anyhow::bail!(
                "this binary was built without MLX support; rebuild with `cargo build --release --features mlx`"
            )
        }
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
