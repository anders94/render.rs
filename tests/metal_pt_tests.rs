//! Statistical parity tests for the Metal path tracer vs the CPU
//! reference (roadmap Phase 3). f32 GPU and f64 CPU paths diverge
//! chaotically per sample, so parity is statistical: same light
//! transport, mean and RMSE agreement within Monte Carlo noise.

#![cfg(target_os = "macos")]

mod common;

use common::load_fixture_scene;
use render_rs::math::Vec3;
use render_rs::output::Image;
use render_rs::raytracer::{metal, pt};

fn image_mean(img: &Image) -> Vec3 {
    let mut sum = Vec3::zero();
    let mut n = 0.0;
    for row in img {
        for p in row {
            sum = sum + *p;
            n += 1.0;
        }
    }
    sum / n
}

fn rmse(a: &Image, b: &Image) -> f64 {
    let mut sum = 0.0;
    let mut n = 0.0;
    for (ra, rb) in a.iter().zip(b.iter()) {
        for (pa, pb) in ra.iter().zip(rb.iter()) {
            let d = *pa - *pb;
            sum += d.dot(&d);
            n += 3.0;
        }
    }
    (sum / n).sqrt()
}

/// CPU-PT vs Metal-PT: means agree within a few percent, RMSE bounded by
/// Monte Carlo noise at the chosen spp.
fn assert_statistical_parity(fixture: &str, spp: u32, mean_tol: f64, rmse_tol: f64) {
    let scene = load_fixture_scene(fixture, 96, 96);
    let cpu = pt::render(&scene, spp);
    let gpu = metal::render_pt(&scene, spp).unwrap();

    let mc = image_mean(&cpu);
    let mg = image_mean(&gpu);
    for (c, g, ch) in [(mc.x, mg.x, "r"), (mc.y, mg.y, "g"), (mc.z, mg.z, "b")] {
        let denom = c.abs().max(0.02);
        let rel = (c - g).abs() / denom;
        assert!(
            rel < mean_tol,
            "{fixture} channel {ch}: CPU mean {c:.4} vs Metal mean {g:.4} (rel {rel:.4} > {mean_tol})"
        );
    }
    let e = rmse(&cpu, &gpu);
    assert!(e < rmse_tol, "{fixture}: RMSE {e:.4} > {rmse_tol}");
}

#[test]
fn cornell_statistical_parity() {
    assert_statistical_parity("cornell.rib", 256, 0.03, 0.10);
}

#[test]
fn quadric_zoo_statistical_parity() {
    // Whitted-era scene with point/distant lights through the PT path.
    assert_statistical_parity("quadric_zoo.rib", 128, 0.05, 0.15);
}

#[test]
fn mesh_instances_statistical_parity() {
    // Meshes + ObjectInstance through the TLAS/BLAS GPU path.
    let scene_rib = r#"
        Format 96 96 1.0
        Projection "perspective" "fov" [40]
        Option "background" "color" [0.4 0.5 0.7]
        WorldBegin
            LightSource "distantlight" "sun" "from" [4 8 -6] "to" [0 0 6] "intensity" [1.2]
            ObjectBegin "box"
                PointsPolygons [4 4 4 4 4 4]
                    [0 1 2 3  5 4 7 6  4 0 3 7  1 5 6 2  3 2 6 7  4 5 1 0]
                    "P" [-1 -1 -1  1 -1 -1  1 1 -1  -1 1 -1
                         -1 -1 1  1 -1 1  1 1 1  -1 1 1]
            ObjectEnd
            AttributeBegin
                Color 0.7 0.7 0.7
                Surface "matte"
                Polygon "P" [-30 -1.2 -5  30 -1.2 -5  30 -1.2 40  -30 -1.2 40]
            AttributeEnd
            Color 0.8 0.3 0.2
            Translate -1.5 -0.5 6
            Scale 0.7 0.7 0.7
            ObjectInstance "box"
            Identity
            Color 0.2 0.4 0.8
            Translate 1.5 -0.3 7
            Rotate 30 0 1 0
            Scale 0.9 0.9 0.9
            ObjectInstance "box"
        WorldEnd
    "#;
    let requests = render_rs::parser::parse_rib(scene_rib).unwrap();
    let scene = render_rs::parser::SceneBuilder::new().build(&requests).unwrap();
    assert_eq!(scene.instances.len(), 2);

    let cpu = pt::render(&scene, 128);
    let gpu = metal::render_pt(&scene, 128).unwrap();
    let mc = image_mean(&cpu);
    let mg = image_mean(&gpu);
    let rel = ((mc.x - mg.x).abs() + (mc.y - mg.y).abs() + (mc.z - mg.z).abs())
        / (mc.x + mc.y + mc.z).max(0.05);
    assert!(rel < 0.04, "mesh scene mean mismatch: {mc:?} vs {mg:?} (rel {rel:.4})");
    let e = rmse(&cpu, &gpu);
    assert!(e < 0.12, "mesh scene RMSE {e:.4}");
}

#[test]
fn textured_patterns_statistical_parity() {
    // Phase 6: file texture + checker + mix + colorCorrect through the
    // generated-MSL pattern path, vs the CPU tile cache.
    use render_rs::texture::tex::{write_tex, LinearImage};
    let tex_path = std::env::temp_dir().join("render_rs_parity_grad.tex");
    let (w, h) = (64usize, 64usize);
    let mut pixels = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            pixels[i] = x as f32 / (w - 1) as f32;
            pixels[i + 1] = 0.3;
            pixels[i + 2] = y as f32 / (h - 1) as f32;
        }
    }
    write_tex(&tex_path, &LinearImage { width: w, height: h, pixels }).unwrap();

    let scene_rib = format!(
        r#"
        Format 96 96 1.0
        Projection "perspective" "fov" [45]
        Option "background" "color" [0.3 0.4 0.6]
        WorldBegin
            LightSource "distantlight" "sun" "from" [3 8 -4] "to" [0 0 6] "intensity" [1.3]
            AttributeBegin
                Pattern "PxrTexture" "grad" "filename" ["{tex}"]
                Pattern "PxrColorCorrect" "graded" "reference color inputRGB" ["grad:resultRGB"]
                    "gain" [1.2 1.0 0.8] "gamma" [1.4] "saturation" [1.3]
                Bxdf "PxrSurface" "floor" "reference color diffuseColor" ["graded:resultRGB"]
                    "specularIor" [1]
                PointsPolygons [4] [0 1 2 3]
                    "P" [-6 -1 2  6 -1 2  6 -1 14  -6 -1 14]
                    "st" [0 0  4 0  4 4  0 4]
            AttributeEnd
            AttributeBegin
                Pattern "PxrChecker" "check" "colorA" [0.9 0.2 0.15] "colorB" [0.15 0.2 0.9]
                    "sScale" [6] "tScale" [6]
                Pattern "PxrMix" "softened" "reference color color1" ["check:resultRGB"]
                    "color2" [0.5 0.5 0.5] "mix" [0.3]
                Bxdf "PxrSurface" "boxmat" "reference color diffuseColor" ["softened:resultRGB"]
                    "specularIor" [1]
                Translate 0 0 7
                Rotate 25 0 1 0
                PointsPolygons [4 4 4 4 4 4]
                    [0 1 2 3  5 4 7 6  4 0 3 7  1 5 6 2  3 2 6 7  4 5 1 0]
                    "P" [-1 -1 -1  1 -1 -1  1 1 -1  -1 1 -1
                         -1 -1 1  1 -1 1  1 1 1  -1 1 1]
                    "st" [0 0  1 0  1 1  0 1  0 1  1 1  1 0  0 0]
            AttributeEnd
        WorldEnd
    "#,
        tex = tex_path.display()
    );
    let requests = render_rs::parser::parse_rib(&scene_rib).unwrap();
    let scene = render_rs::parser::SceneBuilder::new().build(&requests).unwrap();
    assert_eq!(scene.patterns.len(), 4);

    let cpu = pt::render(&scene, 128);
    let gpu = metal::render_pt(&scene, 128).unwrap();
    let mc = image_mean(&cpu);
    let mg = image_mean(&gpu);
    let rel = ((mc.x - mg.x).abs() + (mc.y - mg.y).abs() + (mc.z - mg.z).abs())
        / (mc.x + mc.y + mc.z).max(0.05);
    assert!(rel < 0.04, "textured scene mean mismatch: {mc:?} vs {mg:?} (rel {rel:.4})");
    let e = rmse(&cpu, &gpu);
    assert!(e < 0.12, "textured scene RMSE {e:.4}");
    std::fs::remove_file(&tex_path).ok();
}

#[test]
fn motion_dof_statistical_parity() {
    // Phase 7: transform + deformation motion blur with thin-lens DoF and
    // a gaussian pixel filter, CPU vs Metal.
    let scene_rib = r#"
        Format 96 96 1.0
        Projection "perspective" "fov" [40]
        Shutter 0 1
        DepthOfField 4 0.8 7
        PixelFilter "gaussian" 2 2
        Option "background" "color" [0.35 0.45 0.65]
        WorldBegin
            LightSource "distantlight" "sun" "from" [4 8 -6] "to" [0 0 6] "intensity" [1.2]
            AttributeBegin
                Color 0.7 0.7 0.7
                Polygon "P" [-30 -1.2 -5  30 -1.2 -5  30 -1.2 40  -30 -1.2 40]
            AttributeEnd
            AttributeBegin
                Color 0.85 0.3 0.2
                MotionBegin [0 1]
                    Translate -1.6 -0.4 7
                    Translate 0.4 -0.4 7
                MotionEnd
                Rotate 20 0 1 0
                Scale 0.6 0.6 0.6
                PointsPolygons [4 4 4 4 4 4]
                    [0 1 2 3  5 4 7 6  4 0 3 7  1 5 6 2  3 2 6 7  4 5 1 0]
                    "P" [-1 -1 -1  1 -1 -1  1 1 -1  -1 1 -1
                         -1 -1 1  1 -1 1  1 1 1  -1 1 1]
            AttributeEnd
            AttributeBegin
                Color 0.2 0.5 0.85
                Translate 1.6 -0.2 8
                MotionBegin [0 1]
                    PointsPolygons [4] [0 1 2 3] "P" [-0.8 -0.8 0  0.8 -0.8 0  0.8 0.8 0  -0.8 0.8 0]
                    PointsPolygons [4] [0 1 2 3] "P" [-0.8 -0.8 0  0.8 -0.8 0  1.6 1.6 0  -1.6 1.6 0]
                MotionEnd
            AttributeEnd
        WorldEnd
    "#;
    let requests = render_rs::parser::parse_rib(scene_rib).unwrap();
    let scene = render_rs::parser::SceneBuilder::new().build(&requests).unwrap();
    assert!(scene.has_motion);
    assert!(scene.camera.lens_radius > 0.0);

    let cpu = pt::render(&scene, 192);
    let gpu = metal::render_pt(&scene, 192).unwrap();
    let mc = image_mean(&cpu);
    let mg = image_mean(&gpu);
    let rel = ((mc.x - mg.x).abs() + (mc.y - mg.y).abs() + (mc.z - mg.z).abs())
        / (mc.x + mc.y + mc.z).max(0.05);
    assert!(rel < 0.04, "motion scene mean mismatch: {mc:?} vs {mg:?} (rel {rel:.4})");
    let e = rmse(&cpu, &gpu);
    assert!(e < 0.12, "motion scene RMSE {e:.4}");
}

#[test]
fn hair_curves_statistical_parity() {
    // Phase 8: capsule curves + Marschner hair BSDF, CPU vs Metal.
    let mut curves = String::new();
    let mut nv = Vec::new();
    let mut seed = 12345u64;
    let mut rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as f64 / (1u64 << 31) as f64 - 1.0
    };
    for i in 0..400 {
        let phi = i as f64 * 0.61803 * std::f64::consts::TAU;
        let z: f64 = -0.2 + 1.15 * ((i as f64 * 0.377).fract());
        let z = z.clamp(-0.2, 0.98);
        let r = (1.0 - z * z).max(0.0).sqrt();
        let (nx, ny, nz) = (r * phi.cos(), z, r * phi.sin());
        nv.push(4);
        for k in 0..4 {
            let t = k as f64 / 3.0;
            let g = 1.0 + 0.45 * t;
            curves.push_str(&format!(
                "{:.4} {:.4} {:.4} ",
                nx * g + 0.08 * t * t * rand(),
                ny * g - 0.25 * t * t,
                nz * g + 0.08 * t * t * rand()
            ));
        }
    }
    let scene_rib = format!(
        r#"
        Format 96 96 1.0
        Projection "perspective" "fov" [35]
        Option "background" "color" [0.4 0.45 0.55]
        WorldBegin
            LightSource "distantlight" "sun" "from" [4 8 -6] "to" [0 0 6] "intensity" [1.4]
            AttributeBegin
                Color 0.6 0.6 0.6
                Polygon "P" [-30 -1.3 -5  30 -1.3 -5  30 -1.3 40  -30 -1.3 40]
            AttributeEnd
            AttributeBegin
                Bxdf "PxrMarschnerHair" "fur" "color" [0.5 0.3 0.12]
                    "roughness" [0.3] "azimuthalRoughness" [0.35]
                Translate 0 0.4 7
                Curves "linear" [{nv}] "nonperiodic"
                    "P" [{curves}]
                    "width" [0.03 0.008]
            AttributeEnd
        WorldEnd
    "#,
        nv = nv
            .iter()
            .map(|n: &i32| n.to_string())
            .collect::<Vec<_>>()
            .join(" "),
        curves = curves
    );
    let requests = render_rs::parser::parse_rib(&scene_rib).unwrap();
    let scene = render_rs::parser::SceneBuilder::new().build(&requests).unwrap();
    assert!(scene.curve_segment_count() > 1000);

    let cpu = pt::render(&scene, 160);
    let gpu = metal::render_pt(&scene, 160).unwrap();
    let mc = image_mean(&cpu);
    let mg = image_mean(&gpu);
    let rel = ((mc.x - mg.x).abs() + (mc.y - mg.y).abs() + (mc.z - mg.z).abs())
        / (mc.x + mc.y + mc.z).max(0.05);
    assert!(rel < 0.04, "hair scene mean mismatch: {mc:?} vs {mg:?} (rel {rel:.4})");
    let e = rmse(&cpu, &gpu);
    assert!(e < 0.12, "hair scene RMSE {e:.4}");
}

#[test]
fn metal_pt_deterministic() {
    let scene = load_fixture_scene("cornell.rib", 64, 64);
    let a = metal::render_pt(&scene, 16).unwrap();
    let b = metal::render_pt(&scene, 16).unwrap();
    for (ra, rb) in a.iter().zip(b.iter()) {
        for (pa, pb) in ra.iter().zip(rb.iter()) {
            assert!(
                pa.x == pb.x && pa.y == pb.y && pa.z == pb.z,
                "Metal PT output is not deterministic"
            );
        }
    }
}
