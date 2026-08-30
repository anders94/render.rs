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
fn many_lights_statistical_parity() {
    // Phase 9: 500 point lights through the light BVH, CPU vs Metal.
    let mut lights = String::new();
    for i in 0..500 {
        let a = i as f64 * 0.61803 * std::f64::consts::TAU;
        let r = 2.0 + (i % 23) as f64 * 1.2;
        let (x, z) = (r * a.cos(), 8.0 + r * a.sin().abs() * 2.0);
        let y = 0.4 + (i % 7) as f64 * 0.5;
        let (cr, cg, cb) = match i % 3 {
            0 => (1.0, 0.6, 0.3),
            1 => (0.4, 0.7, 1.0),
            _ => (0.9, 0.3, 0.5),
        };
        lights.push_str(&format!(
            "LightSource \"pointlight\" \"l{i}\" \"from\" [{x:.3} {y:.3} {z:.3}] \
             \"intensity\" [0.05] \"lightcolor\" [{cr} {cg} {cb}]\n"
        ));
    }
    let scene_rib = format!(
        r#"
        Format 96 96 1.0
        Projection "perspective" "fov" [40]
        WorldBegin
            {lights}
            AttributeBegin
                Color 0.65 0.65 0.65
                Polygon "P" [-60 0 -5  60 0 -5  60 0 80  -60 0 80]
            AttributeEnd
            Color 0.7 0.4 0.3
            Translate 0 1 9
            Sphere 1 -1 1 360
        WorldEnd
    "#
    );
    let requests = render_rs::parser::parse_rib(&scene_rib).unwrap();
    let scene = render_rs::parser::SceneBuilder::new().build(&requests).unwrap();
    assert_eq!(scene.lights.len(), 500);
    assert!(scene.light_sampler.nodes.len() >= 999);

    let cpu = pt::render(&scene, 128);
    let gpu = metal::render_pt(&scene, 128).unwrap();
    let mc = image_mean(&cpu);
    let mg = image_mean(&gpu);
    let rel = ((mc.x - mg.x).abs() + (mc.y - mg.y).abs() + (mc.z - mg.z).abs())
        / (mc.x + mc.y + mc.z).max(0.05);
    assert!(rel < 0.04, "many-light mean mismatch: {mc:?} vs {mg:?} (rel {rel:.4})");
    let e = rmse(&cpu, &gpu);
    assert!(e < 0.12, "many-light RMSE {e:.4}");
}

#[test]
fn volumes_sss_statistical_parity() {
    // Phase 10: atmosphere + heterogeneous cloud hull + SSS sphere.
    let scene_rib = r#"
        Format 96 96 1.0
        Projection "perspective" "fov" [40]
        Option "background" "color" [0.05 0.05 0.08]
        Translate 0 -1.5 0
        Atmosphere "haze" "sigma_a" [0.005 0.005 0.005] "sigma_s" [0.02 0.025 0.03] "g" [0.2]
        WorldBegin
            LightSource "pointlight" "beam" "from" [0 6 12] "intensity" [30] "lightcolor" [1 0.9 0.7]
            LightSource "distantlight" "sun" "from" [4 8 -6] "to" [0 0 8] "intensity" [0.7]
            Bxdf "PxrSurface" "floor" "diffuseColor" [0.3 0.3 0.32] "specularIor" [1]
            Polygon "P" [-40 0 -5  40 0 -5  40 0 60  -40 0 60]
            Bxdf "PxrSurface" "sss" "diffuseGain" [0]
                "subsurfaceGain" [1] "subsurfaceColor" [0.8 0.45 0.3] "subsurfaceDmfp" [0.3 0.18 0.1]
            TransformBegin Translate -1.8 1.1 9 Sphere 1.1 -1.1 1.1 360 TransformEnd
            AttributeBegin
                Bxdf "PxrSurface" "hull" "diffuseGain" [0] "specularIor" [1]
                Interior "cloud" "sigma_a" [0.02 0.02 0.02] "sigma_s" [2.5 2.5 2.5] "g" [0.4]
                    "density" ["fbm"] "frequency" [0.7] "octaves" [4] "coverage" [0.6] "sharpness" [5]
                Translate 2.2 2.6 13
                Sphere 2.2 -2.2 2.2 360
            AttributeEnd
        WorldEnd
    "#;
    let requests = render_rs::parser::parse_rib(scene_rib).unwrap();
    let scene = render_rs::parser::SceneBuilder::new().build(&requests).unwrap();
    assert_eq!(scene.media.len(), 2);
    assert!(scene.atmosphere.is_some());

    let cpu = pt::render(&scene, 160);
    let gpu = metal::render_pt(&scene, 160).unwrap();
    let mc = image_mean(&cpu);
    let mg = image_mean(&gpu);
    let rel = ((mc.x - mg.x).abs() + (mc.y - mg.y).abs() + (mc.z - mg.z).abs())
        / (mc.x + mc.y + mc.z).max(0.05);
    assert!(rel < 0.05, "volume scene mean mismatch: {mc:?} vs {mg:?} (rel {rel:.4})");
    let e = rmse(&cpu, &gpu);
    assert!(e < 0.15, "volume scene RMSE {e:.4}");
}

#[test]
fn camera_motion_statistical_parity() {
    // Deferral: camera transform blur — a pre-WorldBegin motion block
    // pans the camera over the shutter.
    let scene_rib = r#"
        Format 96 96 1.0
        Projection "perspective" "fov" [40]
        Shutter 0 1
        Option "background" "color" [0.35 0.45 0.65]
        MotionBegin [0 1]
            Translate 0 0 0
            Translate 1.2 0.2 0
        MotionEnd
        WorldBegin
            LightSource "distantlight" "sun" "from" [4 8 -6] "to" [0 0 6] "intensity" [1.2]
            Color 0.7 0.7 0.7
            Polygon "P" [-30 -1.2 -5  30 -1.2 -5  30 -1.2 40  -30 -1.2 40]
            Color 0.8 0.3 0.2
            Translate 0 0 7
            Sphere 1 -1 1 360
        WorldEnd
    "#;
    let requests = render_rs::parser::parse_rib(scene_rib).unwrap();
    let scene = render_rs::parser::SceneBuilder::new().build(&requests).unwrap();
    assert!(scene.camera.motion_inv.is_some());
    assert!(scene.has_motion);

    let cpu = pt::render(&scene, 128);
    let gpu = metal::render_pt(&scene, 128).unwrap();
    let mc = image_mean(&cpu);
    let mg = image_mean(&gpu);
    let rel = ((mc.x - mg.x).abs() + (mc.y - mg.y).abs() + (mc.z - mg.z).abs())
        / (mc.x + mc.y + mc.z).max(0.05);
    assert!(rel < 0.04, "camera motion mean mismatch: {mc:?} vs {mg:?} (rel {rel:.4})");
    let e = rmse(&cpu, &gpu);
    assert!(e < 0.12, "camera motion RMSE {e:.4}");

    // And the blur is real: the sphere's edge must smear horizontally
    // compared to a static render.
    let static_rib = scene_rib.replace(
        "MotionBegin [0 1]
            Translate 0 0 0
            Translate 1.2 0.2 0
        MotionEnd",
        "",
    );
    let sscene = render_rs::parser::SceneBuilder::new()
        .build(&render_rs::parser::parse_rib(&static_rib).unwrap())
        .unwrap();
    let stat = pt::render(&sscene, 128);
    // The pan drags the sphere across the frame over the shutter, so its
    // redness-weighted centroid must shift horizontally vs the static
    // render (robust against noise and partial-coverage thresholds).
    let centroid_of = |img: &Image| -> f64 {
        let mut wsum = 0.0;
        let mut xsum = 0.0;
        for row in img.iter().take(70).skip(20) {
            for (x, p) in row.iter().enumerate() {
                let redness = (p.x - p.z).max(0.0);
                wsum += redness;
                xsum += redness * x as f64;
            }
        }
        xsum / wsum.max(1e-9)
    };
    let c_moving = centroid_of(&cpu);
    let c_static = centroid_of(&stat);
    assert!(
        (c_moving - c_static).abs() > 4.0,
        "no smear: moving centroid {c_moving:.1} vs static {c_static:.1}"
    );
}

#[test]
fn wavefront_matches_megakernel_exactly() {
    // The wavefront scheduler runs the same pt_shade_step and RNG streams
    // as the megakernel — output must match to floating-point identity,
    // not just statistically.
    let scene = load_fixture_scene("cornell.rib", 96, 96);
    let mega = metal::render_pt(&scene, 24).unwrap();
    let session = metal::WfSession::new(&scene).unwrap();
    session.render_samples(0, 24).unwrap();
    let wave = session.image();
    let mut max_delta = 0.0f64;
    for (ra, rb) in mega.iter().zip(&wave) {
        for (pa, pb) in ra.iter().zip(rb) {
            max_delta = max_delta
                .max((pa.x - pb.x).abs())
                .max((pa.y - pb.y).abs())
                .max((pa.z - pb.z).abs());
        }
    }
    assert!(
        max_delta < 1e-6,
        "wavefront diverged from megakernel: max delta {max_delta}"
    );
}

#[test]
fn wavefront_adaptive_saves_samples() {
    // GPU adaptive sampling: converged pixels stop spawning paths; the
    // weight channel keeps the estimator exact.
    let scene = load_fixture_scene("quadric_zoo.rib", 96, 96);
    let mut session = metal::WfSession::new(&scene).unwrap();
    session.set_adaptive(0.05);
    session.render_samples(0, 256).unwrap();
    let avg = session.average_spp();
    assert!(avg < 200.0, "adaptive averaged {avg} spp");
    assert!(avg >= 32.0, "warmup floor violated: {avg}");
    let adaptive = session.image();
    let full = metal::render_pt(&scene, 256).unwrap();
    let ma = image_mean(&adaptive);
    let mf = image_mean(&full);
    let rel = ((ma.x - mf.x).abs() + (ma.y - mf.y).abs() + (ma.z - mf.z).abs())
        / (mf.x + mf.y + mf.z).max(0.05);
    assert!(rel < 0.02, "adaptive mean {ma:?} vs full {mf:?} (rel {rel:.4})");
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
