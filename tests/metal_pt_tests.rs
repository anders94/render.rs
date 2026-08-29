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
