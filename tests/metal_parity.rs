//! Parity tests for the native Metal backend (macOS only; runs in the
//! default `cargo test` on macOS — no feature flags needed).

#![cfg(target_os = "macos")]

mod common;

use common::{
    assert_quantized_parity, flat_object_for, load_fixture_scene, quantize, random_rays, Lcg,
};
use render_rs::geometry::{
    Cone, Cylinder, Disk, Hyperboloid, Intersectable, Paraboloid, Sphere, Torus, Triangle,
};
use render_rs::math::Matrix4;
use render_rs::raytracer::metal;
use render_rs::raytracer::renderer::render as render_cpu;
use render_rs::raytracer::Ray;

// ---------------------------------------------------------------------------
// Pipeline smoke: compile + dispatch + readback on a tiny render.

#[test]
fn metal_pipeline_smoke() {
    let scene = load_fixture_scene("minimal_fixed.rib", 32, 24);
    let image = metal::render(&scene).unwrap();
    assert_eq!(image.len(), 24);
    assert_eq!(image[0].len(), 32);
    // The red sphere must actually show up.
    let max_r = image
        .iter()
        .flatten()
        .map(|p| p.x)
        .fold(0.0f64, f64::max);
    assert!(max_r > 0.1, "expected a visible object, got max red {max_r}");
}

// ---------------------------------------------------------------------------
// Primitive-level parity via the intersect_probe kernel.

fn assert_primitive_parity(object: &dyn Intersectable, rays: &[Ray], label: &str) {
    let flat = flat_object_for(object);
    let probe_rays: Vec<[f32; 6]> = rays
        .iter()
        .map(|r| {
            [
                r.origin.x as f32,
                r.origin.y as f32,
                r.origin.z as f32,
                r.direction.x as f32,
                r.direction.y as f32,
                r.direction.z as f32,
            ]
        })
        .collect();
    let hits = metal::intersect_probe(&flat, &probe_rays).unwrap();

    let mut mismatches = 0usize;
    for (i, ray) in rays.iter().enumerate() {
        let cpu = object.intersect(ray);
        let (gpu_valid, gpu_t) = hits[i];
        match (cpu, gpu_valid) {
            (Some(cpu_hit), true) => {
                let rel = ((cpu_hit.t - gpu_t as f64) / cpu_hit.t.max(1e-6)).abs();
                if rel > 1e-3 {
                    mismatches += 1;
                }
            }
            (None, false) => {}
            // f32 vs f64 can legitimately flip razor-edge hits; count them.
            _ => mismatches += 1,
        }
    }
    let ratio = mismatches as f64 / rays.len() as f64;
    assert!(
        ratio < 0.01,
        "{label}: {mismatches}/{} rays disagree with CPU (allowing <1% edge flips)",
        rays.len()
    );
}

#[test]
fn sphere_parity() {
    let mut rng = Lcg(42);
    let rays = random_rays(&mut rng, 4000);
    let transform = Matrix4::translate(0.3, -0.2, 0.5) * Matrix4::rotate(30.0, 0.0, 1.0, 0.0);
    assert_primitive_parity(
        &Sphere::new(1.0, -1.0, 1.0, 360.0, 0, transform),
        &rays,
        "sphere full",
    );
    assert_primitive_parity(
        &Sphere::new(1.2, -0.6, 0.9, 270.0, 0, Matrix4::scale(1.5, 0.7, 1.0)),
        &rays,
        "sphere partial (zclip + thetamax, nonuniform scale)",
    );
}

#[test]
fn cylinder_parity() {
    let mut rng = Lcg(7);
    let rays = random_rays(&mut rng, 4000);
    assert_primitive_parity(
        &Cylinder::new(0.8, -1.0, 1.0, 360.0, 0, Matrix4::rotate(-90.0, 1.0, 0.0, 0.0)),
        &rays,
        "cylinder full",
    );
    assert_primitive_parity(
        &Cylinder::new(0.8, -0.5, 0.8, 270.0, 0, Matrix4::translate(0.2, 0.1, 0.4)),
        &rays,
        "cylinder partial",
    );
}

#[test]
fn cone_parity() {
    let mut rng = Lcg(1234);
    let rays = random_rays(&mut rng, 4000);
    assert_primitive_parity(
        &Cone::new(2.0, 0.8, 360.0, 0, Matrix4::rotate(-90.0, 1.0, 0.0, 0.0)),
        &rays,
        "cone full",
    );
    assert_primitive_parity(
        &Cone::new(1.5, 0.6, 270.0, 0, Matrix4::translate(0.1, -0.2, 0.3)),
        &rays,
        "cone partial",
    );
}

#[test]
fn torus_parity() {
    let mut rng = Lcg(99);
    let rays = random_rays(&mut rng, 4000);
    assert_primitive_parity(
        &Torus::new(1.2, 0.4, 0.0, 360.0, 360.0, 0, Matrix4::rotate(35.0, 1.0, 0.0, 0.0)),
        &rays,
        "torus full",
    );
    assert_primitive_parity(
        &Torus::new(1.0, 0.3, -90.0, 90.0, 270.0, 0, Matrix4::translate(0.2, 0.0, 0.3)),
        &rays,
        "torus partial (phi + theta sweep)",
    );
}

#[test]
fn disk_paraboloid_hyperboloid_parity() {
    let mut rng = Lcg(7777);
    let rays = random_rays(&mut rng, 4000);
    assert_primitive_parity(
        &Disk::new(0.2, 1.5, 300.0, 0, Matrix4::rotate(-40.0, 1.0, 0.0, 0.0)),
        &rays,
        "disk",
    );
    assert_primitive_parity(
        &Paraboloid::new(1.0, 0.1, 1.4, 360.0, 0, Matrix4::rotate(-90.0, 1.0, 0.0, 0.0)),
        &rays,
        "paraboloid",
    );
    assert_primitive_parity(
        &Hyperboloid::new([1.0, 0.2, -0.8], [0.4, 0.0, 0.8], 360.0, 0, Matrix4::identity()),
        &rays,
        "hyperboloid",
    );
}

#[test]
fn triangle_parity() {
    let mut rng = Lcg(31337);
    let rays = random_rays(&mut rng, 4000);
    assert_primitive_parity(
        &Triangle::new(
            [-1.5, -1.0, 0.2],
            [1.5, -0.8, -0.3],
            [0.1, 1.4, 0.1],
            0,
            Matrix4::rotate(20.0, 0.0, 1.0, 0.0),
        ),
        &rays,
        "triangle",
    );
}

// ---------------------------------------------------------------------------
// Full-image parity over the fixture scenes.

fn assert_image_parity(fixture: &str) {
    let scene = load_fixture_scene(fixture, 160, 120);
    let cpu = quantize(&render_cpu(&scene));
    let gpu = quantize(&metal::render(&scene).unwrap());
    assert_quantized_parity(&cpu, &gpu, fixture);
}

#[test]
fn parity_minimal() {
    assert_image_parity("minimal_fixed.rib");
}

#[test]
fn parity_simple_sphere() {
    assert_image_parity("simple_sphere.rib");
}

#[test]
fn parity_three_primitives() {
    assert_image_parity("three_primitives.rib");
}

#[test]
fn parity_transforms() {
    assert_image_parity("transforms.rib");
}

#[test]
fn parity_materials() {
    assert_image_parity("materials.rib");
}

#[test]
fn parity_shadows() {
    assert_image_parity("shadows.rib");
}

#[test]
fn parity_scene() {
    assert_image_parity("scene.rib");
}

#[test]
fn parity_showcase() {
    assert_image_parity("showcase.rib");
}

#[test]
fn parity_quadric_zoo() {
    assert_image_parity("quadric_zoo.rib");
}

// ---------------------------------------------------------------------------
// Determinism: no atomics, per-pixel writes — output must be bit-stable.

#[test]
fn deterministic_output() {
    let scene = load_fixture_scene("scene.rib", 160, 120);
    let a = metal::render(&scene).unwrap();
    let b = metal::render(&scene).unwrap();
    for (ra, rb) in a.iter().zip(b.iter()) {
        for (pa, pb) in ra.iter().zip(rb.iter()) {
            assert!(
                pa.x == pb.x && pa.y == pb.y && pa.z == pb.z,
                "non-deterministic output"
            );
        }
    }
}
