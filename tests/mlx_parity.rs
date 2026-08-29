//! Parity tests for the MLX GPU backend: run with
//! `cargo test --features mlx`.

#![cfg(feature = "mlx")]

mod common;

use common::{
    assert_quantized_parity, flat_object_for, load_fixture_scene, quantize, random_rays, Lcg,
};
use render_rs::geometry::{Cone, Cylinder, Intersectable, Sphere};
use render_rs::math::Matrix4;
use render_rs::parser::{parse_rib, SceneBuilder};
use render_rs::raytracer::mlx;
use render_rs::raytracer::mlx::intersect::{intersect_object, V3B};
use render_rs::raytracer::mlx::scene_arrays::FlatScene;
use render_rs::raytracer::renderer::render as render_cpu;
use render_rs::raytracer::Ray;
use std::fs;
use std::sync::Mutex;

/// MLX is not safe to drive from multiple test threads at once; serialize.
static MLX_LOCK: Mutex<()> = Mutex::new(());

fn mlx_guard() -> std::sync::MutexGuard<'static, ()> {
    MLX_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

// ---------------------------------------------------------------------------
// mlx-rs op smoke test: every op the renderer relies on, in one place, so
// API drift in future mlx-rs versions fails loudly.
#[test]
fn mlx_ops_smoke() {
    use mlx_rs::{ops, transforms, Array};
    let _guard = mlx_guard();

    let a = Array::from_slice(&[1.0f32, -2.0, 3.0], &[3]);
    let b = Array::from_slice(&[2.0f32, 2.0, 2.0], &[3]);

    let _ = ops::sqrt(&ops::maximum(&a, Array::from_f32(0.0)).unwrap()).unwrap();
    let _ = ops::minimum(&a, &b).unwrap();
    let _ = ops::abs(&a).unwrap();
    let _ = ops::atan2(&a, &b).unwrap();
    let _ = ops::degrees(&a).unwrap();
    let _ = ops::power(&ops::abs(&a).unwrap(), &b).unwrap();
    let lt = a.lt(&b).unwrap();
    let le = a.le(&b).unwrap();
    let _ = a.gt(&b).unwrap();
    let _ = a.ge(&b).unwrap();
    let and = ops::logical_and(&lt, &le).unwrap();
    let _ = ops::logical_or(&lt, &le).unwrap();
    let not = ops::logical_not(&and).unwrap();
    let sel = ops::r#where(&not, &a, &b).unwrap();
    let idx = Array::from_slice(&[0i32, 2, 1], &[3]);
    let took = sel.take(&idx).unwrap();
    let _ = ops::zeros::<i32>(&[3]).unwrap();
    let _ = ops::zeros::<bool>(&[3]).unwrap();
    let _ = ops::zeros_like(&a).unwrap();
    let _ = ops::full::<f32>(&[3], &Array::from_f32(1.5)).unwrap();
    let reshaped = took.reshape(&[3, 1]).unwrap();
    let mean = reshaped.mean_axes(&[1], None).unwrap();
    transforms::eval([&mean]).unwrap();
    let s: &[f32] = mean.as_slice();
    assert_eq!(s.len(), 3);

    // operator overloads used throughout
    let c = &a * 2.0f32 + &b - 1.0f32;
    let d = -&c / &b;
    transforms::eval([&d]).unwrap();
}

// ---------------------------------------------------------------------------
// Primitive-level parity: batched intersects vs the CPU implementations.

fn assert_primitive_parity(object: &dyn Intersectable, rays: &[Ray], label: &str) {
    use mlx_rs::transforms;

    let _guard = mlx_guard();
    let n = rays.len();
    let fx: Vec<f32> = rays.iter().map(|r| r.origin.x as f32).collect();
    let fy: Vec<f32> = rays.iter().map(|r| r.origin.y as f32).collect();
    let fz: Vec<f32> = rays.iter().map(|r| r.origin.z as f32).collect();
    let dx: Vec<f32> = rays.iter().map(|r| r.direction.x as f32).collect();
    let dy: Vec<f32> = rays.iter().map(|r| r.direction.y as f32).collect();
    let dz: Vec<f32> = rays.iter().map(|r| r.direction.z as f32).collect();
    let shape = [n as i32];
    let origin = V3B {
        x: mlx_rs::Array::from_slice(&fx, &shape),
        y: mlx_rs::Array::from_slice(&fy, &shape),
        z: mlx_rs::Array::from_slice(&fz, &shape),
    };
    let direction = V3B {
        x: mlx_rs::Array::from_slice(&dx, &shape),
        y: mlx_rs::Array::from_slice(&dy, &shape),
        z: mlx_rs::Array::from_slice(&dz, &shape),
    };

    let flat = flat_object_for(object);
    let hit = intersect_object(&origin, &direction, &flat).unwrap();
    transforms::eval([&hit.valid, &hit.t]).unwrap();
    let valid: &[bool] = hit.valid.as_slice();
    let ts: &[f32] = hit.t.as_slice();

    let mut mismatches = 0usize;
    for (i, ray) in rays.iter().enumerate() {
        let cpu = object.intersect(ray);
        match (cpu, valid[i]) {
            (Some(cpu_hit), true) => {
                let rel = ((cpu_hit.t - ts[i] as f64) / cpu_hit.t.max(1e-6)).abs();
                if rel > 1e-3 {
                    mismatches += 1;
                }
            }
            (None, false) => {}
            // f32 vs f64 can legitimately flip razor-edge hits; count them.
            _ => mismatches += 1,
        }
    }
    let ratio = mismatches as f64 / n as f64;
    assert!(
        ratio < 0.01,
        "{label}: {mismatches}/{n} rays disagree with CPU (allowing <1% edge flips)"
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

// ---------------------------------------------------------------------------
// Scene flattening sanity.

#[test]
fn flatten_scene() {
    let rib = fs::read_to_string("tests/fixtures/scene.rib").unwrap();
    let commands = parse_rib(&rib).unwrap();
    let scene = SceneBuilder::new().build(&commands).unwrap();
    let flat = FlatScene::from_scene(&scene).unwrap();
    assert_eq!(flat.objects.len(), 4);
    assert_eq!(flat.lights.len(), 2);
    assert_eq!(flat.pixel_samples, (2, 2));
    assert_eq!(flat.mat_r.size(), 4);
}

// ---------------------------------------------------------------------------
// Full-image parity over the fixture scenes.

fn assert_image_parity(fixture: &str) {
    let _guard = mlx_guard();
    let scene = load_fixture_scene(fixture, 160, 120);

    let cpu = quantize(&render_cpu(&scene));
    let gpu = quantize(&mlx::render(&scene).unwrap());
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
