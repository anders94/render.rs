//! Helpers shared by the GPU-backend parity test suites
//! (metal_parity.rs, pt_tests.rs).

#![allow(dead_code)]

use render_rs::geometry::Intersectable;
use render_rs::math::{Matrix4, Point3};
use render_rs::output::{clamp, gamma_correct, Image};
use render_rs::parser::{parse_rib, SceneBuilder};
use render_rs::raytracer::flatten::FlatObject;
use render_rs::raytracer::Ray;
use render_rs::scene::Scene;
use std::fs;

/// Deterministic LCG so tests need no rand dependency.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

pub fn random_rays(rng: &mut Lcg, count: usize) -> Vec<Ray> {
    (0..count)
        .map(|_| {
            let origin = Point3::new(
                rng.range(-4.0, 4.0),
                rng.range(-4.0, 4.0),
                rng.range(-6.0, -2.0),
            );
            let target = Point3::new(
                rng.range(-1.5, 1.5),
                rng.range(-1.5, 1.5),
                rng.range(-1.0, 2.0),
            );
            Ray::new(origin, target - origin)
        })
        .collect()
}

pub fn load_fixture_scene(fixture: &str, width: u32, height: u32) -> Scene {
    let rib = fs::read_to_string(format!("tests/fixtures/{fixture}")).unwrap();
    let commands = parse_rib(&rib).unwrap();
    let mut scene = SceneBuilder::new()
        .with_base_dir("tests/fixtures")
        .build(&commands)
        .unwrap();
    scene.camera.width = width;
    scene.camera.height = height;
    scene
}

pub fn flat_object_for(object: &dyn Intersectable) -> FlatObject {
    let desc = object.describe();
    let to16 = |m: &Matrix4| {
        let rows = m.rows();
        let mut out = [0.0f32; 16];
        for r in 0..4 {
            for c in 0..4 {
                out[r * 4 + c] = rows[r][c] as f32;
            }
        }
        out
    };
    FlatObject {
        kind: desc.kind,
        inv: to16(&desc.inverse_transform),
        fwd: to16(&desc.transform),
    }
}

/// Gamma-correct (2.2) and quantize to 8-bit, matching the image writers.
pub fn quantize(image: &Image) -> Vec<u8> {
    let mut out = Vec::new();
    for row in image {
        for pixel in row {
            let c = gamma_correct(*pixel, 2.2);
            out.push((clamp(c.x, 0.0, 1.0) * 255.0) as u8);
            out.push((clamp(c.y, 0.0, 1.0) * 255.0) as u8);
            out.push((clamp(c.z, 0.0, 1.0) * 255.0) as u8);
        }
    }
    out
}

/// f32-GPU-vs-f64-CPU tolerance: overwhelmingly identical after
/// quantization, with a small allowance for razor-edge samples at
/// silhouettes and shadow terminators.
pub fn assert_quantized_parity(cpu: &[u8], gpu: &[u8], label: &str) {
    assert_eq!(cpu.len(), gpu.len(), "{label}: image size mismatch");

    let total = cpu.len();
    let mut close = 0usize;
    let mut outliers = 0usize;
    let mut max_diff = 0u8;
    let mut sum_diff = 0u64;
    for (c, g) in cpu.iter().zip(gpu.iter()) {
        let diff = c.abs_diff(*g);
        if diff <= 1 {
            close += 1;
        }
        if diff > 8 {
            outliers += 1;
        }
        max_diff = max_diff.max(diff);
        sum_diff += diff as u64;
    }
    let close_ratio = close as f64 / total as f64;
    let outlier_ratio = outliers as f64 / total as f64;
    let mean_diff = sum_diff as f64 / total as f64;
    assert!(
        close_ratio >= 0.995,
        "{label}: only {:.2}% of channels within 1 LSB",
        close_ratio * 100.0
    );
    assert!(
        outlier_ratio <= 0.0005,
        "{label}: {outliers} channels ({:.3}%) differ by more than 8 LSB",
        outlier_ratio * 100.0
    );
    // Isolated silhouette pixels can flip hard (e.g. torus root-finding is
    // analytic f64 on CPU vs bisection f32 on GPU); the outlier-count and
    // mean bounds above carry the systematic-error signal.
    assert!(max_diff <= 128, "{label}: max channel diff {max_diff} > 128");
    assert!(mean_diff <= 0.5, "{label}: mean diff {mean_diff:.3} > 0.5");
}
