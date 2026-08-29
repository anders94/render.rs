//! Wavefront render loop: all samples for a chunk of pixels are traced as
//! one batch, reflections handled iteratively with masks and throughput.
//!
//! The graph is executed op-by-op (eval per bounce). Running it through
//! `transforms::compile` for kernel fusion was tried and abandoned: MLX
//! 0.25's fusion pass exhausts Metal argument buffers on chains this long
//! ("Too many inputs/outputs fused in the Metal Compiled primitive"), and
//! the mlx-rs `compile` wrapper re-traces on every call anyway. Revisit if
//! mlx-rs grows a custom-kernel API or the fusion limit is lifted.

use super::intersect::{intersect_scene, V3B};
use super::scene_arrays::FlatScene;
use super::shade::{gather_materials, shade};
use super::REFL_EPS;
use crate::math::Vec3;
use crate::output::Image;
use crate::scene::Scene;
use anyhow::Result;
use mlx_rs::{ops, transforms, Array};

const MAX_DEPTH: u32 = 5;
/// Target samples in flight per chunk (~20 live f32 arrays ≈ 80 MB).
const TARGET_SAMPLES: usize = 1 << 20;

fn scalar(v: f32) -> Array {
    Array::from_f32(v)
}

pub fn render(scene: &Scene) -> Result<Image> {
    let flat = FlatScene::from_scene(scene)?;
    let width = flat.width as usize;
    let height = flat.height as usize;

    let mut image = vec![vec![Vec3::zero(); width]; height];

    if flat.objects.is_empty() {
        let bg = Vec3::new(
            flat.background[0] as f64,
            flat.background[1] as f64,
            flat.background[2] as f64,
        );
        for row in &mut image {
            row.fill(bg);
        }
        return Ok(image);
    }

    let (samples_x, samples_y) = flat.pixel_samples;
    let spp = (samples_x * samples_y) as usize;
    let chunk_pixels = (TARGET_SAMPLES / spp).max(1);
    let total_pixels = width * height;

    let mut start = 0usize;
    while start < total_pixels {
        let end = (start + chunk_pixels).min(total_pixels);
        let pixel_colors = render_chunk(&flat, start, end, width, samples_x, samples_y)?;
        for (i, rgb) in pixel_colors.iter().enumerate() {
            let pix = start + i;
            image[pix / width][pix % width] =
                Vec3::new(rgb[0] as f64, rgb[1] as f64, rgb[2] as f64);
        }
        start = end;
    }

    Ok(image)
}

/// Trace all samples for pixels [start, end) and return per-pixel RGB.
fn render_chunk(
    flat: &FlatScene,
    start: usize,
    end: usize,
    width: usize,
    samples_x: u32,
    samples_y: u32,
) -> Result<Vec<[f32; 3]>> {
    let npix = end - start;
    let spp = (samples_x * samples_y) as usize;
    let n = npix * spp;

    // Subpixel sample positions, same stratification as the CPU renderer.
    let mut pxs = Vec::with_capacity(n);
    let mut pys = Vec::with_capacity(n);
    for pix in start..end {
        let x = (pix % width) as f64;
        let y = (pix / width) as f64;
        for sy in 0..samples_y {
            for sx in 0..samples_x {
                pxs.push((x + (sx as f64 + 0.5) / samples_x as f64) as f32);
                pys.push((y + (sy as f64 + 0.5) / samples_y as f64) as f32);
            }
        }
    }
    let shape = [n as i32];
    let px = Array::from_slice(&pxs, &shape);
    let py = Array::from_slice(&pys, &shape);

    // u = px/w*2-1, v = 1-py/h*2; dir = fwd + right*u*hw + up*v*hh
    // (unnormalized), exactly as Camera::generate_ray.
    let cam = &flat.camera;
    let u = &px * (2.0f32 / flat.width as f32) - 1.0f32;
    let v = -(&py * (2.0f32 / flat.height as f32)) + 1.0f32;
    let mut direction = V3B::constant(cam.forward)
        .add(&V3B::constant(cam.right).scale(&(&u * cam.half_width)))
        .add(&V3B::constant(cam.up).scale(&(&v * cam.half_height)));
    let mut origin = V3B::constant(cam.eye);

    let background = V3B::constant(flat.background);
    let white = V3B::constant([1.0, 1.0, 1.0]);

    // color = Σ throughput_i * c_i(1-r_i), throughput = Π tint_j * r_j;
    // rays that miss (or survive MAX_DEPTH) emit throughput * background.
    let mut color = V3B::zero();
    let mut throughput = white.clone();
    let mut active = ops::full::<bool>(&shape, &Array::from_bool(true))?;

    for _bounce in 0..MAX_DEPTH {
        let hits = intersect_scene(flat, &origin, &direction, n as i32)?;

        let miss = ops::logical_and(&active, &ops::logical_not(&hits.hit_any)?)?;
        color = add_where(&color, &miss, &throughput.mul(&background))?;

        let live = ops::logical_and(&active, &hits.hit_any)?;
        let mats = gather_materials(flat, &hits.obj_idx)?;
        let local = shade(flat, &hits, &mats)?;

        // color += T * c_i * (1 - r_i); T *= tint_i * r_i (metals tint).
        let reflect_weight = &mats.reflectivity;
        let one_minus_r = -reflect_weight + 1.0f32;
        color = add_where(&color, &live, &throughput.mul(&local).scale(&one_minus_r))?;

        let tint = V3B::select(&mats.is_metal, &mats.color, &white)?;
        throughput = throughput.mul(&tint).scale(reflect_weight);
        active = ops::logical_and(&live, &reflect_weight.gt(scalar(0.0))?)?;

        // Reflected ray: dir - n*2(dir·n), origin nudged along the normal.
        let d_dot_n = direction.dot(&hits.n);
        direction = direction.sub(&hits.n.scale(&(&d_dot_n * 2.0f32)));
        origin = hits.p.add(&hits.n.scale_f(REFL_EPS));

        transforms::eval([
            &color.x,
            &color.y,
            &color.z,
            &throughput.x,
            &throughput.y,
            &throughput.z,
            &active,
            &origin.x,
            &origin.y,
            &origin.z,
            &direction.x,
            &direction.y,
            &direction.z,
        ])?;
    }

    // Rays still bouncing at the depth limit resolve to background, exactly
    // as the CPU recursion does.
    color = add_where(&color, &active, &throughput.mul(&background))?;

    // Average the spp samples of each pixel.
    let mean_channel = |a: &Array| -> Result<Array> {
        Ok(a.reshape(&[npix as i32, spp as i32])?.mean_axes(&[1], None)?)
    };
    let r = mean_channel(&color.x)?;
    let g = mean_channel(&color.y)?;
    let b = mean_channel(&color.z)?;
    transforms::eval([&r, &g, &b])?;

    let rs: &[f32] = r.as_slice();
    let gs: &[f32] = g.as_slice();
    let bs: &[f32] = b.as_slice();
    Ok((0..npix).map(|i| [rs[i], gs[i], bs[i]]).collect())
}

fn add_where(base: &V3B, mask: &Array, value: &V3B) -> Result<V3B> {
    let gated = V3B::select(mask, value, &V3B::zero())?;
    Ok(base.add(&gated))
}
