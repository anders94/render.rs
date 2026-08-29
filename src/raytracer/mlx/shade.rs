//! Batched Phong shading, transcribed from src/shading/mod.rs.

use super::intersect::{light_dir_dist, occluded, SceneHits, V3B};
use super::scene_arrays::FlatScene;
use anyhow::Result;
use mlx_rs::{ops, Array};

fn scalar(v: f32) -> Array {
    Array::from_f32(v)
}

/// Material properties gathered per ray by hit-object index.
pub struct GatheredMats {
    pub color: V3B,
    pub ka: Array,
    pub kd: Array,
    pub ks: Array,
    pub shininess: Array,
    pub reflectivity: Array,
    pub is_metal: Array,
}

pub fn gather_materials(flat: &FlatScene, obj_idx: &Array) -> Result<GatheredMats> {
    Ok(GatheredMats {
        color: V3B {
            x: flat.mat_r.take(obj_idx)?,
            y: flat.mat_g.take(obj_idx)?,
            z: flat.mat_b.take(obj_idx)?,
        },
        ka: flat.mat_ka.take(obj_idx)?,
        kd: flat.mat_kd.take(obj_idx)?,
        ks: flat.mat_ks.take(obj_idx)?,
        shininess: flat.mat_shininess.take(obj_idx)?,
        reflectivity: flat.mat_reflectivity.take(obj_idx)?,
        is_metal: flat.mat_is_metal.take(obj_idx)?,
    })
}

/// Add `value` where `mask`, keep `base` elsewhere.
fn add_where(base: &V3B, mask: &Array, value: &V3B) -> Result<V3B> {
    let gated = V3B::select(mask, value, &V3B::zero())?;
    Ok(base.add(&gated))
}

pub fn shade(flat: &FlatScene, hits: &SceneHits, mats: &GatheredMats) -> Result<V3B> {
    // Ambient
    let mut color = mats.color.scale(&mats.ka);

    // View direction is always from the camera eye, matching the CPU shader
    // (even for reflected hits).
    let view = V3B::constant(flat.camera.eye).sub(&hits.p).normalized()?;

    if flat.lights.is_empty() {
        // Headlight shading for depth perception.
        let n_dot_v = ops::maximum(&hits.n.dot(&view), scalar(0.0))?;
        color = color.add(&mats.color.scale(&(&mats.kd * &n_dot_v)));
        return Ok(color);
    }

    for light in &flat.lights {
        let (ldir, _) = light_dir_dist(light, &hits.p)?;
        let n_dot_l = hits.n.dot(&ldir);
        let front = n_dot_l.gt(scalar(0.0))?;
        let shadowed = occluded(flat, &hits.p, &hits.n, light)?;
        let lit = ops::logical_and(&front, &ops::logical_not(&shadowed)?)?;

        let light_color = V3B::constant(light.color);

        let diffuse = mats
            .color
            .mul(&light_color)
            .scale(&(&(&mats.kd * &n_dot_l) * light.intensity));
        color = add_where(&color, &lit, &diffuse)?;

        // Specular: metals tint by their color, dielectrics don't. The CPU
        // skips this only when ks == 0, where it contributes 0 anyway.
        let half = ldir.add(&view).normalized()?;
        let n_dot_h = ops::maximum(&hits.n.dot(&half), scalar(0.0))?;
        let powed = ops::power(&n_dot_h, &mats.shininess)?;
        let spec_color = V3B::select(&mats.is_metal, &mats.color.mul(&light_color), &light_color)?;
        let specular = spec_color.scale(&(&(&mats.ks * &powed) * light.intensity));
        color = add_where(&color, &lit, &specular)?;
    }

    Ok(color)
}
