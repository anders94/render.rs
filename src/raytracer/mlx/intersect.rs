//! Batched (per-ray-array) intersection math, transcribed from the CPU
//! primitives in src/geometry/. Conventions preserved exactly: local ray
//! direction is normalized, t is the world-space distance from the ray
//! origin to the hit, normals go through the inverse-transpose.
//!
//! Lanes masked out by `valid` may still compute garbage (guarded against
//! NaN); every consumer must select through a mask.

use super::scene_arrays::{FlatLight, FlatLightKind, FlatObject, FlatScene};
use super::{SHADOW_EPS, T_EPS};
use crate::geometry::PrimitiveKind;
use anyhow::Result;
use mlx_rs::{ops, Array};

/// Batch of 3D vectors as separate component arrays (scalars broadcast).
#[derive(Clone)]
pub struct V3B {
    pub x: Array,
    pub y: Array,
    pub z: Array,
}

fn scalar(v: f32) -> Array {
    Array::from_f32(v)
}

impl V3B {
    pub fn constant(v: [f32; 3]) -> Self {
        Self {
            x: scalar(v[0]),
            y: scalar(v[1]),
            z: scalar(v[2]),
        }
    }

    pub fn zero() -> Self {
        Self::constant([0.0, 0.0, 0.0])
    }

    pub fn dot(&self, other: &V3B) -> Array {
        &self.x * &other.x + &self.y * &other.y + &self.z * &other.z
    }

    pub fn add(&self, other: &V3B) -> V3B {
        V3B {
            x: &self.x + &other.x,
            y: &self.y + &other.y,
            z: &self.z + &other.z,
        }
    }

    pub fn sub(&self, other: &V3B) -> V3B {
        V3B {
            x: &self.x - &other.x,
            y: &self.y - &other.y,
            z: &self.z - &other.z,
        }
    }

    /// Componentwise product (used for color math).
    pub fn mul(&self, other: &V3B) -> V3B {
        V3B {
            x: &self.x * &other.x,
            y: &self.y * &other.y,
            z: &self.z * &other.z,
        }
    }

    pub fn scale(&self, s: &Array) -> V3B {
        V3B {
            x: &self.x * s,
            y: &self.y * s,
            z: &self.z * s,
        }
    }

    pub fn scale_f(&self, s: f32) -> V3B {
        V3B {
            x: &self.x * s,
            y: &self.y * s,
            z: &self.z * s,
        }
    }

    pub fn length(&self) -> Result<Array> {
        Ok(ops::sqrt(&self.dot(self))?)
    }

    /// Matches Vec3::normalize: divide only where length > 1e-6.
    pub fn normalized(&self) -> Result<V3B> {
        let len = self.length()?;
        let safe = ops::maximum(&len, scalar(1e-20))?;
        let scaled = self.scale(&(scalar(1.0) / &safe));
        let big = len.gt(scalar(1e-6))?;
        V3B::select(&big, &scaled, self)
    }

    pub fn select(mask: &Array, a: &V3B, b: &V3B) -> Result<V3B> {
        Ok(V3B {
            x: ops::r#where(mask, &a.x, &b.x)?,
            y: ops::r#where(mask, &a.y, &b.y)?,
            z: ops::r#where(mask, &a.z, &b.z)?,
        })
    }

    /// Affine point transform by a row-major matrix (w assumed 1; asserted
    /// affine during flattening).
    pub fn transform_point(&self, m: &[f32; 16]) -> V3B {
        V3B {
            x: &self.x * m[0] + &self.y * m[1] + &self.z * m[2] + m[3],
            y: &self.x * m[4] + &self.y * m[5] + &self.z * m[6] + m[7],
            z: &self.x * m[8] + &self.y * m[9] + &self.z * m[10] + m[11],
        }
    }

    pub fn transform_vec(&self, m: &[f32; 16]) -> V3B {
        V3B {
            x: &self.x * m[0] + &self.y * m[1] + &self.z * m[2],
            y: &self.x * m[4] + &self.y * m[5] + &self.z * m[6],
            z: &self.x * m[8] + &self.y * m[9] + &self.z * m[10],
        }
    }

    /// Normal transform: multiply by the transpose of the given (inverse)
    /// matrix's upper 3x3, matching Matrix4::transform_normal.
    pub fn transform_normal_by_inv(&self, inv: &[f32; 16]) -> V3B {
        V3B {
            x: &self.x * inv[0] + &self.y * inv[4] + &self.z * inv[8],
            y: &self.x * inv[1] + &self.y * inv[5] + &self.z * inv[9],
            z: &self.x * inv[2] + &self.y * inv[6] + &self.z * inv[10],
        }
    }
}

/// Result of intersecting one object with a batch of rays.
pub struct Hit {
    pub valid: Array,
    /// World-space distance from the ray origin (the CPU t convention).
    pub t: Array,
    pub p: V3B,
    pub n: V3B,
}

/// Merged closest-hit state over all objects.
pub struct SceneHits {
    pub t: Array,
    pub p: V3B,
    pub n: V3B,
    pub obj_idx: Array,
    pub hit_any: Array,
}

/// phi in [0, 360) from local x/y, then `phi <= thetamax`.
fn theta_ok(px: &Array, py: &Array, thetamax: f32) -> Result<Array> {
    let phi = ops::degrees(&ops::atan2(py, px)?)?;
    let phi = ops::r#where(&phi.lt(scalar(0.0))?, &(&phi + 360.0f32), &phi)?;
    Ok(phi.le(scalar(thetamax))?)
}

/// World-space hit point, t (distance to ray origin), and normal from a
/// local hit point and local normal.
fn to_world(
    ray_origin: &V3B,
    local_p: &V3B,
    local_n: &V3B,
    obj: &FlatObject,
) -> Result<(Array, V3B, V3B)> {
    let wp = local_p.transform_point(&obj.fwd);
    let t_world = wp.sub(ray_origin).length()?;
    let wn = local_n.transform_normal_by_inv(&obj.inv).normalized()?;
    Ok((t_world, wp, wn))
}

fn intersect_sphere(
    origin_w: &V3B,
    lo: &V3B,
    ld: &V3B,
    obj: &FlatObject,
    radius: f32,
    zmin: f32,
    zmax: f32,
    thetamax: f32,
) -> Result<Hit> {
    // a = d.d (≈1 after normalization, kept for parity), b = 2 oc.d, c = oc.oc - r²
    let a = ld.dot(ld);
    let b = lo.dot(ld) * 2.0f32;
    let c = lo.dot(lo) - radius * radius;

    let disc = &b * &b - (&a * &c) * 4.0f32;
    let has = disc.ge(scalar(0.0))?;
    let sq = ops::sqrt(&ops::maximum(&disc, scalar(0.0))?)?;
    // CPU uses only the near root; inside-sphere rays miss.
    let t1 = (-&b - &sq) / (&a * 2.0f32);

    let mut valid = ops::logical_and(&has, &t1.gt(scalar(T_EPS))?)?;

    let p = lo.add(&ld.scale(&t1));
    valid = ops::logical_and(&valid, &p.z.ge(scalar(zmin))?)?;
    valid = ops::logical_and(&valid, &p.z.le(scalar(zmax))?)?;
    if thetamax < 360.0 {
        valid = ops::logical_and(&valid, &theta_ok(&p.x, &p.y, thetamax)?)?;
    }

    let local_n = p.normalized()?;
    let (t, wp, wn) = to_world(origin_w, &p, &local_n, obj)?;
    Ok(Hit { valid, t, p: wp, n: wn })
}

fn intersect_cylinder(
    origin_w: &V3B,
    lo: &V3B,
    ld: &V3B,
    obj: &FlatObject,
    radius: f32,
    zmin: f32,
    zmax: f32,
    thetamax: f32,
) -> Result<Hit> {
    let a = &ld.x * &ld.x + &ld.y * &ld.y;
    let b = (&lo.x * &ld.x + &lo.y * &ld.y) * 2.0f32;
    let c = &lo.x * &lo.x + &lo.y * &lo.y - radius * radius;

    let nondegenerate = ops::abs(&a)?.ge(scalar(1e-6))?;
    let a_safe = ops::maximum(&a, scalar(1e-12))?; // a >= 0 (sum of squares)

    let disc = &b * &b - (&a * &c) * 4.0f32;
    let has = disc.ge(scalar(0.0))?;
    let sq = ops::sqrt(&ops::maximum(&disc, scalar(0.0))?)?;
    let t1 = (-&b - &sq) / (&a_safe * 2.0f32);
    let t2 = (-&b + &sq) / (&a_safe * 2.0f32);

    // CPU: t = t1 if t1 > EPS else t2; if z-clip fails and t was t1, retry t2.
    let z_at = |t: &Array| -> Array { &lo.z + &ld.z * t };
    let z_ok = |t: &Array| -> Result<Array> {
        let z = z_at(t);
        Ok(ops::logical_and(&z.ge(scalar(zmin))?, &z.le(scalar(zmax))?)?)
    };

    let first_is_t1 = t1.gt(scalar(T_EPS))?;
    let t_first = ops::r#where(&first_is_t1, &t1, &t2)?;
    let t_first_pos = t_first.gt(scalar(T_EPS))?;
    let z_ok_first = z_ok(&t_first)?;
    let use_second = ops::logical_and(
        &ops::logical_and(&first_is_t1, &ops::logical_not(&z_ok_first)?)?,
        &ops::logical_and(&t2.gt(scalar(T_EPS))?, &z_ok(&t2)?)?,
    )?;
    let t = ops::r#where(&z_ok_first, &t_first, &t2)?;

    let mut valid = ops::logical_and(&nondegenerate, &has)?;
    valid = ops::logical_and(&valid, &t_first_pos)?;
    valid = ops::logical_and(&valid, &ops::logical_or(&z_ok_first, &use_second)?)?;

    let p = lo.add(&ld.scale(&t));
    if thetamax < 360.0 {
        valid = ops::logical_and(&valid, &theta_ok(&p.x, &p.y, thetamax)?)?;
    }

    let local_n = V3B {
        x: p.x.clone(),
        y: p.y.clone(),
        z: ops::zeros_like(&p.x)?,
    }
    .normalized()?;
    let (t_world, wp, wn) = to_world(origin_w, &p, &local_n, obj)?;
    Ok(Hit { valid, t: t_world, p: wp, n: wn })
}

fn intersect_cone(
    origin_w: &V3B,
    lo: &V3B,
    ld: &V3B,
    obj: &FlatObject,
    height: f32,
    radius: f32,
    thetamax: f32,
) -> Result<Hit> {
    let k = (radius / height) * (radius / height);

    let a = &ld.x * &ld.x + &ld.y * &ld.y - (&ld.z * &ld.z) * k;
    let b = (&lo.x * &ld.x + &lo.y * &ld.y - (&lo.z * &ld.z) * k) * 2.0f32;
    let c = &lo.x * &lo.x + &lo.y * &lo.y - (&lo.z * &lo.z) * k;

    let degenerate = ops::abs(&a)?.lt(scalar(1e-6))?;

    // Degenerate linear path: t = -c/b, z-clipped, no theta check (CPU quirk).
    let b_ok = ops::abs(&b)?.ge(scalar(1e-6))?;
    let b_safe = ops::r#where(&b_ok, &b, &scalar(1.0))?;
    let t_lin = -&c / &b_safe;
    let z_lin = &lo.z + &ld.z * &t_lin;
    let lin_valid = ops::logical_and(
        &ops::logical_and(&b_ok, &t_lin.gt(scalar(T_EPS))?)?,
        &ops::logical_and(&z_lin.ge(scalar(0.0))?, &z_lin.le(scalar(height))?)?,
    )?;

    // Quadratic path: try t1 then t2; z-clip AND theta both retry to t2.
    let a_safe = ops::r#where(&degenerate, &scalar(1.0), &a)?;
    let disc = &b * &b - (&a * &c) * 4.0f32;
    let has = disc.ge(scalar(0.0))?;
    let sq = ops::sqrt(&ops::maximum(&disc, scalar(0.0))?)?;
    let t1 = (-&b - &sq) / (&a_safe * 2.0f32);
    let t2 = (-&b + &sq) / (&a_safe * 2.0f32);

    let candidate_ok = |t: &Array| -> Result<Array> {
        let p = lo.add(&ld.scale(t));
        let mut ok = ops::logical_and(&t.gt(scalar(T_EPS))?, &p.z.ge(scalar(0.0))?)?;
        ok = ops::logical_and(&ok, &p.z.le(scalar(height))?)?;
        if thetamax < 360.0 {
            ok = ops::logical_and(&ok, &theta_ok(&p.x, &p.y, thetamax)?)?;
        }
        Ok(ok)
    };
    let ok1 = candidate_ok(&t1)?;
    let ok2 = candidate_ok(&t2)?;
    let t_quad = ops::r#where(&ok1, &t1, &t2)?;
    let quad_valid = ops::logical_and(&has, &ops::logical_or(&ok1, &ok2)?)?;

    let t = ops::r#where(&degenerate, &t_lin, &t_quad)?;
    let valid = ops::r#where(&degenerate, &lin_valid, &quad_valid)?;

    let p = lo.add(&ld.scale(&t));

    // Normal: (px/r_xy, py/r_xy, -radius/height) normalized — the exact
    // quadric gradient, as in Cone::create_intersection.
    let r_xy = ops::maximum(
        &ops::sqrt(&(&p.x * &p.x + &p.y * &p.y))?,
        &scalar(1e-20),
    )?;
    let local_n = V3B {
        x: &p.x / &r_xy,
        y: &p.y / &r_xy,
        z: ops::full::<f32>(&[1], &scalar(-(radius / height)))?,
    }
    .normalized()?;
    let (t_world, wp, wn) = to_world(origin_w, &p, &local_n, obj)?;
    Ok(Hit { valid, t: t_world, p: wp, n: wn })
}

/// Intersect one object with a batch of world-space rays.
pub fn intersect_object(origin: &V3B, direction: &V3B, obj: &FlatObject) -> Result<Hit> {
    let lo = origin.transform_point(&obj.inv);
    let ld = direction.transform_vec(&obj.inv).normalized()?;

    match obj.kind {
        PrimitiveKind::Sphere { radius, zmin, zmax, thetamax } => intersect_sphere(
            origin,
            &lo,
            &ld,
            obj,
            radius as f32,
            zmin as f32,
            zmax as f32,
            thetamax as f32,
        ),
        PrimitiveKind::Cylinder { radius, zmin, zmax, thetamax } => intersect_cylinder(
            origin,
            &lo,
            &ld,
            obj,
            radius as f32,
            zmin as f32,
            zmax as f32,
            thetamax as f32,
        ),
        PrimitiveKind::Cone { height, radius, thetamax } => intersect_cone(
            origin,
            &lo,
            &ld,
            obj,
            height as f32,
            radius as f32,
            thetamax as f32,
        ),
        // MLX backend is frozen (see ROADMAP.md): primitives added after the
        // freeze never hit here.
        _ => Ok(Hit {
            valid: Array::from_bool(false),
            t: scalar(f32::INFINITY),
            p: V3B::zero(),
            n: V3B::zero(),
        }),
    }
}

/// Closest hit over all scene objects: running elementwise minimum, so
/// memory stays O(rays) regardless of object count.
pub fn intersect_scene(flat: &FlatScene, origin: &V3B, direction: &V3B, n: i32) -> Result<SceneHits> {
    let mut t = ops::full::<f32>(&[n], &scalar(f32::INFINITY))?;
    let mut p = V3B::zero();
    let mut normal = V3B::zero();
    let mut obj_idx = ops::zeros::<i32>(&[n])?;
    let mut hit_any = ops::zeros::<bool>(&[n])?;

    for (i, obj) in flat.objects.iter().enumerate() {
        let h = intersect_object(origin, direction, obj)?;
        let closer = ops::logical_and(&h.valid, &h.t.lt(&t)?)?;
        t = ops::r#where(&closer, &h.t, &t)?;
        p = V3B::select(&closer, &h.p, &p)?;
        normal = V3B::select(&closer, &h.n, &normal)?;
        obj_idx = ops::r#where(&closer, &Array::from_int(i as i32), &obj_idx)?;
        hit_any = ops::logical_or(&hit_any, &h.valid)?;
    }

    Ok(SceneHits { t, p, n: normal, obj_idx, hit_any })
}

/// Direction toward the light and distance to it, per shading point.
pub fn light_dir_dist(light: &FlatLight, p: &V3B) -> Result<(V3B, Array)> {
    match &light.kind {
        FlatLightKind::Point { position } => {
            let to_light = V3B::constant(*position).sub(p);
            let dist = to_light.length()?;
            Ok((to_light.normalized()?, dist))
        }
        FlatLightKind::Distant { direction } => Ok((
            V3B::constant([-direction[0], -direction[1], -direction[2]]),
            scalar(f32::INFINITY),
        )),
    }
}

/// Per-ray shadow test, mirroring Scene::is_occluded.
pub fn occluded(flat: &FlatScene, p: &V3B, n: &V3B, light: &FlatLight) -> Result<Array> {
    let origin = p.add(&n.scale_f(SHADOW_EPS));
    let (ldir, max_t) = light_dir_dist(light, p)?;
    let limit = &max_t - SHADOW_EPS;

    let mut occ = Array::from_bool(false);
    for obj in &flat.objects {
        let h = intersect_object(&origin, &ldir, obj)?;
        let blocks = ops::logical_and(&h.valid, &h.t.lt(&limit)?)?;
        occ = ops::logical_or(&occ, &blocks)?;
    }
    Ok(occ)
}
