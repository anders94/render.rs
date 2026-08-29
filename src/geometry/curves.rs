//! Curves and Points (roadmap Phase 8). Cubic/linear curves dice into
//! chains of *rounded cones* — line segments with linearly varying radius
//! and spherical caps — which are view-independent (unlike camera-facing
//! ribbons), cheap to intersect, and good enough for hair at production
//! strand counts. Points are the degenerate case (zero-length segments =
//! spheres), so one primitive type serves both.
//!
//! Storage is GPU-shaped (flat f32 arrays) like Mesh; a per-set BLAS over
//! segment bounds plugs into the same TLAS instancing as meshes.

use crate::accel::{Aabb, Bvh};
use crate::math::{Point3, Vec3};
use crate::raytracer::Ray;

pub struct CurveSet {
    /// Segment start: xyz + radius.
    pub p0: Vec<[f32; 4]>,
    /// Segment end: xyz + radius.
    pub p1: Vec<[f32; 4]>,
    /// Curve parameter v at each segment's start/end (0 root, 1 tip) —
    /// becomes st = [0.5, v] at hits.
    pub v0: Vec<f32>,
    pub v1: Vec<f32>,
    pub blas: Bvh,
    pub local_bounds: Aabb,
}

impl CurveSet {
    pub fn new(p0: Vec<[f32; 4]>, p1: Vec<[f32; 4]>, v0: Vec<f32>, v1: Vec<f32>) -> Self {
        let bounds: Vec<Aabb> = p0
            .iter()
            .zip(&p1)
            .map(|(a, b)| {
                let mut bb = Aabb::empty();
                bb.grow_point(&Vec3::new(
                    a[0] as f64 - a[3] as f64,
                    a[1] as f64 - a[3] as f64,
                    a[2] as f64 - a[3] as f64,
                ));
                bb.grow_point(&Vec3::new(
                    a[0] as f64 + a[3] as f64,
                    a[1] as f64 + a[3] as f64,
                    a[2] as f64 + a[3] as f64,
                ));
                bb.grow_point(&Vec3::new(
                    b[0] as f64 - b[3] as f64,
                    b[1] as f64 - b[3] as f64,
                    b[2] as f64 - b[3] as f64,
                ));
                bb.grow_point(&Vec3::new(
                    b[0] as f64 + b[3] as f64,
                    b[1] as f64 + b[3] as f64,
                    b[2] as f64 + b[3] as f64,
                ));
                bb
            })
            .collect();
        let mut local_bounds = Aabb::empty();
        for b in &bounds {
            local_bounds.grow(b);
        }
        let blas = Bvh::build(&bounds);
        Self { p0, p1, v0, v1, blas, local_bounds }
    }

    pub fn segment_count(&self) -> usize {
        self.p0.len()
    }

    /// Closest rounded-cone hit for segment `seg`, parametric t along the
    /// (unnormalized) ray direction. Returns (t, normal, tangent, v).
    pub fn intersect_segment(
        &self,
        ray: &Ray,
        seg: u32,
        t_max: f64,
    ) -> Option<(f64, Vec3, Vec3, f64)> {
        let a = self.p0[seg as usize];
        let b = self.p1[seg as usize];
        let pa = Point3::new(a[0] as f64, a[1] as f64, a[2] as f64);
        let pb = Point3::new(b[0] as f64, b[1] as f64, b[2] as f64);
        let ra = a[3] as f64;
        let rb = b[3] as f64;
        let (t, n) = rounded_cone_intersect(ray, pa, ra, pb, rb, t_max)?;
        let axis = pb - pa;
        let len2 = axis.length_squared();
        let hit = ray.at(t);
        let v_along = if len2 > 1e-24 {
            ((hit - pa).dot(&axis) / len2).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let v = self.v0[seg as usize] as f64
            + (self.v1[seg as usize] - self.v0[seg as usize]) as f64 * v_along;
        let tangent = if len2 > 1e-24 {
            axis.normalize()
        } else {
            // Point particle: any tangent will do; make it stable.
            Vec3::new(0.0, 0.0, 1.0)
        };
        Some((t, n, tangent, v))
    }
}

/// Ray vs rounded cone (sphere-swept segment with lerped radius), after
/// Quilez. Ray direction need not be normalized: t is parametric.
/// Returns (t, outward normal).
fn rounded_cone_intersect(
    ray: &Ray,
    pa: Point3,
    ra: f64,
    pb: Point3,
    rb: f64,
    t_max: f64,
) -> Option<(f64, Vec3)> {
    let ba = pb - pa;
    let oa = ray.origin - pa;
    let ob = ray.origin - pb;
    let rr = ra - rb;
    let m0 = ba.dot(&ba);
    let m1 = ba.dot(&oa);
    let m2 = ba.dot(&ray.direction);
    let m3 = ray.direction.dot(&oa);
    let m5 = oa.dot(&oa);
    let m6 = ob.dot(&ray.direction);
    let m7 = ob.dot(&ob);
    let dd = ray.direction.dot(&ray.direction);

    if m0 < 1e-24 {
        // Degenerate: a sphere of radius max(ra, rb).
        return sphere_hit(ray, pa, ra.max(rb), t_max);
    }

    let d2 = m0 - rr * rr;
    let k2 = d2 * dd - m2 * m2;
    let k1 = d2 * m3 - m1 * m2 + m2 * rr * ra;
    let k0 = d2 * m5 - m1 * m1 + m1 * rr * ra * 2.0 - m0 * ra * ra;

    if k2.abs() > 1e-24 {
        let h = k1 * k1 - k0 * k2;
        if h >= 0.0 {
            let t = (-k1 - h.sqrt()) / k2;
            let y = m1 + t * m2;
            if t > 1e-9 && t < t_max && y > 0.0 && y < d2 {
                // Quilez's closed-form cone-body normal (taper included).
                let n = ((oa + ray.direction * t) * d2 - ba * y).normalize();
                return Some((t, n));
            }
        }
    }

    // Spherical caps: start cap owns y < 0, end cap owns y > d2.
    let _ = (m6, m7);
    let mut best: Option<(f64, Vec3)> = None;
    if let Some((t, n)) = sphere_hit(ray, pa, ra, t_max) {
        let y = m1 + t * m2;
        if y <= 0.0 {
            best = Some((t, n));
        }
    }
    let t_lim = best.map(|(t, _)| t).unwrap_or(t_max);
    if let Some((t, n)) = sphere_hit(ray, pb, rb, t_lim) {
        let y = m1 + t * m2;
        if y >= d2 {
            best = Some((t, n));
        }
    }
    best
}

fn sphere_hit(ray: &Ray, center: Point3, radius: f64, t_max: f64) -> Option<(f64, Vec3)> {
    let oc = ray.origin - center;
    let a = ray.direction.dot(&ray.direction);
    let b = oc.dot(&ray.direction);
    let c = oc.dot(&oc) - radius * radius;
    let disc = b * b - a * c;
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    let mut t = (-b - s) / a;
    if t <= 1e-9 {
        t = (-b + s) / a;
    }
    if t <= 1e-9 || t >= t_max {
        return None;
    }
    let n = (ray.at(t) - center).normalize();
    Some((t, n))
}

/// Dice one curve's control points into rounded-cone segments.
/// `basis`/`step` follow the RiSpec Basis request (cubic only); linear
/// curves pass `None`. Widths interpolate root->tip along the curve.
#[allow(clippy::too_many_arguments)]
pub fn dice_curve(
    ctrl: &[[f64; 3]],
    cubic: Option<(&crate::geometry::patches::Basis4, usize)>,
    width_root: f64,
    width_tip: f64,
    segs_per_span: usize,
    p0: &mut Vec<[f32; 4]>,
    p1: &mut Vec<[f32; 4]>,
    v0: &mut Vec<f32>,
    v1: &mut Vec<f32>,
) {
    // Sample points along the curve (position + global v in [0,1]).
    let mut samples: Vec<[f64; 3]> = Vec::new();
    match cubic {
        None => samples.extend_from_slice(ctrl),
        Some((basis, step)) => {
            let n = ctrl.len();
            if n < 4 {
                return;
            }
            let spans = (n - 4) / step + 1;
            for span in 0..spans {
                let g0 = ctrl[span * step];
                let g1 = ctrl[span * step + 1];
                let g2 = ctrl[span * step + 2];
                let g3 = ctrl[span * step + 3];
                let last = span == spans - 1;
                let count = if last { segs_per_span + 1 } else { segs_per_span };
                for i in 0..count {
                    let u = i as f64 / segs_per_span as f64;
                    // p(u) = [u^3 u^2 u 1] * M * G  per axis.
                    let uv = [u * u * u, u * u, u, 1.0];
                    let mut w = [0.0f64; 4];
                    for (r, wr) in w.iter_mut().enumerate() {
                        *wr = uv[0] * basis[0][r]
                            + uv[1] * basis[1][r]
                            + uv[2] * basis[2][r]
                            + uv[3] * basis[3][r];
                    }
                    let mut p = [0.0f64; 3];
                    for a in 0..3 {
                        p[a] = w[0] * g0[a] + w[1] * g1[a] + w[2] * g2[a] + w[3] * g3[a];
                    }
                    samples.push(p);
                }
            }
        }
    }
    if samples.len() < 2 {
        return;
    }
    let denom = (samples.len() - 1) as f64;
    for i in 0..samples.len() - 1 {
        let va = i as f64 / denom;
        let vb = (i + 1) as f64 / denom;
        let ra = width_root + (width_tip - width_root) * va;
        let rb = width_root + (width_tip - width_root) * vb;
        let a = samples[i];
        let b = samples[i + 1];
        p0.push([a[0] as f32, a[1] as f32, a[2] as f32, (ra * 0.5) as f32]);
        p1.push([b[0] as f32, b[1] as f32, b[2] as f32, (rb * 0.5) as f32]);
        v0.push(va as f32);
        v1.push(vb as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::patches::BEZIER;
    use crate::math::Vec3;

    #[test]
    fn capsule_hit_and_normal() {
        // Horizontal segment along x at origin, radius 0.5 both ends.
        let set = CurveSet::new(
            vec![[-1.0, 0.0, 0.0, 0.5]],
            vec![[1.0, 0.0, 0.0, 0.5]],
            vec![0.0],
            vec![1.0],
        );
        let ray = Ray::new(Point3::new(0.2, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        let (t, n, tang, v) = set.intersect_segment(&ray, 0, f64::INFINITY).expect("hit");
        assert!((t - 4.5).abs() < 1e-9, "t = {t}");
        assert!(n.z < -0.99, "normal {n:?}");
        assert!(tang.x.abs() > 0.99, "tangent {tang:?}");
        assert!(v > 0.5 && v < 0.7, "v = {v}");
        // Cap hit past the end.
        let ray2 = Ray::new(Point3::new(1.4, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        let (_, n2, _, _) = set.intersect_segment(&ray2, 0, f64::INFINITY).expect("cap hit");
        assert!(n2.x > 0.3, "cap normal {n2:?}");
        // Clean miss.
        let ray3 = Ray::new(Point3::new(0.0, 2.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(set.intersect_segment(&ray3, 0, f64::INFINITY).is_none());
    }

    #[test]
    fn tapered_cone_radius() {
        // Radius 0.5 -> 0.1: at x=0 the surface sits near y ~ 0.3.
        let set = CurveSet::new(
            vec![[-1.0, 0.0, 0.0, 0.5]],
            vec![[1.0, 0.0, 0.0, 0.1]],
            vec![0.0],
            vec![1.0],
        );
        let ray = Ray::new(Point3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let (t, _, _, _) = set.intersect_segment(&ray, 0, f64::INFINITY).expect("hit");
        let y = 5.0 - t;
        assert!(y > 0.25 && y < 0.36, "surface at y = {y}");
    }

    #[test]
    fn dice_bezier_curve() {
        let ctrl = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
        ];
        let mut p0 = Vec::new();
        let mut p1 = Vec::new();
        let mut v0 = Vec::new();
        let mut v1 = Vec::new();
        dice_curve(&ctrl, Some((&BEZIER, 3)), 0.1, 0.02, 8, &mut p0, &mut p1, &mut v0, &mut v1);
        assert_eq!(p0.len(), 8);
        // Endpoints interpolate the bezier ends.
        assert!((p0[0][0] - 0.0).abs() < 1e-6 && (p0[0][1] - 0.0).abs() < 1e-6);
        assert!((p1[7][0] - 2.0).abs() < 1e-6 && (p1[7][1] - 2.0).abs() < 1e-6);
        // Root wider than tip (half-widths).
        assert!(p0[0][3] > p1[7][3]);
        assert!((p0[0][3] - 0.05).abs() < 1e-6);
        // v runs 0 -> 1.
        assert_eq!(v0[0], 0.0);
        assert_eq!(v1[7], 1.0);
    }

    #[test]
    fn point_particle_is_sphere() {
        let set = CurveSet::new(
            vec![[0.0, 0.0, 5.0, 0.25]],
            vec![[0.0, 0.0, 5.0, 0.25]],
            vec![0.0],
            vec![1.0],
        );
        let ray = Ray::new(Point3::origin(), Vec3::new(0.0, 0.0, 1.0));
        let (t, n, _, _) = set.intersect_segment(&ray, 0, f64::INFINITY).expect("hit");
        assert!((t - 4.75).abs() < 1e-9, "t = {t}");
        assert!(n.z < -0.99);
    }
}
