//! RiSpec primitives beyond the original three quadrics: Torus, Disk,
//! Paraboloid, Hyperboloid, and object-space Triangles (from Polygon
//! fan-triangulation). Same conventions as the rest of src/geometry/:
//! rays are transformed to object space with the local direction
//! normalized, t is the world-space distance from the ray origin, and
//! normals go through the inverse-transpose.

use crate::geometry::{Intersectable, PrimitiveDesc, PrimitiveKind};
use crate::math::{Matrix4, Point3, Vec3, EPSILON};
use crate::raytracer::{Intersection, Ray};

/// Object-space ray with normalized direction (the shared prologue).
fn local_ray(inverse: &Matrix4, ray: &Ray) -> (Point3, Vec3) {
    let lo = inverse.transform_point(&ray.origin);
    let ld = inverse.transform_vec(&ray.direction).normalize();
    (lo, ld)
}

/// World-space intersection from an object-space hit (the shared epilogue).
fn world_hit(
    transform: &Matrix4,
    inverse: &Matrix4,
    ray: &Ray,
    local_p: Point3,
    local_n: Vec3,
    material_id: usize,
) -> Intersection {
    let world_p = transform.transform_point(&local_p);
    let t_world = world_p.distance(&ray.origin);
    let world_n = inverse.transform_normal(&local_n).normalize();
    Intersection::new(t_world, world_p, world_n, material_id)
}

/// phi in [0, 360) from object-space x/y; accept when phi <= thetamax.
fn theta_ok(x: f64, y: f64, thetamax: f64) -> bool {
    if thetamax >= 360.0 {
        return true;
    }
    let mut phi = y.atan2(x).to_degrees();
    if phi < 0.0 {
        phi += 360.0;
    }
    phi <= thetamax
}

// ---------------------------------------------------------------------------
// Torus

pub struct Torus {
    pub major_radius: f64,
    pub minor_radius: f64,
    pub phimin: f64,
    pub phimax: f64,
    pub thetamax: f64,
    pub material_id: usize,
    pub transform: Matrix4,
    pub inverse_transform: Matrix4,
}

impl Torus {
    pub fn new(
        major_radius: f64,
        minor_radius: f64,
        phimin: f64,
        phimax: f64,
        thetamax: f64,
        material_id: usize,
        transform: Matrix4,
    ) -> Self {
        let inverse_transform = transform.inverse().unwrap_or(Matrix4::identity());
        Self {
            major_radius,
            minor_radius,
            phimin,
            phimax,
            thetamax,
            material_id,
            transform,
            inverse_transform,
        }
    }

    fn phi_ok(&self, p: &Point3) -> bool {
        if self.phimax - self.phimin >= 360.0 {
            return true;
        }
        let r_xy = (p.x * p.x + p.y * p.y).sqrt();
        let mut phi = p.z.atan2(r_xy - self.major_radius).to_degrees();
        // Bring phi into [phimin, phimin + 360).
        while phi < self.phimin {
            phi += 360.0;
        }
        while phi >= self.phimin + 360.0 {
            phi -= 360.0;
        }
        phi <= self.phimax
    }
}

/// Real roots of a4 t^4 + a3 t^3 + a2 t^2 + a1 t + a0, ascending, found by
/// deflated Newton iteration from analytic seeds (robust enough in f64 for
/// rendering; each root is polished against the original polynomial).
fn solve_quartic(a4: f64, a3: f64, a2: f64, a1: f64, a0: f64) -> Vec<f64> {
    // Normalize to monic.
    if a4.abs() < 1e-12 {
        return solve_cubic(a3, a2, a1, a0);
    }
    let b = a3 / a4;
    let c = a2 / a4;
    let d = a1 / a4;
    let e = a0 / a4;

    // Depressed quartic: t = y - b/4 → y^4 + p y^2 + q y + r = 0
    let p = c - 3.0 * b * b / 8.0;
    let q = d - b * c / 2.0 + b * b * b / 8.0;
    let r = e - b * d / 4.0 + b * b * c / 16.0 - 3.0 * b * b * b * b / 256.0;

    let shift = -b / 4.0;
    let poly = |t: f64| (((a4 * t + a3) * t + a2) * t + a1) * t + a0;
    let dpoly = |t: f64| ((4.0 * a4 * t + 3.0 * a3) * t + 2.0 * a2) * t + a1;
    let polish = |t0: f64| {
        let mut t = t0;
        for _ in 0..8 {
            let f = poly(t);
            let df = dpoly(t);
            if df.abs() < 1e-30 {
                break;
            }
            let step = f / df;
            t -= step;
            if step.abs() < 1e-12 * t.abs().max(1.0) {
                break;
            }
        }
        t
    };

    let mut roots = Vec::with_capacity(4);
    if q.abs() < 1e-12 {
        // Biquadratic: y^4 + p y^2 + r = 0
        let disc = p * p - 4.0 * r;
        if disc >= 0.0 {
            let s = disc.sqrt();
            for y2 in [(-p - s) / 2.0, (-p + s) / 2.0] {
                if y2 >= 0.0 {
                    let y = y2.sqrt();
                    roots.push(polish(y + shift));
                    roots.push(polish(-y + shift));
                }
            }
        }
    } else {
        // Ferrari: resolvent cubic z^3 + 2p z^2 + (p^2 - 4r) z - q^2 = 0,
        // take a positive root z, then two quadratics.
        let res = solve_cubic(1.0, 2.0 * p, p * p - 4.0 * r, -q * q);
        let z = res
            .iter()
            .copied()
            .filter(|z| *z > 1e-12)
            .fold(f64::NAN, f64::max);
        if z.is_finite() {
            let m = z.sqrt();
            let n = q / (2.0 * m);
            // y^2 ± m y + (p + z)/2 ∓ n = 0
            for (sm, sn) in [(1.0, -1.0), (-1.0, 1.0)] {
                let bq = sm * m;
                let cq = (p + z) / 2.0 + sn * n;
                let disc = bq * bq - 4.0 * cq;
                if disc >= 0.0 {
                    let s = disc.sqrt();
                    roots.push(polish((-bq - s) / 2.0 + shift));
                    roots.push(polish((-bq + s) / 2.0 + shift));
                }
            }
        }
    }

    // Deduplicate and keep genuine roots only.
    roots.retain(|t| t.is_finite() && poly(*t).abs() < 1e-6 * (1.0 + t.abs().powi(4) * a4.abs()));
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
    roots.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    roots
}

/// Real roots of a3 t^3 + a2 t^2 + a1 t + a0 (ascending).
fn solve_cubic(a3: f64, a2: f64, a1: f64, a0: f64) -> Vec<f64> {
    if a3.abs() < 1e-12 {
        // Quadratic fallback.
        let disc = a1 * a1 - 4.0 * a2 * a0;
        if a2.abs() < 1e-12 {
            if a1.abs() < 1e-12 {
                return vec![];
            }
            return vec![-a0 / a1];
        }
        if disc < 0.0 {
            return vec![];
        }
        let s = disc.sqrt();
        let mut v = vec![(-a1 - s) / (2.0 * a2), (-a1 + s) / (2.0 * a2)];
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        return v;
    }
    let a = a2 / a3;
    let b = a1 / a3;
    let c = a0 / a3;
    // Depressed: t = y - a/3 → y^3 + py + q
    let p = b - a * a / 3.0;
    let q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + c;
    let shift = -a / 3.0;
    let disc = q * q / 4.0 + p * p * p / 27.0;
    let mut roots = if disc > 0.0 {
        let s = disc.sqrt();
        let u = (-q / 2.0 + s).cbrt();
        let v = (-q / 2.0 - s).cbrt();
        vec![u + v + shift]
    } else {
        // Three real roots (trigonometric form).
        let m = 2.0 * (-p / 3.0).max(0.0).sqrt();
        if m < 1e-30 {
            vec![shift]
        } else {
            let theta = (3.0 * q / (p * m)).clamp(-1.0, 1.0).acos() / 3.0;
            (0..3)
                .map(|k| m * (theta - 2.0 * std::f64::consts::PI * k as f64 / 3.0).cos() + shift)
                .collect()
        }
    };
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
    roots
}

impl Intersectable for Torus {
    fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        let (lo, ld) = local_ray(&self.inverse_transform, ray);
        let big_r = self.major_radius;
        let r = self.minor_radius;

        let g = ld.dot(&ld);
        let o = lo - Point3::origin();
        let h = 2.0 * o.dot(&ld);
        let i = o.dot(&o) + big_r * big_r - r * r;
        let j = ld.x * ld.x + ld.y * ld.y;
        let k = 2.0 * (lo.x * ld.x + lo.y * ld.y);
        let l = lo.x * lo.x + lo.y * lo.y;

        let four_r2 = 4.0 * big_r * big_r;
        let roots = solve_quartic(
            g * g,
            2.0 * g * h,
            h * h + 2.0 * g * i - four_r2 * j,
            2.0 * h * i - four_r2 * k,
            i * i - four_r2 * l,
        );

        for t in roots {
            if t <= EPSILON {
                continue;
            }
            let p = lo + ld * t;
            if !self.phi_ok(&p) || !theta_ok(p.x, p.y, self.thetamax) {
                continue;
            }
            let r_xy = (p.x * p.x + p.y * p.y).sqrt().max(1e-12);
            let scale = 1.0 - big_r / r_xy;
            let local_n = Vec3::new(p.x * scale, p.y * scale, p.z).normalize();
            return Some(world_hit(
                &self.transform,
                &self.inverse_transform,
                ray,
                p,
                local_n,
                self.material_id,
            ));
        }
        None
    }

    fn describe(&self) -> PrimitiveDesc {
        PrimitiveDesc {
            kind: PrimitiveKind::Torus {
                major_radius: self.major_radius,
                minor_radius: self.minor_radius,
                phimin: self.phimin,
                phimax: self.phimax,
                thetamax: self.thetamax,
            },
            material_id: self.material_id,
            transform: self.transform,
            inverse_transform: self.inverse_transform,
        }
    }
}

// ---------------------------------------------------------------------------
// Disk

pub struct Disk {
    pub height: f64,
    pub radius: f64,
    pub thetamax: f64,
    pub material_id: usize,
    pub transform: Matrix4,
    pub inverse_transform: Matrix4,
}

impl Disk {
    pub fn new(
        height: f64,
        radius: f64,
        thetamax: f64,
        material_id: usize,
        transform: Matrix4,
    ) -> Self {
        let inverse_transform = transform.inverse().unwrap_or(Matrix4::identity());
        Self { height, radius, thetamax, material_id, transform, inverse_transform }
    }
}

impl Intersectable for Disk {
    fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        let (lo, ld) = local_ray(&self.inverse_transform, ray);
        if ld.z.abs() < EPSILON {
            return None;
        }
        let t = (self.height - lo.z) / ld.z;
        if t <= EPSILON {
            return None;
        }
        let p = lo + ld * t;
        if p.x * p.x + p.y * p.y > self.radius * self.radius {
            return None;
        }
        if !theta_ok(p.x, p.y, self.thetamax) {
            return None;
        }
        let local_n = Vec3::new(0.0, 0.0, 1.0);
        Some(world_hit(
            &self.transform,
            &self.inverse_transform,
            ray,
            p,
            local_n,
            self.material_id,
        ))
    }

    fn describe(&self) -> PrimitiveDesc {
        PrimitiveDesc {
            kind: PrimitiveKind::Disk {
                height: self.height,
                radius: self.radius,
                thetamax: self.thetamax,
            },
            material_id: self.material_id,
            transform: self.transform,
            inverse_transform: self.inverse_transform,
        }
    }
}

// ---------------------------------------------------------------------------
// Paraboloid

pub struct Paraboloid {
    pub rmax: f64,
    pub zmin: f64,
    pub zmax: f64,
    pub thetamax: f64,
    pub material_id: usize,
    pub transform: Matrix4,
    pub inverse_transform: Matrix4,
}

impl Paraboloid {
    pub fn new(
        rmax: f64,
        zmin: f64,
        zmax: f64,
        thetamax: f64,
        material_id: usize,
        transform: Matrix4,
    ) -> Self {
        let inverse_transform = transform.inverse().unwrap_or(Matrix4::identity());
        Self { rmax, zmin, zmax, thetamax, material_id, transform, inverse_transform }
    }
}

impl Intersectable for Paraboloid {
    fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        if self.zmax.abs() < EPSILON {
            return None;
        }
        let (lo, ld) = local_ray(&self.inverse_transform, ray);
        // x^2 + y^2 = k z with k = rmax^2 / zmax
        let k = self.rmax * self.rmax / self.zmax;
        let a = ld.x * ld.x + ld.y * ld.y;
        let b = 2.0 * (lo.x * ld.x + lo.y * ld.y) - k * ld.z;
        let c = lo.x * lo.x + lo.y * lo.y - k * lo.z;

        let candidates: Vec<f64> = if a.abs() < EPSILON {
            if b.abs() < EPSILON {
                return None;
            }
            vec![-c / b]
        } else {
            let disc = b * b - 4.0 * a * c;
            if disc < 0.0 {
                return None;
            }
            let s = disc.sqrt();
            vec![(-b - s) / (2.0 * a), (-b + s) / (2.0 * a)]
        };

        for t in candidates {
            if t <= EPSILON {
                continue;
            }
            let p = lo + ld * t;
            if p.z < self.zmin || p.z > self.zmax {
                continue;
            }
            if !theta_ok(p.x, p.y, self.thetamax) {
                continue;
            }
            let local_n = Vec3::new(2.0 * p.x, 2.0 * p.y, -k).normalize();
            return Some(world_hit(
                &self.transform,
                &self.inverse_transform,
                ray,
                p,
                local_n,
                self.material_id,
            ));
        }
        None
    }

    fn describe(&self) -> PrimitiveDesc {
        PrimitiveDesc {
            kind: PrimitiveKind::Paraboloid {
                rmax: self.rmax,
                zmin: self.zmin,
                zmax: self.zmax,
                thetamax: self.thetamax,
            },
            material_id: self.material_id,
            transform: self.transform,
            inverse_transform: self.inverse_transform,
        }
    }
}

// ---------------------------------------------------------------------------
// Hyperboloid

pub struct Hyperboloid {
    pub p1: [f64; 3],
    pub p2: [f64; 3],
    pub thetamax: f64,
    pub material_id: usize,
    pub transform: Matrix4,
    pub inverse_transform: Matrix4,
}

impl Hyperboloid {
    pub fn new(
        p1: [f64; 3],
        p2: [f64; 3],
        thetamax: f64,
        material_id: usize,
        transform: Matrix4,
    ) -> Self {
        let inverse_transform = transform.inverse().unwrap_or(Matrix4::identity());
        Self { p1, p2, thetamax, material_id, transform, inverse_transform }
    }

    /// Radius-squared profile as a quadratic in z: r^2(z) = A z^2 + B z + C.
    fn profile(&self) -> (f64, f64, f64) {
        let dz = self.p2[2] - self.p1[2];
        let u = 1.0 / dz;
        let sx = (self.p2[0] - self.p1[0]) * u;
        let sy = (self.p2[1] - self.p1[1]) * u;
        let cx = self.p1[0] - self.p1[2] * sx;
        let cy = self.p1[1] - self.p1[2] * sy;
        (sx * sx + sy * sy, 2.0 * (cx * sx + cy * sy), cx * cx + cy * cy)
    }
}

impl Intersectable for Hyperboloid {
    fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        let (lo, ld) = local_ray(&self.inverse_transform, ray);
        let zlo = self.p1[2].min(self.p2[2]);
        let zhi = self.p1[2].max(self.p2[2]);

        if (self.p2[2] - self.p1[2]).abs() < EPSILON {
            // Degenerate: flat annulus at z = z1 between the two radii.
            if ld.z.abs() < EPSILON {
                return None;
            }
            let t = (self.p1[2] - lo.z) / ld.z;
            if t <= EPSILON {
                return None;
            }
            let p = lo + ld * t;
            let r2 = p.x * p.x + p.y * p.y;
            let r1 = self.p1[0] * self.p1[0] + self.p1[1] * self.p1[1];
            let r2b = self.p2[0] * self.p2[0] + self.p2[1] * self.p2[1];
            if r2 < r1.min(r2b) || r2 > r1.max(r2b) {
                return None;
            }
            if !theta_ok(p.x, p.y, self.thetamax) {
                return None;
            }
            return Some(world_hit(
                &self.transform,
                &self.inverse_transform,
                ray,
                p,
                Vec3::new(0.0, 0.0, 1.0),
                self.material_id,
            ));
        }

        let (pa, pb, pc) = self.profile();
        // x^2 + y^2 - (A z^2 + B z + C) = 0 along the ray.
        let a = ld.x * ld.x + ld.y * ld.y - pa * ld.z * ld.z;
        let b = 2.0 * (lo.x * ld.x + lo.y * ld.y - pa * lo.z * ld.z) - pb * ld.z;
        let c = lo.x * lo.x + lo.y * lo.y - pa * lo.z * lo.z - pb * lo.z - pc;

        let candidates: Vec<f64> = if a.abs() < EPSILON {
            if b.abs() < EPSILON {
                return None;
            }
            vec![-c / b]
        } else {
            let disc = b * b - 4.0 * a * c;
            if disc < 0.0 {
                return None;
            }
            let s = disc.sqrt();
            vec![(-b - s) / (2.0 * a), (-b + s) / (2.0 * a)]
        };

        for t in candidates {
            if t <= EPSILON {
                continue;
            }
            let p = lo + ld * t;
            if p.z < zlo || p.z > zhi {
                continue;
            }
            if !theta_ok(p.x, p.y, self.thetamax) {
                continue;
            }
            let local_n = Vec3::new(2.0 * p.x, 2.0 * p.y, -(2.0 * pa * p.z + pb)).normalize();
            return Some(world_hit(
                &self.transform,
                &self.inverse_transform,
                ray,
                p,
                local_n,
                self.material_id,
            ));
        }
        None
    }

    fn describe(&self) -> PrimitiveDesc {
        PrimitiveDesc {
            kind: PrimitiveKind::Hyperboloid {
                p1: self.p1,
                p2: self.p2,
                thetamax: self.thetamax,
            },
            material_id: self.material_id,
            transform: self.transform,
            inverse_transform: self.inverse_transform,
        }
    }
}

// ---------------------------------------------------------------------------
// Triangle (from Polygon fan-triangulation; double-sided)

pub struct Triangle {
    pub v0: [f64; 3],
    pub v1: [f64; 3],
    pub v2: [f64; 3],
    pub material_id: usize,
    pub transform: Matrix4,
    pub inverse_transform: Matrix4,
}

impl Triangle {
    pub fn new(
        v0: [f64; 3],
        v1: [f64; 3],
        v2: [f64; 3],
        material_id: usize,
        transform: Matrix4,
    ) -> Self {
        let inverse_transform = transform.inverse().unwrap_or(Matrix4::identity());
        Self { v0, v1, v2, material_id, transform, inverse_transform }
    }
}

impl Intersectable for Triangle {
    fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        let (lo, ld) = local_ray(&self.inverse_transform, ray);
        let v0 = Point3::new(self.v0[0], self.v0[1], self.v0[2]);
        let e1 = Point3::new(self.v1[0], self.v1[1], self.v1[2]) - v0;
        let e2 = Point3::new(self.v2[0], self.v2[1], self.v2[2]) - v0;

        // Möller–Trumbore in object space.
        let pvec = ld.cross(&e2);
        let det = e1.dot(&pvec);
        if det.abs() < 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;
        let tvec = lo - v0;
        let u = tvec.dot(&pvec) * inv_det;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let qvec = tvec.cross(&e1);
        let v = ld.dot(&qvec) * inv_det;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = e2.dot(&qvec) * inv_det;
        if t <= EPSILON {
            return None;
        }

        let p = lo + ld * t;
        let mut local_n = e1.cross(&e2).normalize();
        // Record the geometric side before the double-sided flip.
        let front_face = local_n.dot(&ld) < 0.0;
        if !front_face {
            local_n = -local_n;
        }
        Some(
            world_hit(
                &self.transform,
                &self.inverse_transform,
                ray,
                p,
                local_n,
                self.material_id,
            )
            .with_front_face(front_face),
        )
    }

    fn describe(&self) -> PrimitiveDesc {
        PrimitiveDesc {
            kind: PrimitiveKind::Triangle { v0: self.v0, v1: self.v1, v2: self.v2 },
            material_id: self.material_id,
            transform: self.transform,
            inverse_transform: self.inverse_transform,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quartic_known_roots() {
        // (t-1)(t-2)(t-3)(t-4) = t^4 -10t^3 +35t^2 -50t +24
        let roots = solve_quartic(1.0, -10.0, 35.0, -50.0, 24.0);
        assert_eq!(roots.len(), 4);
        for (r, expect) in roots.iter().zip([1.0, 2.0, 3.0, 4.0]) {
            assert!((r - expect).abs() < 1e-6, "root {r} vs {expect}");
        }
    }

    #[test]
    fn torus_axis_ray() {
        // Ray along +x through a torus R=2 r=0.5 centered at origin:
        // entry at x = -2.5 from origin at (-5,0,0) → t = 2.5.
        let torus = Torus::new(2.0, 0.5, 0.0, 360.0, 360.0, 0, Matrix4::identity());
        let ray = Ray::new(Point3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let hit = torus.intersect(&ray).expect("should hit");
        assert!((hit.t - 2.5).abs() < 1e-6, "t = {}", hit.t);
    }

    #[test]
    fn disk_hit_and_theta() {
        let disk = Disk::new(1.0, 2.0, 360.0, 0, Matrix4::identity());
        let ray = Ray::new(Point3::new(0.5, 0.5, -3.0), Vec3::new(0.0, 0.0, 1.0));
        let hit = disk.intersect(&ray).expect("should hit");
        assert!((hit.t - 4.0).abs() < 1e-9);
        let partial = Disk::new(1.0, 2.0, 90.0, 0, Matrix4::identity());
        // (0.5, -0.5) has phi = 315° > 90° → miss.
        let ray2 = Ray::new(Point3::new(0.5, -0.5, -3.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(partial.intersect(&ray2).is_none());
    }

    #[test]
    fn paraboloid_hit() {
        // rmax=1 at zmax=1: surface x^2+y^2=z. Ray down +z at x=0.5:
        // hits at z=0.25.
        let par = Paraboloid::new(1.0, 0.0, 1.0, 360.0, 0, Matrix4::identity());
        let ray = Ray::new(Point3::new(0.5, 0.0, -2.0), Vec3::new(0.0, 0.0, 1.0));
        let hit = par.intersect(&ray).expect("should hit");
        assert!((hit.point.z - 0.25).abs() < 1e-9, "z = {}", hit.point.z);
    }

    #[test]
    fn hyperboloid_cylinder_case() {
        // p1=(1,0,-1), p2=(1,0,1) is a cylinder of radius 1.
        let hyp = Hyperboloid::new([1.0, 0.0, -1.0], [1.0, 0.0, 1.0], 360.0, 0, Matrix4::identity());
        let ray = Ray::new(Point3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let hit = hyp.intersect(&ray).expect("should hit");
        assert!((hit.t - 4.0).abs() < 1e-9, "t = {}", hit.t);
    }

    #[test]
    fn triangle_hit_double_sided() {
        let tri = Triangle::new(
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            0,
            Matrix4::identity(),
        );
        let front = Ray::new(Point3::new(0.0, 0.0, -2.0), Vec3::new(0.0, 0.0, 1.0));
        let back = Ray::new(Point3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, -1.0));
        let hf = tri.intersect(&front).expect("front hit");
        let hb = tri.intersect(&back).expect("back hit");
        // Normal faces the ray origin in both cases.
        assert!(hf.normal.z < 0.0);
        assert!(hb.normal.z > 0.0);
    }
}
