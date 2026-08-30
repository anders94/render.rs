use crate::math::{Matrix4, Point3, Vec3};
use crate::raytracer::Ray;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Projection {
    Perspective,
    Orthographic,
}

/// Reconstruction filter for pixel sampling (filter importance sampling:
/// subpixel offsets are drawn from the filter's distribution, so every
/// sample carries weight 1).
#[derive(Debug, Clone, Copy)]
pub enum PixelFilter {
    /// Uniform over [-w/2, w/2].
    Box { width: f64 },
    /// Tent over [-w/2, w/2].
    Triangle { width: f64 },
    /// Truncated gaussian, sigma = w/4, support [-w/2, w/2].
    Gaussian { width: f64 },
}

impl PixelFilter {
    pub fn from_name(name: &str, xwidth: f64, ywidth: f64) -> Self {
        // Anisotropic widths collapse to the average (isotropic sampling).
        let w = (xwidth + ywidth) * 0.5;
        match name {
            "triangle" => PixelFilter::Triangle { width: w.max(1e-3) },
            "gaussian" => PixelFilter::Gaussian { width: w.max(1e-3) },
            "box" => PixelFilter::Box { width: w.max(1e-3) },
            _ => PixelFilter::Box { width: 1.0 },
        }
    }

    /// One filter-distributed offset from a uniform u in [0,1).
    fn sample_1d(&self, u: f64) -> f64 {
        match self {
            PixelFilter::Box { width } => (u - 0.5) * width,
            PixelFilter::Triangle { width } => {
                let half = width * 0.5;
                if u < 0.5 {
                    ((2.0 * u).sqrt() - 1.0) * half
                } else {
                    (1.0 - (2.0 * (1.0 - u)).sqrt()) * half
                }
            }
            PixelFilter::Gaussian { width } => {
                // Inverse-CDF via rational approximation is overkill here;
                // sum of uniforms (Irwin-Hall, n=4) approximates a gaussian
                // with sigma ~= w/4 after scaling, deterministic in one u.
                // Instead: use the exact inverse CDF of a *tent-tempered*
                // gaussian? Keep it simple and robust: Box-Muller needs two
                // uniforms, so derive the second from bit-mixing u.
                let u1 = u.max(1e-12);
                let u2 = (u * 2654435761.0).fract();
                let sigma = width / 4.0;
                let g = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                (g * sigma).clamp(-width * 0.5, width * 0.5)
            }
        }
    }

    /// Subpixel offset pair from two uniforms.
    pub fn sample(&self, u1: f64, u2: f64) -> (f64, f64) {
        (self.sample_1d(u1), self.sample_1d(u2))
    }
}

pub struct Camera {
    pub width: u32,
    pub height: u32,
    pub fov: f64,
    pub eye: Point3,
    pub forward: Vec3,
    pub up: Vec3,
    pub right: Vec3,
    pub projection: Projection,
    /// Thin-lens aperture radius in scene units (0 = pinhole).
    pub lens_radius: f64,
    /// Distance along `forward` to the plane in perfect focus.
    pub focal_distance: f64,
    /// Half-extents of the orthographic screen window in camera units.
    pub ortho_half: (f64, f64),
    pub filter: PixelFilter,
    /// Shutter open/close times (motion samples map onto [open, close]).
    pub shutter: (f64, f64),
    /// Camera motion: the inverse world-to-camera delta at shutter close
    /// (identity at open). Rays at time t are transformed by
    /// lerp(I, motion_inv, t) — the world the baked geometry lives in IS
    /// camera space at shutter open, so moving the camera means counter-
    /// moving the rays.
    pub motion_inv: Option<Matrix4>,
}

impl Camera {
    pub fn new(width: u32, height: u32, fov: f64) -> Self {
        let eye = Point3::origin();
        let forward = Vec3::new(0.0, 0.0, 1.0);
        let up = Vec3::new(0.0, 1.0, 0.0);
        let right = up.cross(&forward).normalize();
        let aspect = width as f64 / height.max(1) as f64;

        Self {
            width,
            height,
            fov,
            eye,
            forward,
            up,
            right,
            projection: Projection::Perspective,
            lens_radius: 0.0,
            focal_distance: 1.0,
            ortho_half: (aspect, 1.0),
            filter: PixelFilter::Box { width: 1.0 },
            shutter: (0.0, 0.0),
            motion_inv: None,
        }
    }

    pub fn set_transform(&mut self, eye: Point3, look_at: Point3, up: Vec3) {
        self.eye = eye;
        self.forward = (look_at - eye).normalize();
        self.right = self.forward.cross(&up).normalize();
        self.up = self.right.cross(&self.forward).normalize();
    }

    /// Pinhole ray through pixel-space coordinates (px, py); fractional
    /// values address subpixel positions. (The Whitted path and tests.)
    pub fn generate_ray(&self, px: f64, py: f64) -> Ray {
        self.generate_ray_lens(px, py, 0.5, 0.5)
    }

    /// Apply camera motion to a ray at shutter time t.
    pub fn apply_motion(&self, ray: Ray, t: f64) -> Ray {
        let Some(m1) = &self.motion_inv else { return ray };
        if t <= 0.0 {
            return ray;
        }
        let m = Matrix4::identity().lerp(m1, t);
        Ray {
            origin: m.transform_point(&ray.origin),
            direction: m.transform_vec(&ray.direction).normalize(),
            time: ray.time,
        }
    }

    /// Ray through (px, py) with a lens sample (lu, lv) in [0,1)^2 for
    /// thin-lens depth of field. Ortho cameras ignore the lens.
    pub fn generate_ray_lens(&self, px: f64, py: f64, lu: f64, lv: f64) -> Ray {
        let u = (px / self.width as f64) * 2.0 - 1.0;
        let v = 1.0 - (py / self.height as f64) * 2.0;

        if self.projection == Projection::Orthographic {
            let origin = self.eye
                + self.right * (u * self.ortho_half.0)
                + self.up * (v * self.ortho_half.1);
            return Ray::new(origin, self.forward);
        }

        let aspect_ratio = self.width as f64 / self.height as f64;
        let half_height = (self.fov.to_radians() / 2.0).tan();
        let half_width = aspect_ratio * half_height;
        // Forward component is exactly 1, so `direction * focal_distance`
        // lands on the focal plane.
        let direction =
            self.forward + self.right * (u * half_width) + self.up * (v * half_height);

        if self.lens_radius <= 0.0 {
            return Ray::new(self.eye, direction);
        }
        // Concentric-free uniform disk sample of the aperture.
        let r = self.lens_radius * lu.sqrt();
        let phi = 2.0 * std::f64::consts::PI * lv;
        let offset = self.right * (r * phi.cos()) + self.up * (r * phi.sin());
        let focus = self.eye + direction * self.focal_distance;
        let origin = self.eye + offset;
        Ray::new(origin, focus - origin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lens_rays_converge_at_focal_plane() {
        let mut cam = Camera::new(64, 64, 45.0);
        cam.lens_radius = 0.2;
        cam.focal_distance = 8.0;
        // All lens samples through the same pixel hit the same focal point.
        let center = cam.generate_ray_lens(20.0, 40.0, 0.5, 0.5);
        let t_center = 8.0 / center.direction.dot(&cam.forward);
        let focus = center.at(t_center);
        for (lu, lv) in [(0.1, 0.3), (0.9, 0.7), (0.5, 0.05)] {
            let ray = cam.generate_ray_lens(20.0, 40.0, lu, lv);
            let t = 8.0 / ray.direction.dot(&cam.forward);
            let p = ray.at(t);
            assert!(p.distance(&focus) < 1e-9, "lens ray missed focus: {p:?}");
        }
    }

    #[test]
    fn ortho_rays_are_parallel() {
        let mut cam = Camera::new(64, 64, 45.0);
        cam.projection = Projection::Orthographic;
        let a = cam.generate_ray(5.0, 5.0);
        let b = cam.generate_ray(60.0, 60.0);
        assert!((a.direction - b.direction).length() < 1e-12);
        assert!(a.origin.distance(&b.origin) > 0.1);
    }

    #[test]
    fn filter_samples_stay_in_support_and_center() {
        for filter in [
            PixelFilter::Box { width: 1.0 },
            PixelFilter::Triangle { width: 2.0 },
            PixelFilter::Gaussian { width: 2.0 },
        ] {
            let mut mean = 0.0;
            let n = 4096;
            for i in 0..n {
                let u = (i as f64 + 0.5) / n as f64;
                let dx = filter.sample_1d(u);
                let half = match filter {
                    PixelFilter::Box { width }
                    | PixelFilter::Triangle { width }
                    | PixelFilter::Gaussian { width } => width * 0.5,
                };
                assert!(dx.abs() <= half + 1e-9, "{filter:?} produced {dx}");
                mean += dx;
            }
            mean /= n as f64;
            assert!(mean.abs() < 0.05, "{filter:?} mean {mean}");
        }
    }
}
