use crate::math::{Point3, Vec3};
use crate::raytracer::Ray;

pub struct Camera {
    pub width: u32,
    pub height: u32,
    pub fov: f64,
    pub eye: Point3,
    pub forward: Vec3,
    pub up: Vec3,
    pub right: Vec3,
}

impl Camera {
    pub fn new(width: u32, height: u32, fov: f64) -> Self {
        let eye = Point3::origin();
        let forward = Vec3::new(0.0, 0.0, 1.0);
        let up = Vec3::new(0.0, 1.0, 0.0);
        let right = up.cross(&forward).normalize();

        Self {
            width,
            height,
            fov,
            eye,
            forward,
            up,
            right,
        }
    }

    pub fn set_transform(&mut self, eye: Point3, look_at: Point3, up: Vec3) {
        self.eye = eye;
        self.forward = (look_at - eye).normalize();
        self.right = self.forward.cross(&up).normalize();
        self.up = self.right.cross(&self.forward).normalize();
    }

    /// Generate a ray through pixel-space coordinates (px, py), where
    /// fractional values address subpixel sample positions.
    pub fn generate_ray(&self, px: f64, py: f64) -> Ray {
        let aspect_ratio = self.width as f64 / self.height as f64;
        let fov_radians = self.fov.to_radians();
        let half_height = (fov_radians / 2.0).tan();
        let half_width = aspect_ratio * half_height;

        let u = (px / self.width as f64) * 2.0 - 1.0;
        let v = 1.0 - (py / self.height as f64) * 2.0;

        let direction = self.forward + self.right * (u * half_width) + self.up * (v * half_height);

        Ray::new(self.eye, direction)
    }
}
