mod camera;
mod light;
mod material;
pub mod transform;

pub use camera::Camera;
pub use light::{Light, LightType};
pub use material::{Material, MaterialType};
pub use transform::TransformStack;

use crate::geometry::Intersectable;
use crate::math::{Point3, Vec3};
use crate::raytracer::{Intersection, Ray};
use std::sync::Arc;

pub struct Scene {
    pub camera: Camera,
    pub objects: Vec<Arc<dyn Intersectable>>,
    pub lights: Vec<Light>,
    pub materials: Vec<Material>,
    pub background_color: Vec3,
    /// Samples per pixel in x and y (from the PixelSamples directive).
    pub pixel_samples: (u32, u32),
}

impl Scene {
    pub fn new(camera: Camera) -> Self {
        Self {
            camera,
            objects: Vec::new(),
            lights: Vec::new(),
            materials: Vec::new(),
            background_color: Vec3::new(0.0, 0.0, 0.0),
            pixel_samples: (1, 1),
        }
    }

    /// Closest intersection of the ray with any object in the scene.
    pub fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        let mut closest: Option<Intersection> = None;
        let mut closest_t = f64::INFINITY;

        for object in &self.objects {
            if let Some(intersection) = object.intersect(ray) {
                if intersection.t < closest_t {
                    closest_t = intersection.t;
                    closest = Some(intersection);
                }
            }
        }

        closest
    }

    /// True if anything blocks the path from `point` toward `light`.
    pub fn is_occluded(&self, point: &Point3, normal: &Vec3, light: &Light) -> bool {
        let light_dir = light.direction_from(point);
        let max_t = light.distance_from(point);
        // Offset along the normal to avoid self-shadowing acne.
        let origin = *point + *normal * 1e-4;
        let shadow_ray = Ray::new(origin, light_dir);

        for object in &self.objects {
            if let Some(intersection) = object.intersect(&shadow_ray) {
                if intersection.t < max_t - 1e-4 {
                    return true;
                }
            }
        }

        false
    }
}
