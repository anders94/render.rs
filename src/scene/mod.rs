mod camera;
pub mod envmap;
mod light;
pub mod light_sampler;
pub mod ies;
pub mod medium;
mod material;
pub mod pbr;
pub mod transform;

pub use camera::{Camera, PixelFilter, Projection};
pub use envmap::EnvMap;
pub use light::{Light, LightType};
pub use light_sampler::LightSampler;
pub use ies::IesProfile;
pub use medium::{DensityField, Medium};
pub use material::{Material, MaterialType};
pub use pbr::PbrParams;
pub use transform::TransformStack;

use crate::accel::Bvh;
use crate::geometry::{to_intersection, CurveSet, Instance, Intersectable, Mesh};
use crate::math::{Point3, Vec3};
use crate::raytracer::{Intersection, Ray};
use std::sync::Arc;

pub struct Scene {
    pub camera: Camera,
    /// Standalone primitives (quadrics, loose polygons): linear scan.
    pub objects: Vec<Arc<dyn Intersectable>>,
    /// Mesh library (BLAS per mesh) and their placed instances, traversed
    /// through the TLAS.
    pub meshes: Vec<Mesh>,
    /// Curve/point sets, addressed by instances with GeomKind::Curves.
    pub curve_sets: Vec<CurveSet>,
    pub instances: Vec<Instance>,
    pub lights: Vec<Light>,
    pub materials: Vec<Material>,
    /// Pattern node graph shared by all materials (see Material::pattern_bindings).
    pub patterns: Vec<crate::texture::pattern::PatternNode>,
    pub background_color: Vec3,
    /// True when any instance or mesh carries motion endpoints; rays then
    /// draw a shutter time per sample.
    pub has_motion: bool,
    /// Samples per pixel in x and y (from the PixelSamples directive).
    pub pixel_samples: (u32, u32),
    /// Many-light sampler (built once after lights are final).
    pub light_sampler: LightSampler,
    /// Participating media, referenced by Material::interior.
    pub media: Vec<Medium>,
    /// Global medium the camera starts in (Atmosphere request).
    pub atmosphere: Option<u32>,
    /// Object-id manifest for the id AOV (id -> identifier name).
    pub id_manifest: std::collections::BTreeMap<u32, String>,
    tlas: Bvh,
}

impl Scene {
    pub fn new(camera: Camera) -> Self {
        Self {
            camera,
            objects: Vec::new(),
            meshes: Vec::new(),
            curve_sets: Vec::new(),
            instances: Vec::new(),
            lights: Vec::new(),
            materials: Vec::new(),
            patterns: Vec::new(),
            background_color: Vec3::new(0.0, 0.0, 0.0),
            has_motion: false,
            pixel_samples: (1, 1),
            light_sampler: LightSampler::build(&[]),
            media: Vec::new(),
            atmosphere: None,
            id_manifest: std::collections::BTreeMap::new(),
            tlas: Bvh::build(&[]),
        }
    }

    /// Rebuild the TLAS over instance world bounds. Call after instances
    /// change (SceneBuilder does this once at the end of build()).
    pub fn build_tlas(&mut self) {
        let bounds: Vec<_> = self.instances.iter().map(|i| i.world_bounds).collect();
        self.tlas = Bvh::build(&bounds);
        self.light_sampler = LightSampler::build(&self.lights);
    }

    pub fn triangle_count(&self) -> usize {
        self.instances
            .iter()
            .filter(|i| i.kind == crate::geometry::GeomKind::Mesh)
            .map(|i| self.meshes[i.mesh_id as usize].triangle_count())
            .sum()
    }

    pub fn curve_segment_count(&self) -> usize {
        self.instances
            .iter()
            .filter(|i| i.kind == crate::geometry::GeomKind::Curves)
            .map(|i| self.curve_sets[i.mesh_id as usize].segment_count())
            .sum()
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

        // TLAS runs in parametric t (units of |ray.direction|); the legacy
        // primitives above use euclidean t. Convert the budget going in and
        // the result coming out.
        if !self.instances.is_empty() {
            let dir_len = ray.direction.length().max(1e-300);
            let mut best_hit: Option<(usize, crate::geometry::MeshHit)> = None;
            let t_budget = closest_t / dir_len;
            let winner = self.tlas.traverse(ray, t_budget, |inst_id, t_max| {
                let instance = &self.instances[inst_id as usize];
                instance
                    .intersect(&self.meshes, &self.curve_sets, ray, t_max)
                    .map(|hit| {
                        let t = hit.t_param;
                        best_hit = Some((instance.material_id, hit));
                        t
                    })
            });
            if winner.is_some() {
                if let Some((material_id, hit)) = best_hit {
                    let intersection = to_intersection(&hit, ray, material_id);
                    if intersection.t < closest_t {
                        closest = Some(intersection);
                    }
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

        if !self.instances.is_empty() {
            // light_dir is normalized, so parametric and euclidean t match.
            let limit = max_t - 1e-4;
            if self.tlas.any_hit(&shadow_ray, limit, |inst_id, lim| {
                self.instances[inst_id as usize]
                    .occludes(&self.meshes, &self.curve_sets, &shadow_ray, lim)
            }) {
                return true;
            }
        }

        false
    }
}
