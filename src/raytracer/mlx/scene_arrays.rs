//! MLX view of the flattened scene: wraps the backend-neutral
//! `raytracer::flatten::FlatScene`, turning its per-object material table
//! into MLX arrays gathered per ray by hit index.

pub use crate::raytracer::flatten::{FlatCamera, FlatLight, FlatLightKind, FlatObject};
use crate::scene::Scene;
use anyhow::Result;
use mlx_rs::Array;

pub struct FlatScene {
    pub objects: Vec<FlatObject>,
    pub mat_r: Array,
    pub mat_g: Array,
    pub mat_b: Array,
    pub mat_ka: Array,
    pub mat_kd: Array,
    pub mat_ks: Array,
    pub mat_shininess: Array,
    pub mat_reflectivity: Array,
    pub mat_is_metal: Array,
    pub lights: Vec<FlatLight>,
    pub background: [f32; 3],
    pub camera: FlatCamera,
    pub pixel_samples: (u32, u32),
    pub width: u32,
    pub height: u32,
}

impl FlatScene {
    pub fn from_scene(scene: &Scene) -> Result<Self> {
        let flat = crate::raytracer::flatten::FlatScene::from_scene(scene)?;
        let count = flat.materials.len() as i32;
        let column = |f: &dyn Fn(&crate::raytracer::flatten::FlatMaterial) -> f32| -> Array {
            let v: Vec<f32> = flat.materials.iter().map(f).collect();
            Array::from_slice(&v, &[count])
        };
        let is_metal: Vec<bool> = flat.materials.iter().map(|m| m.is_metal).collect();

        Ok(Self {
            mat_r: column(&|m| m.color[0]),
            mat_g: column(&|m| m.color[1]),
            mat_b: column(&|m| m.color[2]),
            mat_ka: column(&|m| m.ka),
            mat_kd: column(&|m| m.kd),
            mat_ks: column(&|m| m.ks),
            mat_shininess: column(&|m| m.shininess),
            mat_reflectivity: column(&|m| m.reflectivity),
            mat_is_metal: Array::from_slice(&is_metal, &[count]),
            objects: flat.objects,
            lights: flat.lights,
            background: flat.background,
            camera: flat.camera,
            pixel_samples: flat.pixel_samples,
            width: flat.width,
            height: flat.height,
        })
    }
}
