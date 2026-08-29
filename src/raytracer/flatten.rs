//! Backend-neutral scene flattening: plain f32 data with no GPU
//! dependencies, consumed by the MLX and Metal backends.
//!
//! Material tables are per-object (entry i is the material object i
//! references), so a hit-object index doubles as a material index.

use crate::geometry::PrimitiveKind;
use crate::math::Matrix4;
use crate::scene::{LightType, MaterialType, Scene};
use crate::shading::shininess_for;
use anyhow::{bail, Result};

pub struct FlatObject {
    pub kind: PrimitiveKind,
    /// Row-major world-to-local transform.
    pub inv: [f32; 16],
    /// Row-major local-to-world transform.
    pub fwd: [f32; 16],
}

#[derive(Clone, Copy)]
pub struct FlatMaterial {
    pub color: [f32; 3],
    pub ka: f32,
    pub kd: f32,
    pub ks: f32,
    /// Precomputed shading::shininess_for().
    pub shininess: f32,
    /// Precomputed Material::reflectivity().
    pub reflectivity: f32,
    pub is_metal: bool,
}

pub enum FlatLightKind {
    Point { position: [f32; 3] },
    Distant { direction: [f32; 3] },
}

pub struct FlatLight {
    pub kind: FlatLightKind,
    pub intensity: f32,
    pub color: [f32; 3],
}

pub struct FlatCamera {
    pub eye: [f32; 3],
    pub forward: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub half_width: f32,
    pub half_height: f32,
}

pub struct FlatScene {
    pub objects: Vec<FlatObject>,
    pub materials: Vec<FlatMaterial>,
    pub lights: Vec<FlatLight>,
    pub background: [f32; 3],
    pub camera: FlatCamera,
    pub pixel_samples: (u32, u32),
    pub width: u32,
    pub height: u32,
}

pub fn matrix_to_f32(m: &Matrix4) -> Result<[f32; 16]> {
    let rows = m.rows();
    let bottom = rows[3];
    if (bottom[0].abs() > 1e-9)
        || (bottom[1].abs() > 1e-9)
        || (bottom[2].abs() > 1e-9)
        || ((bottom[3] - 1.0).abs() > 1e-9)
    {
        bail!("GPU backends require affine object transforms (bottom row [0,0,0,1])");
    }
    let mut out = [0.0f32; 16];
    for r in 0..4 {
        for c in 0..4 {
            out[r * 4 + c] = rows[r][c] as f32;
        }
    }
    Ok(out)
}

impl FlatScene {
    pub fn from_scene(scene: &Scene) -> Result<Self> {
        let mut objects = Vec::with_capacity(scene.objects.len());
        let mut materials = Vec::with_capacity(scene.objects.len());

        for object in &scene.objects {
            let desc = object.describe();
            objects.push(FlatObject {
                kind: desc.kind,
                inv: matrix_to_f32(&desc.inverse_transform)?,
                fwd: matrix_to_f32(&desc.transform)?,
            });

            let material = scene
                .materials
                .get(desc.material_id)
                .ok_or_else(|| anyhow::anyhow!("object references missing material"))?;
            materials.push(FlatMaterial {
                color: [
                    material.color.x as f32,
                    material.color.y as f32,
                    material.color.z as f32,
                ],
                ka: material.ka as f32,
                kd: material.kd as f32,
                ks: material.ks as f32,
                shininess: shininess_for(material) as f32,
                reflectivity: material.reflectivity() as f32,
                is_metal: matches!(material.material_type, MaterialType::Metal { .. }),
            });
        }

        let lights = scene
            .lights
            .iter()
            .map(|light| {
                let kind = match &light.light_type {
                    LightType::Point { position } => FlatLightKind::Point {
                        position: [position.x as f32, position.y as f32, position.z as f32],
                    },
                    LightType::Distant { direction } => FlatLightKind::Distant {
                        direction: [direction.x as f32, direction.y as f32, direction.z as f32],
                    },
                };
                FlatLight {
                    kind,
                    intensity: light.intensity as f32,
                    color: [
                        light.color.x as f32,
                        light.color.y as f32,
                        light.color.z as f32,
                    ],
                }
            })
            .collect();

        // Same math as Camera::generate_ray, precomputed in f64.
        let cam = &scene.camera;
        let aspect_ratio = cam.width as f64 / cam.height as f64;
        let half_height = (cam.fov.to_radians() / 2.0).tan();
        let half_width = aspect_ratio * half_height;
        let camera = FlatCamera {
            eye: [cam.eye.x as f32, cam.eye.y as f32, cam.eye.z as f32],
            forward: [cam.forward.x as f32, cam.forward.y as f32, cam.forward.z as f32],
            right: [cam.right.x as f32, cam.right.y as f32, cam.right.z as f32],
            up: [cam.up.x as f32, cam.up.y as f32, cam.up.z as f32],
            half_width: half_width as f32,
            half_height: half_height as f32,
        };

        Ok(Self {
            objects,
            materials,
            lights,
            background: [
                scene.background_color.x as f32,
                scene.background_color.y as f32,
                scene.background_color.z as f32,
            ],
            camera,
            pixel_samples: scene.pixel_samples,
            width: cam.width,
            height: cam.height,
        })
    }
}
