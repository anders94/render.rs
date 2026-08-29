use crate::math::Vec3;

#[derive(Debug, Clone)]
pub enum MaterialType {
    Matte,
    Plastic { roughness: f64 },
    Metal { roughness: f64 },
}

#[derive(Debug, Clone)]
pub struct Material {
    pub material_type: MaterialType,
    pub color: Vec3,
    pub ka: f64,
    pub kd: f64,
    pub ks: f64,
    /// Emitted radiance (area-light surfaces; zero for everything else).
    pub emission: Vec3,
    /// Index into scene.lights when this surface is an area light, so the
    /// path tracer can compute MIS weights on emitter hits.
    pub area_light: Option<usize>,
}

impl Material {
    pub fn matte(color: Vec3) -> Self {
        Self {
            material_type: MaterialType::Matte,
            color,
            ka: 0.3,  // Increased ambient for brighter rendering without lights
            kd: 0.9,
            ks: 0.0,
            emission: Vec3::zero(),
            area_light: None,
        }
    }

    pub fn plastic(color: Vec3, roughness: f64) -> Self {
        Self {
            material_type: MaterialType::Plastic { roughness },
            color,
            ka: 0.3,  // Increased ambient for brighter rendering without lights
            kd: 0.6,
            ks: 0.4,
            emission: Vec3::zero(),
            area_light: None,
        }
    }

    pub fn metal(color: Vec3, roughness: f64) -> Self {
        Self {
            material_type: MaterialType::Metal { roughness },
            color,
            ka: 0.3,  // Increased ambient for brighter rendering without lights
            kd: 0.2,
            ks: 0.8,
            emission: Vec3::zero(),
            area_light: None,
        }
    }

    /// Fraction of incoming light mirrored by this surface. Rougher surfaces
    /// reflect less.
    pub fn reflectivity(&self) -> f64 {
        match self.material_type {
            MaterialType::Metal { roughness } => 0.6 * (1.0 - roughness.min(1.0) * 0.5),
            MaterialType::Plastic { .. } => 0.08,
            MaterialType::Matte => 0.0,
        }
    }
}
