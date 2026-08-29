use crate::math::{Point3, Vec3};

#[derive(Debug, Clone)]
pub enum LightType {
    Point { position: Point3 },
    Distant { direction: Vec3 },
}

#[derive(Debug, Clone)]
pub struct Light {
    pub light_type: LightType,
    pub intensity: f64,
    pub color: Vec3,
}

impl Light {
    pub fn point(position: Point3, intensity: f64, color: Vec3) -> Self {
        Self {
            light_type: LightType::Point { position },
            intensity,
            color,
        }
    }

    pub fn distant(direction: Vec3, intensity: f64, color: Vec3) -> Self {
        Self {
            light_type: LightType::Distant {
                direction: direction.normalize(),
            },
            intensity,
            color,
        }
    }

    pub fn direction_from(&self, point: &Point3) -> Vec3 {
        match &self.light_type {
            LightType::Point { position } => (*position - *point).normalize(),
            LightType::Distant { direction } => -*direction,
        }
    }

    /// Distance from `point` to the light (infinite for distant lights).
    pub fn distance_from(&self, point: &Point3) -> f64 {
        match &self.light_type {
            LightType::Point { position } => position.distance(point),
            LightType::Distant { .. } => f64::INFINITY,
        }
    }
}
