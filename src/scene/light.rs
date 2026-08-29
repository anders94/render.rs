use crate::math::{Point3, Vec3};

#[derive(Debug, Clone)]
pub enum LightType {
    Point { position: Point3 },
    Distant { direction: Vec3 },
    /// Parallelogram area light: corner + two edge vectors. Emits from both
    /// faces. Sampled by the path tracer; the Whitted paths skip it.
    Rect {
        corner: Point3,
        edge1: Vec3,
        edge2: Vec3,
        normal: Vec3,
        area: f64,
    },
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

    pub fn rect(corner: Point3, edge1: Vec3, edge2: Vec3, intensity: f64, color: Vec3) -> Self {
        let cross = edge1.cross(&edge2);
        let area = cross.length();
        Self {
            light_type: LightType::Rect {
                corner,
                edge1,
                edge2,
                normal: cross.normalize(),
                area,
            },
            intensity,
            color,
        }
    }

    /// Emitted radiance (for area lights).
    pub fn radiance(&self) -> Vec3 {
        self.color * self.intensity
    }

    pub fn direction_from(&self, point: &Point3) -> Vec3 {
        match &self.light_type {
            LightType::Point { position } => (*position - *point).normalize(),
            LightType::Distant { direction } => -*direction,
            LightType::Rect { corner, edge1, edge2, .. } => {
                (*corner + *edge1 * 0.5 + *edge2 * 0.5 - *point).normalize()
            }
        }
    }

    /// Distance from `point` to the light (infinite for distant lights).
    pub fn distance_from(&self, point: &Point3) -> f64 {
        match &self.light_type {
            LightType::Point { position } => position.distance(point),
            LightType::Distant { .. } => f64::INFINITY,
            LightType::Rect { corner, edge1, edge2, .. } => {
                (*corner + *edge1 * 0.5 + *edge2 * 0.5).distance(point)
            }
        }
    }
}
