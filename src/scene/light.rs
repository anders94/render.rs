use crate::math::{Point3, Vec3};
use crate::scene::envmap::EnvMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum LightType {
    Point { position: Point3 },
    /// `angular_radius` (radians) > 0 turns the delta sun into a soft disk.
    Distant { direction: Vec3, angular_radius: f64 },
    /// Parallelogram area light: corner + two edge vectors. Emits from both
    /// faces. Sampled by the path tracer; the Whitted paths skip it.
    Rect {
        corner: Point3,
        edge1: Vec3,
        edge2: Vec3,
        normal: Vec3,
        area: f64,
    },
    /// Spherical area light (PxrSphereLight).
    SphereArea { center: Point3, radius: f64 },
    /// Disk area light (PxrDiskLight); e1/e2 span the radius.
    DiskArea {
        center: Point3,
        e1: Vec3,
        e2: Vec3,
        normal: Vec3,
        area: f64,
    },
    /// Environment dome (PxrDomeLight). Radiance from `Light::env` when
    /// present, else constant `radiance()`.
    Dome,
}

#[derive(Clone)]
pub struct Light {
    pub light_type: LightType,
    pub intensity: f64,
    pub color: Vec3,
    /// HDRI for dome lights.
    pub env: Option<Arc<EnvMap>>,
}

impl std::fmt::Debug for Light {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Light")
            .field("light_type", &self.light_type)
            .field("intensity", &self.intensity)
            .field("color", &self.color)
            .field("env", &self.env.is_some())
            .finish()
    }
}

impl Light {
    pub fn point(position: Point3, intensity: f64, color: Vec3) -> Self {
        Self {
            light_type: LightType::Point { position },
            intensity,
            color,
            env: None,
        }
    }

    pub fn distant(direction: Vec3, intensity: f64, color: Vec3) -> Self {
        Self {
            light_type: LightType::Distant {
                direction: direction.normalize(),
                angular_radius: 0.0,
            },
            intensity,
            color,
            env: None,
        }
    }

    pub fn distant_soft(direction: Vec3, angular_radius: f64, intensity: f64, color: Vec3) -> Self {
        Self {
            light_type: LightType::Distant {
                direction: direction.normalize(),
                angular_radius,
            },
            intensity,
            color,
            env: None,
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
            env: None,
        }
    }

    pub fn sphere_area(center: Point3, radius: f64, intensity: f64, color: Vec3) -> Self {
        Self {
            light_type: LightType::SphereArea { center, radius },
            intensity,
            color,
            env: None,
        }
    }

    pub fn disk_area(center: Point3, e1: Vec3, e2: Vec3, intensity: f64, color: Vec3) -> Self {
        let normal = e1.cross(&e2).normalize();
        let area = std::f64::consts::PI * e1.length() * e2.length();
        Self {
            light_type: LightType::DiskArea { center, e1, e2, normal, area },
            intensity,
            color,
            env: None,
        }
    }

    pub fn dome(intensity: f64, color: Vec3, env: Option<Arc<EnvMap>>) -> Self {
        Self {
            light_type: LightType::Dome,
            intensity,
            color,
            env,
        }
    }

    /// Emitted radiance (for area lights) / radiance scale (dome).
    pub fn radiance(&self) -> Vec3 {
        self.color * self.intensity
    }

    /// Representative direction toward the light (Whitted shading; area
    /// lights use their center).
    pub fn direction_from(&self, point: &Point3) -> Vec3 {
        match &self.light_type {
            LightType::Point { position } => (*position - *point).normalize(),
            LightType::Distant { direction, .. } => -*direction,
            LightType::Rect { corner, edge1, edge2, .. } => {
                (*corner + *edge1 * 0.5 + *edge2 * 0.5 - *point).normalize()
            }
            LightType::SphereArea { center, .. } | LightType::DiskArea { center, .. } => {
                (*center - *point).normalize()
            }
            LightType::Dome => Vec3::new(0.0, 1.0, 0.0),
        }
    }

    /// Distance from `point` to the light (infinite for distant/dome).
    pub fn distance_from(&self, point: &Point3) -> f64 {
        match &self.light_type {
            LightType::Point { position } => position.distance(point),
            LightType::Distant { .. } | LightType::Dome => f64::INFINITY,
            LightType::Rect { corner, edge1, edge2, .. } => {
                (*corner + *edge1 * 0.5 + *edge2 * 0.5).distance(point)
            }
            LightType::SphereArea { center, .. } | LightType::DiskArea { center, .. } => {
                center.distance(point)
            }
        }
    }
}
