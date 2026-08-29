use crate::math::Matrix4;
use crate::raytracer::{Intersection, Ray};

/// Type-specific parameters of a primitive, for backends that need the raw
/// geometry (e.g. the GPU renderer flattens the scene through this).
#[derive(Debug, Clone, Copy)]
pub enum PrimitiveKind {
    Sphere { radius: f64, zmin: f64, zmax: f64, thetamax: f64 },
    Cylinder { radius: f64, zmin: f64, zmax: f64, thetamax: f64 },
    Cone { height: f64, radius: f64, thetamax: f64 },
    Torus { major_radius: f64, minor_radius: f64, phimin: f64, phimax: f64, thetamax: f64 },
    Disk { height: f64, radius: f64, thetamax: f64 },
    Paraboloid { rmax: f64, zmin: f64, zmax: f64, thetamax: f64 },
    Hyperboloid { p1: [f64; 3], p2: [f64; 3], thetamax: f64 },
    /// Object-space triangle (polygons are fan-triangulated at build time).
    Triangle { v0: [f64; 3], v1: [f64; 3], v2: [f64; 3] },
}

/// Backend-independent description of a scene object.
#[derive(Debug, Clone, Copy)]
pub struct PrimitiveDesc {
    pub kind: PrimitiveKind,
    pub material_id: usize,
    pub transform: Matrix4,
    pub inverse_transform: Matrix4,
}

pub trait Intersectable: Send + Sync {
    fn intersect(&self, ray: &Ray) -> Option<Intersection>;
    fn describe(&self) -> PrimitiveDesc;
}
