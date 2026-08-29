use crate::math::{Point3, Vec3};

#[derive(Debug, Clone)]
pub struct Intersection {
    pub t: f64,
    pub point: Point3,
    pub normal: Vec3,
    pub material_id: usize,
}

impl Intersection {
    pub fn new(t: f64, point: Point3, normal: Vec3, material_id: usize) -> Self {
        Self {
            t,
            point,
            normal: normal.normalize(),
            material_id,
        }
    }
}
