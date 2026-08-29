use crate::math::{Point3, Vec3};

#[derive(Debug, Clone)]
pub struct Intersection {
    pub t: f64,
    pub point: Point3,
    pub normal: Vec3,
    pub material_id: usize,
    /// True when the ray hit the geometrically front-facing side (i.e. the
    /// ray origin was outside a closed primitive). Primitives that flip
    /// their reported normal toward the viewer (triangles, meshes) must
    /// record this from the *unflipped* orientation — refraction depends
    /// on it.
    pub front_face: bool,
}

impl Intersection {
    pub fn new(t: f64, point: Point3, normal: Vec3, material_id: usize) -> Self {
        Self {
            t,
            point,
            normal: normal.normalize(),
            material_id,
            front_face: true,
        }
    }

    pub fn with_front_face(mut self, front_face: bool) -> Self {
        self.front_face = front_face;
        self
    }
}
