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
    /// Surface parameterization at the hit (mesh st or quadric parametric
    /// uv); [0,0] when the primitive carries none.
    pub st: [f64; 2],
    /// Approximate st-units per world-unit at the hit (isotropic scalar,
    /// for ray-cone mip selection). 0 = unknown / no st.
    pub st_density: f64,
    /// Curve tangent at the hit (zero for surface geometry) — the hair
    /// BSDF shades against this.
    pub tangent: Vec3,
}

impl Intersection {
    pub fn new(t: f64, point: Point3, normal: Vec3, material_id: usize) -> Self {
        Self {
            t,
            point,
            normal: normal.normalize(),
            material_id,
            front_face: true,
            st: [0.0, 0.0],
            st_density: 0.0,
            tangent: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
        }
    }

    pub fn with_front_face(mut self, front_face: bool) -> Self {
        self.front_face = front_face;
        self
    }

    pub fn with_st(mut self, st: [f64; 2], st_density: f64) -> Self {
        self.st = st;
        self.st_density = st_density;
        self
    }

    pub fn with_tangent(mut self, tangent: Vec3) -> Self {
        self.tangent = tangent;
        self
    }
}
