//! Triangle meshes (PointsPolygons et al.) with a per-mesh BVH (BLAS), and
//! transformed instances gathered under a scene-level TLAS.
//!
//! Storage is GPU-shaped on purpose (f32 positions/normals, u32 indices,
//! flat arrays) so Phase 3 can upload the same buffers; intersection math
//! runs in f64 on the CPU reference path.

use crate::accel::{Aabb, Bvh};
use crate::math::{Matrix4, Point3, Vec3};
use crate::raytracer::{Intersection, Ray};

pub struct Mesh {
    /// xyz triples.
    pub positions: Vec<[f32; 3]>,
    /// Triangle vertex indices, 3 per triangle.
    pub indices: Vec<u32>,
    /// Optional per-vertex shading normals (interpolated when present).
    pub normals: Option<Vec<[f32; 3]>>,
    /// Optional per-vertex texture coordinates (carried for later phases).
    pub st: Option<Vec<[f32; 2]>>,
    pub blas: Bvh,
    pub local_bounds: Aabb,
}

impl Mesh {
    pub fn new(
        positions: Vec<[f32; 3]>,
        indices: Vec<u32>,
        normals: Option<Vec<[f32; 3]>>,
        st: Option<Vec<[f32; 2]>>,
    ) -> Self {
        let tri_count = indices.len() / 3;
        let point = |i: u32| {
            let p = positions[i as usize];
            Point3::new(p[0] as f64, p[1] as f64, p[2] as f64)
        };
        let bounds: Vec<Aabb> = (0..tri_count)
            .map(|t| {
                Aabb::from_points([
                    point(indices[t * 3]),
                    point(indices[t * 3 + 1]),
                    point(indices[t * 3 + 2]),
                ])
            })
            .collect();
        let mut local_bounds = Aabb::empty();
        for b in &bounds {
            local_bounds.grow(b);
        }
        let blas = Bvh::build(&bounds);
        Self { positions, indices, normals, st, blas, local_bounds }
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    #[inline]
    fn vertex(&self, i: u32) -> Point3 {
        let p = self.positions[i as usize];
        Point3::new(p[0] as f64, p[1] as f64, p[2] as f64)
    }

    /// Möller–Trumbore against triangle `tri`, in mesh-local space, with
    /// the parametric t measured along the (unnormalized) ray direction.
    /// (A watertight Woop test replaces this alongside the f32/GPU port.)
    #[inline]
    fn intersect_triangle(&self, ray: &Ray, tri: u32, t_max: f64) -> Option<(f64, f64, f64)> {
        let i0 = self.indices[tri as usize * 3];
        let i1 = self.indices[tri as usize * 3 + 1];
        let i2 = self.indices[tri as usize * 3 + 2];
        let v0 = self.vertex(i0);
        let e1 = self.vertex(i1) - v0;
        let e2 = self.vertex(i2) - v0;
        let p = ray.direction.cross(&e2);
        let det = e1.dot(&p);
        if det.abs() < 1e-14 {
            return None;
        }
        let inv = 1.0 / det;
        let tv = ray.origin - v0;
        let u = tv.dot(&p) * inv;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = tv.cross(&e1);
        let v = ray.direction.dot(&q) * inv;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = e2.dot(&q) * inv;
        (t > 1e-9 && t < t_max).then_some((t, u, v))
    }

    /// Interpolated (or geometric) local-space normal for a hit.
    fn local_normal(&self, tri: u32, u: f64, v: f64) -> Vec3 {
        let i0 = self.indices[tri as usize * 3] as usize;
        let i1 = self.indices[tri as usize * 3 + 1] as usize;
        let i2 = self.indices[tri as usize * 3 + 2] as usize;
        if let Some(normals) = &self.normals {
            let n = |i: usize| {
                let n = normals[i];
                Vec3::new(n[0] as f64, n[1] as f64, n[2] as f64)
            };
            let w = 1.0 - u - v;
            return (n(i0) * w + n(i1) * u + n(i2) * v).normalize();
        }
        let v0 = self.vertex(self.indices[tri as usize * 3]);
        let e1 = self.vertex(self.indices[tri as usize * 3 + 1]) - v0;
        let e2 = self.vertex(self.indices[tri as usize * 3 + 2]) - v0;
        e1.cross(&e2).normalize()
    }
}

/// A placed copy of a mesh: transform + material.
pub struct Instance {
    pub mesh_id: u32,
    pub material_id: usize,
    pub transform: Matrix4,
    pub inverse: Matrix4,
    pub world_bounds: Aabb,
}

impl Instance {
    pub fn new(mesh_id: u32, material_id: usize, transform: Matrix4, mesh: &Mesh) -> Self {
        let inverse = transform.inverse().unwrap_or(Matrix4::identity());
        // World bounds from the 8 transformed corners of the local box.
        let lb = &mesh.local_bounds;
        let mut world_bounds = Aabb::empty();
        for i in 0..8 {
            let corner = Point3::new(
                if i & 1 == 0 { lb.min.x } else { lb.max.x },
                if i & 2 == 0 { lb.min.y } else { lb.max.y },
                if i & 4 == 0 { lb.min.z } else { lb.max.z },
            );
            let w = transform.transform_point(&corner);
            world_bounds.grow_point(&Vec3::new(w.x, w.y, w.z));
        }
        Self { mesh_id, material_id, transform, inverse, world_bounds }
    }

    /// Closest hit in *parametric* t (along the world ray direction).
    pub fn intersect(&self, meshes: &[Mesh], world_ray: &Ray, t_max: f64) -> Option<MeshHit> {
        let mesh = &meshes[self.mesh_id as usize];
        // Affine transform WITHOUT normalizing the direction keeps the
        // parametric t identical in local and world space (Ray::new would
        // normalize, so construct the struct directly).
        let local_ray = Ray {
            origin: self.inverse.transform_point(&world_ray.origin),
            direction: self.inverse.transform_vec(&world_ray.direction),
        };
        let mut hit_uv = (0.0, 0.0);
        let (tri, t) = mesh.blas.traverse(&local_ray, t_max, |tri, cur_max| {
            mesh.intersect_triangle(&local_ray, tri, cur_max).map(|(t, u, v)| {
                hit_uv = (u, v);
                t
            })
        })?;
        // hit_uv holds the uv of the LAST accepted (i.e. winning) triangle.
        let local_n = mesh.local_normal(tri, hit_uv.0, hit_uv.1);
        let world_n = self.inverse.transform_normal(&local_n).normalize();
        Some(MeshHit { t_param: t, tri, world_normal: world_n })
    }

    /// Any hit strictly before parametric `t_limit`.
    pub fn occludes(&self, meshes: &[Mesh], world_ray: &Ray, t_limit: f64) -> bool {
        let mesh = &meshes[self.mesh_id as usize];
        let local_ray = Ray {
            origin: self.inverse.transform_point(&world_ray.origin),
            direction: self.inverse.transform_vec(&world_ray.direction),
        };
        mesh.blas.any_hit(&local_ray, t_limit, |tri, lim| {
            mesh.intersect_triangle(&local_ray, tri, lim).is_some()
        })
    }
}

pub struct MeshHit {
    /// Parametric t along the world ray direction.
    pub t_param: f64,
    pub tri: u32,
    pub world_normal: Vec3,
}

/// Convert a parametric mesh hit into the renderer's Intersection
/// convention (t = euclidean world distance, double-sided normal).
pub fn to_intersection(hit: &MeshHit, ray: &Ray, material_id: usize) -> Intersection {
    let point = ray.at(hit.t_param);
    let mut normal = hit.world_normal;
    if normal.dot(&ray.direction) > 0.0 {
        normal = -normal;
    }
    Intersection::new(point.distance(&ray.origin), point, normal, material_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit cube centered at origin.
    fn cube() -> Mesh {
        let p = |x: f64, y: f64, z: f64| [x as f32, y as f32, z as f32];
        let positions = vec![
            p(-1.0, -1.0, -1.0),
            p(1.0, -1.0, -1.0),
            p(1.0, 1.0, -1.0),
            p(-1.0, 1.0, -1.0),
            p(-1.0, -1.0, 1.0),
            p(1.0, -1.0, 1.0),
            p(1.0, 1.0, 1.0),
            p(-1.0, 1.0, 1.0),
        ];
        let quads: [[u32; 4]; 6] = [
            [0, 1, 2, 3],
            [5, 4, 7, 6],
            [4, 0, 3, 7],
            [1, 5, 6, 2],
            [3, 2, 6, 7],
            [4, 5, 1, 0],
        ];
        let mut indices = Vec::new();
        for q in quads {
            indices.extend_from_slice(&[q[0], q[1], q[2], q[0], q[2], q[3]]);
        }
        Mesh::new(positions, indices, None, None)
    }

    #[test]
    fn cube_instance_hits() {
        let mesh = cube();
        let meshes = vec![mesh];
        let inst = Instance::new(0, 0, Matrix4::translate(0.0, 0.0, 5.0), &meshes[0]);
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0));
        let hit = inst.intersect(&meshes, &ray, f64::INFINITY).expect("hit");
        assert!((hit.t_param - 4.0).abs() < 1e-9, "t = {}", hit.t_param);
        let isect = to_intersection(&hit, &ray, 0);
        assert!((isect.t - 4.0).abs() < 1e-9);
        assert!(isect.normal.z < -0.99, "normal faces the ray: {:?}", isect.normal);
    }

    #[test]
    fn scaled_instance_t_is_world() {
        let mesh = cube();
        let meshes = vec![mesh];
        // Scale 2x: front face lands at z = 5 - 2 = 3.
        let xf = Matrix4::translate(0.0, 0.0, 5.0) * Matrix4::scale(2.0, 2.0, 2.0);
        let inst = Instance::new(0, 0, xf, &meshes[0]);
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0));
        let hit = inst.intersect(&meshes, &ray, f64::INFINITY).expect("hit");
        assert!((hit.t_param - 3.0).abs() < 1e-9, "t = {}", hit.t_param);
    }

    #[test]
    fn occlusion_respects_limit() {
        let mesh = cube();
        let meshes = vec![mesh];
        let inst = Instance::new(0, 0, Matrix4::translate(0.0, 0.0, 5.0), &meshes[0]);
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(inst.occludes(&meshes, &ray, 10.0));
        assert!(!inst.occludes(&meshes, &ray, 3.5));
    }
}
