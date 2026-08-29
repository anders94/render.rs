//! Triangle meshes (PointsPolygons et al.) with a per-mesh BVH (BLAS), and
//! transformed instances gathered under a scene-level TLAS.
//!
//! Storage is GPU-shaped on purpose (f32 positions/normals, u32 indices,
//! flat arrays) so Phase 3 can upload the same buffers; intersection math
//! runs in f64 on the CPU reference path.

use crate::accel::{Aabb, Bvh};
use crate::geometry::curves::CurveSet;
use crate::math::{Matrix4, Point3, Vec3};
use crate::raytracer::{Intersection, Ray};

/// What an Instance points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeomKind {
    Mesh,
    Curves,
}

pub struct Mesh {
    /// xyz triples.
    pub positions: Vec<[f32; 3]>,
    /// Triangle vertex indices, 3 per triangle.
    pub indices: Vec<u32>,
    /// Optional per-vertex shading normals (interpolated when present).
    pub normals: Option<Vec<[f32; 3]>>,
    /// Optional per-vertex texture coordinates (carried for later phases).
    pub st: Option<Vec<[f32; 2]>>,
    /// Deformation-blur endpoint: positions at shutter close (same
    /// topology). BLAS bounds cover both endpoints.
    pub positions1: Option<Vec<[f32; 3]>>,
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
        Self::with_motion(positions, indices, normals, st, None)
    }

    /// Mesh with an optional deformation endpoint (positions at shutter
    /// close); triangle bounds grow to cover both endpoints.
    pub fn with_motion(
        positions: Vec<[f32; 3]>,
        indices: Vec<u32>,
        normals: Option<Vec<[f32; 3]>>,
        st: Option<Vec<[f32; 2]>>,
        positions1: Option<Vec<[f32; 3]>>,
    ) -> Self {
        let positions1 = positions1.filter(|p| p.len() == positions.len());
        let tri_count = indices.len() / 3;
        let point = |buf: &[[f32; 3]], i: u32| {
            let p = buf[i as usize];
            Point3::new(p[0] as f64, p[1] as f64, p[2] as f64)
        };
        let bounds: Vec<Aabb> = (0..tri_count)
            .map(|t| {
                let mut b = Aabb::from_points([
                    point(&positions, indices[t * 3]),
                    point(&positions, indices[t * 3 + 1]),
                    point(&positions, indices[t * 3 + 2]),
                ]);
                if let Some(p1) = &positions1 {
                    b.grow(&Aabb::from_points([
                        point(p1, indices[t * 3]),
                        point(p1, indices[t * 3 + 1]),
                        point(p1, indices[t * 3 + 2]),
                    ]));
                }
                b
            })
            .collect();
        let mut local_bounds = Aabb::empty();
        for b in &bounds {
            local_bounds.grow(b);
        }
        let blas = Bvh::build(&bounds);
        Self { positions, indices, normals, st, positions1, blas, local_bounds }
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    #[inline]
    fn vertex(&self, i: u32) -> Point3 {
        let p = self.positions[i as usize];
        Point3::new(p[0] as f64, p[1] as f64, p[2] as f64)
    }

    /// Vertex at shutter time (lerped when a deformation endpoint exists).
    #[inline]
    fn vertex_at(&self, i: u32, time: f64) -> Point3 {
        match &self.positions1 {
            Some(p1) if time > 0.0 => {
                let a = self.positions[i as usize];
                let b = p1[i as usize];
                Point3::new(
                    a[0] as f64 + (b[0] as f64 - a[0] as f64) * time,
                    a[1] as f64 + (b[1] as f64 - a[1] as f64) * time,
                    a[2] as f64 + (b[2] as f64 - a[2] as f64) * time,
                )
            }
            _ => self.vertex(i),
        }
    }

    /// Möller–Trumbore against triangle `tri`, in mesh-local space, with
    /// the parametric t measured along the (unnormalized) ray direction.
    /// (A watertight Woop test replaces this alongside the f32/GPU port.)
    #[inline]
    fn intersect_triangle(&self, ray: &Ray, tri: u32, t_max: f64) -> Option<(f64, f64, f64)> {
        let i0 = self.indices[tri as usize * 3];
        let i1 = self.indices[tri as usize * 3 + 1];
        let i2 = self.indices[tri as usize * 3 + 2];
        let v0 = self.vertex_at(i0, ray.time);
        let e1 = self.vertex_at(i1, ray.time) - v0;
        let e2 = self.vertex_at(i2, ray.time) - v0;
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

    /// Interpolated st at a hit, plus the triangle's st-density (st units
    /// per local-space unit, the sqrt of st-area over surface area).
    fn st_at(&self, tri: u32, u: f64, v: f64) -> Option<([f64; 2], f64)> {
        let st = self.st.as_ref()?;
        let i0 = self.indices[tri as usize * 3] as usize;
        let i1 = self.indices[tri as usize * 3 + 1] as usize;
        let i2 = self.indices[tri as usize * 3 + 2] as usize;
        let (s0, s1, s2) = (st[i0], st[i1], st[i2]);
        let w = 1.0 - u - v;
        let s = s0[0] as f64 * w + s1[0] as f64 * u + s2[0] as f64 * v;
        let t = s0[1] as f64 * w + s1[1] as f64 * u + s2[1] as f64 * v;
        // st area vs geometric area of the triangle.
        let st_area = 0.5
            * ((s1[0] - s0[0]) as f64 * (s2[1] - s0[1]) as f64
                - (s2[0] - s0[0]) as f64 * (s1[1] - s0[1]) as f64)
                .abs();
        let v0 = self.vertex(i0 as u32);
        let e1 = self.vertex(i1 as u32) - v0;
        let e2 = self.vertex(i2 as u32) - v0;
        let geo_area = 0.5 * e1.cross(&e2).length();
        let density = if geo_area > 1e-18 { (st_area / geo_area).sqrt() } else { 0.0 };
        Some(([s, t], density))
    }

    /// Interpolated (or geometric, at shutter time) local-space normal.
    fn local_normal(&self, tri: u32, u: f64, v: f64, time: f64) -> Vec3 {
        let i0 = self.indices[tri as usize * 3] as usize;
        let i1 = self.indices[tri as usize * 3 + 1] as usize;
        let i2 = self.indices[tri as usize * 3 + 2] as usize;
        // Vertex normals are authored for the rest pose; a deforming mesh
        // falls back to the time-correct geometric normal.
        if self.positions1.is_none() {
            if let Some(normals) = &self.normals {
                let n = |i: usize| {
                    let n = normals[i];
                    Vec3::new(n[0] as f64, n[1] as f64, n[2] as f64)
                };
                let w = 1.0 - u - v;
                return (n(i0) * w + n(i1) * u + n(i2) * v).normalize();
            }
        }
        let v0 = self.vertex_at(i0 as u32, time);
        let e1 = self.vertex_at(i1 as u32, time) - v0;
        let e2 = self.vertex_at(i2 as u32, time) - v0;
        e1.cross(&e2).normalize()
    }
}

/// A placed copy of a mesh or curve set: transform + material.
pub struct Instance {
    pub kind: GeomKind,
    /// Index into Scene::meshes or Scene::curve_sets, per `kind`.
    pub mesh_id: u32,
    pub material_id: usize,
    pub transform: Matrix4,
    pub inverse: Matrix4,
    /// Transform-motion endpoint (shutter close); world bounds cover both.
    /// Interpolation is component-wise, fine for the small per-frame
    /// rotations shutters actually see.
    pub transform1: Option<Matrix4>,
    pub world_bounds: Aabb,
    /// Isotropic length scale of the transform (for st-density transfer
    /// from local to world space).
    pub scale: f64,
}

impl Instance {
    pub fn new(mesh_id: u32, material_id: usize, transform: Matrix4, mesh: &Mesh) -> Self {
        Self::with_motion(mesh_id, material_id, transform, None, mesh)
    }

    pub fn with_motion(
        mesh_id: u32,
        material_id: usize,
        transform: Matrix4,
        transform1: Option<Matrix4>,
        mesh: &Mesh,
    ) -> Self {
        Self::build(GeomKind::Mesh, mesh_id, material_id, transform, transform1, &mesh.local_bounds)
    }

    pub fn new_curves(
        set_id: u32,
        material_id: usize,
        transform: Matrix4,
        transform1: Option<Matrix4>,
        set: &CurveSet,
    ) -> Self {
        Self::build(GeomKind::Curves, set_id, material_id, transform, transform1, &set.local_bounds)
    }

    fn build(
        kind: GeomKind,
        mesh_id: u32,
        material_id: usize,
        transform: Matrix4,
        transform1: Option<Matrix4>,
        local_bounds: &Aabb,
    ) -> Self {
        let inverse = transform.inverse().unwrap_or(Matrix4::identity());
        // World bounds from the 8 transformed corners of the local box, at
        // both shutter endpoints when moving.
        let lb = local_bounds;
        let mut world_bounds = Aabb::empty();
        for xf in std::iter::once(&transform).chain(transform1.iter()) {
            for i in 0..8 {
                let corner = Point3::new(
                    if i & 1 == 0 { lb.min.x } else { lb.max.x },
                    if i & 2 == 0 { lb.min.y } else { lb.max.y },
                    if i & 4 == 0 { lb.min.z } else { lb.max.z },
                );
                let w = xf.transform_point(&corner);
                world_bounds.grow_point(&Vec3::new(w.x, w.y, w.z));
            }
        }
        let scale = transform.approx_scale();
        Self { kind, mesh_id, material_id, transform, inverse, transform1, world_bounds, scale }
    }

    /// (forward, inverse) at shutter time.
    #[inline]
    fn transforms_at(&self, time: f64) -> (Matrix4, Matrix4) {
        match &self.transform1 {
            Some(t1) if time > 0.0 => {
                let fwd = self.transform.lerp(t1, time);
                let inv = fwd.inverse().unwrap_or(Matrix4::identity());
                (fwd, inv)
            }
            _ => (self.transform, self.inverse),
        }
    }

    /// Closest hit in *parametric* t (along the world ray direction).
    pub fn intersect(
        &self,
        meshes: &[Mesh],
        curves: &[CurveSet],
        world_ray: &Ray,
        t_max: f64,
    ) -> Option<MeshHit> {
        let (_, inverse) = self.transforms_at(world_ray.time);
        // Affine transform WITHOUT normalizing the direction keeps the
        // parametric t identical in local and world space (Ray::new would
        // normalize, so construct the struct directly).
        let local_ray = Ray {
            origin: inverse.transform_point(&world_ray.origin),
            direction: inverse.transform_vec(&world_ray.direction),
            time: world_ray.time,
        };
        if self.kind == GeomKind::Curves {
            let set = &curves[self.mesh_id as usize];
            let mut best: Option<(Vec3, Vec3, f64)> = None;
            let (_, t) = set.blas.traverse(&local_ray, t_max, |seg, cur_max| {
                set.intersect_segment(&local_ray, seg, cur_max).map(|(t, n, tang, v)| {
                    best = Some((n, tang, v));
                    t
                })
            })?;
            let (n, tang, v) = best?;
            let world_n = inverse.transform_normal(&n).normalize();
            let world_t = self.transform.transform_vec(&tang).normalize();
            return Some(MeshHit {
                t_param: t,
                tri: 0,
                world_normal: world_n,
                st: [0.5, v],
                st_density: 0.0,
                tangent: world_t,
            });
        }
        let mesh = &meshes[self.mesh_id as usize];
        let mut hit_uv = (0.0, 0.0);
        let (tri, t) = mesh.blas.traverse(&local_ray, t_max, |tri, cur_max| {
            mesh.intersect_triangle(&local_ray, tri, cur_max).map(|(t, u, v)| {
                hit_uv = (u, v);
                t
            })
        })?;
        // hit_uv holds the uv of the LAST accepted (i.e. winning) triangle.
        let local_n = mesh.local_normal(tri, hit_uv.0, hit_uv.1, world_ray.time);
        let world_n = inverse.transform_normal(&local_n).normalize();
        let (st, st_density) = mesh
            .st_at(tri, hit_uv.0, hit_uv.1)
            .map(|(st, d)| (st, d / self.scale))
            .unwrap_or(([0.0, 0.0], 0.0));
        Some(MeshHit {
            t_param: t,
            tri,
            world_normal: world_n,
            st,
            st_density,
            tangent: Vec3::zero(),
        })
    }

    /// Any hit strictly before parametric `t_limit`.
    pub fn occludes(
        &self,
        meshes: &[Mesh],
        curves: &[CurveSet],
        world_ray: &Ray,
        t_limit: f64,
    ) -> bool {
        let (_, inverse) = self.transforms_at(world_ray.time);
        let local_ray = Ray {
            origin: inverse.transform_point(&world_ray.origin),
            direction: inverse.transform_vec(&world_ray.direction),
            time: world_ray.time,
        };
        if self.kind == GeomKind::Curves {
            let set = &curves[self.mesh_id as usize];
            return set.blas.any_hit(&local_ray, t_limit, |seg, lim| {
                set.intersect_segment(&local_ray, seg, lim).is_some()
            });
        }
        let mesh = &meshes[self.mesh_id as usize];
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
    pub st: [f64; 2],
    /// st units per world unit (0 when the mesh has no st).
    pub st_density: f64,
    /// Curve tangent at the hit (zero for surface geometry).
    pub tangent: Vec3,
}

/// Convert a parametric mesh hit into the renderer's Intersection
/// convention (t = euclidean world distance, double-sided normal).
pub fn to_intersection(hit: &MeshHit, ray: &Ray, material_id: usize) -> Intersection {
    let point = ray.at(hit.t_param);
    let mut normal = hit.world_normal;
    let front_face = normal.dot(&ray.direction) < 0.0;
    if !front_face {
        normal = -normal;
    }
    Intersection::new(point.distance(&ray.origin), point, normal, material_id)
        .with_front_face(front_face)
        .with_st(hit.st, hit.st_density)
        .with_tangent(hit.tangent)
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
        let hit = inst.intersect(&meshes, &[], &ray, f64::INFINITY).expect("hit");
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
        let hit = inst.intersect(&meshes, &[], &ray, f64::INFINITY).expect("hit");
        assert!((hit.t_param - 3.0).abs() < 1e-9, "t = {}", hit.t_param);
    }

    #[test]
    fn st_interpolates_and_density_scales() {
        // Unit quad in xy with st == xy: density 1 in local space.
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let st = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let mesh = Mesh::new(positions, vec![0, 1, 2, 0, 2, 3], None, Some(st));
        let meshes = vec![mesh];
        // Scale 2x: st density halves in world space.
        let inst = Instance::new(0, 0, Matrix4::scale(2.0, 2.0, 2.0), &meshes[0]);
        let ray = Ray::new(Point3::new(0.5, 1.0, -3.0), Vec3::new(0.0, 0.0, 1.0));
        let hit = inst.intersect(&meshes, &[], &ray, f64::INFINITY).expect("hit");
        assert!((hit.st[0] - 0.25).abs() < 1e-6, "s = {}", hit.st[0]);
        assert!((hit.st[1] - 0.5).abs() < 1e-6, "t = {}", hit.st[1]);
        assert!((hit.st_density - 0.5).abs() < 1e-6, "density = {}", hit.st_density);
    }

    #[test]
    fn occlusion_respects_limit() {
        let mesh = cube();
        let meshes = vec![mesh];
        let inst = Instance::new(0, 0, Matrix4::translate(0.0, 0.0, 5.0), &meshes[0]);
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(inst.occludes(&meshes, &[], &ray, 10.0));
        assert!(!inst.occludes(&meshes, &[], &ray, 3.5));
    }
}
