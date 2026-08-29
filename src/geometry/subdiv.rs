//! Uniform Catmull-Clark subdivision (roadmap Phase 5) over arbitrary
//! polygon cages, with crease/corner sharpness (Pixar semi-sharp decay),
//! boundary interpolation, and hole faces. The subdivided cage is
//! triangulated into the standard Mesh (BLAS) with smoothed vertex
//! normals, so every backend renders it unchanged.
//!
//! Uniform-only for now: adaptive/screen-space dicing is a later phase
//! (OpenSubdiv-via-FFI is the fallback plan before hand-rolling that).

use crate::geometry::Mesh;
use std::collections::HashMap;

pub struct SubdivCage {
    pub positions: Vec<[f64; 3]>,
    /// Faces as vertex-index rings (arbitrary n-gons).
    pub faces: Vec<Vec<u32>>,
    /// (v0, v1) -> sharpness. Order-independent keying handled internally.
    pub crease_edges: Vec<(u32, u32, f64)>,
    /// vertex id -> sharpness.
    pub corners: Vec<(u32, f64)>,
    /// Face indices removed from the surface.
    pub holes: Vec<u32>,
    /// RiSpec "interpolateboundary": pin boundary edges/vertices.
    pub interpolate_boundary: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct EdgeKey(u32, u32);

impl EdgeKey {
    fn new(a: u32, b: u32) -> Self {
        if a < b { Self(a, b) } else { Self(b, a) }
    }
}

struct Topology {
    /// Per-edge: adjacent face ids.
    edge_faces: HashMap<EdgeKey, Vec<u32>>,
    /// Per-vertex: incident edges.
    vertex_edges: HashMap<u32, Vec<EdgeKey>>,
    /// Per-vertex: incident faces.
    vertex_faces: HashMap<u32, Vec<u32>>,
}

fn build_topology(faces: &[Vec<u32>]) -> Topology {
    let mut edge_faces: HashMap<EdgeKey, Vec<u32>> = HashMap::new();
    let mut vertex_edges: HashMap<u32, Vec<EdgeKey>> = HashMap::new();
    let mut vertex_faces: HashMap<u32, Vec<u32>> = HashMap::new();
    for (fi, face) in faces.iter().enumerate() {
        for i in 0..face.len() {
            let a = face[i];
            let b = face[(i + 1) % face.len()];
            let key = EdgeKey::new(a, b);
            edge_faces.entry(key).or_default().push(fi as u32);
            for v in [a, b] {
                let entry = vertex_edges.entry(v).or_default();
                if !entry.contains(&key) {
                    entry.push(key);
                }
            }
            let vf = vertex_faces.entry(a).or_default();
            if !vf.contains(&(fi as u32)) {
                vf.push(fi as u32);
            }
        }
    }
    Topology { edge_faces, vertex_edges, vertex_faces }
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn lerp3(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

impl SubdivCage {
    /// One Catmull-Clark round; sharpnesses decay by 1 (semi-sharp creases).
    fn subdivide_once(&self) -> SubdivCage {
        let topo = build_topology(&self.faces);
        let nv = self.positions.len();

        let mut crease: HashMap<EdgeKey, f64> = HashMap::new();
        for &(a, b, s) in &self.crease_edges {
            crease.insert(EdgeKey::new(a, b), s.max(0.0));
        }
        let corner: HashMap<u32, f64> = self
            .corners
            .iter()
            .map(|&(v, s)| (v, s.max(0.0)))
            .collect();
        let is_hole: Vec<bool> = {
            let mut h = vec![false; self.faces.len()];
            for &f in &self.holes {
                if (f as usize) < h.len() {
                    h[f as usize] = true;
                }
            }
            h
        };

        // Face points.
        let mut new_positions: Vec<[f64; 3]> = Vec::new();
        let face_point_idx: Vec<u32> = self
            .faces
            .iter()
            .map(|face| {
                let mut c = [0.0; 3];
                for &v in face {
                    c = add3(c, self.positions[v as usize]);
                }
                new_positions.push(scale3(c, 1.0 / face.len() as f64));
                (new_positions.len() - 1) as u32
            })
            .collect();

        // Edge points.
        let mut edge_point_idx: HashMap<EdgeKey, u32> = HashMap::new();
        for (key, faces) in &topo.edge_faces {
            let a = self.positions[key.0 as usize];
            let b = self.positions[key.1 as usize];
            let mid = scale3(add3(a, b), 0.5);
            let sharp = crease.get(key).copied().unwrap_or(0.0);
            let boundary = faces.len() < 2;
            let point = if boundary || sharp >= 1.0 {
                mid
            } else {
                let mut fp = [0.0; 3];
                for &f in faces {
                    fp = add3(fp, new_positions[face_point_idx[f as usize] as usize]);
                }
                let smooth = scale3(
                    add3(add3(a, b), fp),
                    1.0 / (2.0 + faces.len() as f64),
                );
                if sharp > 0.0 {
                    lerp3(smooth, mid, sharp.min(1.0))
                } else {
                    smooth
                }
            };
            new_positions.push(point);
            edge_point_idx.insert(*key, (new_positions.len() - 1) as u32);
        }

        // Vertex points.
        let mut vertex_point_idx: Vec<u32> = Vec::with_capacity(nv);
        for v in 0..nv as u32 {
            let p = self.positions[v as usize];
            let edges = topo.vertex_edges.get(&v).cloned().unwrap_or_default();
            let vfaces = topo.vertex_faces.get(&v).cloned().unwrap_or_default();
            let boundary_edges: Vec<EdgeKey> = edges
                .iter()
                .copied()
                .filter(|e| topo.edge_faces.get(e).map(|f| f.len() < 2).unwrap_or(true))
                .collect();
            let sharp_edges: Vec<(EdgeKey, f64)> = edges
                .iter()
                .filter_map(|e| crease.get(e).map(|s| (*e, *s)))
                .filter(|(_, s)| *s > 0.0)
                .collect();
            let corner_sharp = corner.get(&v).copied().unwrap_or(0.0);

            let smooth_point = |edges: &[EdgeKey], vfaces: &[u32]| -> [f64; 3] {
                let n = edges.len().max(1) as f64;
                let mut fsum = [0.0; 3];
                for &f in vfaces {
                    fsum = add3(fsum, new_positions[face_point_idx[f as usize] as usize]);
                }
                let favg = scale3(fsum, 1.0 / vfaces.len().max(1) as f64);
                let mut msum = [0.0; 3];
                for e in edges {
                    let mid = scale3(
                        add3(
                            self.positions[e.0 as usize],
                            self.positions[e.1 as usize],
                        ),
                        0.5,
                    );
                    msum = add3(msum, mid);
                }
                let mavg = scale3(msum, 1.0 / n);
                // (F + 2M + (n-3)P) / n
                scale3(
                    add3(add3(favg, scale3(mavg, 2.0)), scale3(p, n - 3.0)),
                    1.0 / n,
                )
            };
            let crease_point = |pair: &[(EdgeKey, f64)]| -> [f64; 3] {
                // (6P + A + B) / 8 along the crease chain.
                let other = |e: &EdgeKey| if e.0 == v { e.1 } else { e.0 };
                let a = self.positions[other(&pair[0].0) as usize];
                let b = self.positions[other(&pair[1].0) as usize];
                scale3(add3(add3(scale3(p, 6.0), a), b), 1.0 / 8.0)
            };

            let point = if corner_sharp >= 1.0 {
                p
            } else if self.interpolate_boundary && boundary_edges.len() >= 2 {
                // Pinned boundary: crease rule along boundary edges.
                let pair: Vec<(EdgeKey, f64)> =
                    boundary_edges.iter().map(|e| (*e, 1.0)).collect();
                crease_point(&pair[..2])
            } else if sharp_edges.len() >= 3 {
                p // pinned at crease junctions
            } else if sharp_edges.len() == 2 {
                let avg_sharp = (sharp_edges[0].1 + sharp_edges[1].1) * 0.5;
                let cp = crease_point(&sharp_edges);
                if avg_sharp >= 1.0 {
                    cp
                } else {
                    lerp3(smooth_point(&edges, &vfaces), cp, avg_sharp)
                }
            } else {
                let sp = smooth_point(&edges, &vfaces);
                if corner_sharp > 0.0 {
                    lerp3(sp, p, corner_sharp.min(1.0))
                } else {
                    sp
                }
            };
            new_positions.push(point);
            vertex_point_idx.push((new_positions.len() - 1) as u32);
        }

        // New faces: one quad per (vertex, face) corner. Holes drop out.
        let mut new_faces = Vec::new();
        for (fi, face) in self.faces.iter().enumerate() {
            if is_hole[fi] {
                continue;
            }
            let fp = face_point_idx[fi];
            let n = face.len();
            for i in 0..n {
                let prev = face[(i + n - 1) % n];
                let cur = face[i];
                let next = face[(i + 1) % n];
                let e_prev = edge_point_idx[&EdgeKey::new(prev, cur)];
                let e_next = edge_point_idx[&EdgeKey::new(cur, next)];
                new_faces.push(vec![vertex_point_idx[cur as usize], e_next, fp, e_prev]);
            }
        }

        // Sharpness decays by one per level; crease edges map to their two
        // child edges (vertex-point -> edge-point).
        let mut new_creases = Vec::new();
        for &(a, b, s) in &self.crease_edges {
            let ns = s - 1.0;
            if ns <= 0.0 {
                continue;
            }
            if let Some(&ep) = edge_point_idx.get(&EdgeKey::new(a, b)) {
                new_creases.push((vertex_point_idx[a as usize], ep, ns));
                new_creases.push((vertex_point_idx[b as usize], ep, ns));
            }
        }
        let new_corners = self
            .corners
            .iter()
            .filter_map(|&(vtx, s)| {
                let ns = s - 1.0;
                (ns > 0.0 || s >= 1e9).then(|| {
                    (
                        vertex_point_idx[vtx as usize],
                        if s >= 1e9 { s } else { ns },
                    )
                })
            })
            .collect();

        SubdivCage {
            positions: new_positions,
            faces: new_faces,
            crease_edges: new_creases,
            corners: new_corners,
            holes: Vec::new(),
            interpolate_boundary: self.interpolate_boundary,
        }
    }

    /// Uniformly subdivide `levels` times and triangulate into a Mesh with
    /// area-weighted smooth vertex normals.
    pub fn tessellate(mut self, levels: u32) -> Mesh {
        // Infinite sharpness convention: RiSpec uses large values; keep
        // them pinned by mapping >= 10 to "infinite" (1e9 survives decay).
        for c in &mut self.crease_edges {
            if c.2 >= 10.0 {
                c.2 = 1e9;
            }
        }
        for c in &mut self.corners {
            if c.1 >= 10.0 {
                c.1 = 1e9;
            }
        }
        let mut cage = self;
        for _ in 0..levels {
            cage = cage.subdivide_once();
        }

        let positions: Vec<[f32; 3]> = cage
            .positions
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();
        let mut indices = Vec::new();
        for face in &cage.faces {
            for i in 1..face.len() - 1 {
                indices.push(face[0]);
                indices.push(face[i]);
                indices.push(face[i + 1]);
            }
        }
        let normals = smooth_normals(&positions, &indices);
        Mesh::new(positions, indices, Some(normals), None)
    }
}

/// Area-weighted vertex normals.
pub fn smooth_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut acc = vec![[0.0f64; 3]; positions.len()];
    for tri in indices.chunks_exact(3) {
        let p = |i: usize| {
            let v = positions[tri[i] as usize];
            [v[0] as f64, v[1] as f64, v[2] as f64]
        };
        let (a, b, c) = (p(0), p(1), p(2));
        let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        for &vi in tri {
            acc[vi as usize] = add3(acc[vi as usize], n);
        }
    }
    acc.iter()
        .map(|n| {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
            [
                (n[0] / len) as f32,
                (n[1] / len) as f32,
                (n[2] / len) as f32,
            ]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube_cage(creases: Vec<(u32, u32, f64)>) -> SubdivCage {
        let positions = vec![
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ];
        let faces = vec![
            vec![0, 1, 2, 3],
            vec![5, 4, 7, 6],
            vec![4, 0, 3, 7],
            vec![1, 5, 6, 2],
            vec![3, 2, 6, 7],
            vec![4, 5, 1, 0],
        ];
        SubdivCage {
            positions,
            faces,
            crease_edges: creases,
            corners: Vec::new(),
            holes: Vec::new(),
            interpolate_boundary: true,
        }
    }

    #[test]
    fn cube_rounds_toward_sphere() {
        let mesh = cube_cage(Vec::new()).tessellate(3);
        assert!(mesh.triangle_count() > 500);
        // All vertices pulled inside the cube and roughly equidistant.
        let radii: Vec<f64> = mesh
            .positions
            .iter()
            .map(|p| ((p[0] as f64).powi(2) + (p[1] as f64).powi(2) + (p[2] as f64).powi(2)).sqrt())
            .collect();
        let rmin = radii.iter().cloned().fold(f64::INFINITY, f64::min);
        let rmax = radii.iter().cloned().fold(0.0, f64::max);
        assert!(rmax < 1.5, "vertices escaped the cage: {rmax}");
        assert!(
            rmax / rmin < 1.35,
            "smooth subdiv cube should be near-round: {rmin:.3}..{rmax:.3}"
        );
    }

    #[test]
    fn infinite_crease_keeps_edge_sharp() {
        // Crease the four edges of the z=-1 face: those corners stay put.
        let creases = vec![
            (0, 1, 1e9),
            (1, 2, 1e9),
            (2, 3, 1e9),
            (3, 0, 1e9),
        ];
        let mesh = cube_cage(creases).tessellate(3);
        // Some vertex must remain at the original creased corner ring
        // (z=-1, |x|=|y|=1 corners stay pinned along the crease chain).
        let on_plane = mesh
            .positions
            .iter()
            .filter(|p| (p[2] + 1.0).abs() < 1e-4)
            .count();
        assert!(on_plane > 20, "creased face ring collapsed: {on_plane} verts on plane");
        // The crease curve is a smooth B-spline through the chain: corners
        // round slightly (≈0.92) but stay far outside the smooth-subdiv
        // shrink (≈0.68).
        let max_xy = mesh
            .positions
            .iter()
            .filter(|p| (p[2] + 1.0).abs() < 1e-4)
            .map(|p| p[0].abs().max(p[1].abs()) as f64)
            .fold(0.0, f64::max);
        assert!(
            max_xy > 0.9,
            "crease boundary over-shrank: max |xy| = {max_xy}"
        );
    }

    #[test]
    fn corner_tags_pin_vertices() {
        let creases = vec![(0, 1, 1e9), (1, 2, 1e9), (2, 3, 1e9), (3, 0, 1e9)];
        let mut cage = cube_cage(creases);
        cage.corners = vec![(0, 1e9), (1, 1e9), (2, 1e9), (3, 1e9)];
        let mesh = cage.tessellate(3);
        // Tagged corners survive at exactly (±1, ±1, -1).
        let pinned = mesh
            .positions
            .iter()
            .filter(|p| {
                (p[0].abs() - 1.0).abs() < 1e-5
                    && (p[1].abs() - 1.0).abs() < 1e-5
                    && (p[2] + 1.0).abs() < 1e-5
            })
            .count();
        assert!(pinned >= 4, "tagged corners not pinned: {pinned}");
    }

    #[test]
    fn hole_faces_removed() {
        let mut cage = cube_cage(Vec::new());
        cage.holes = vec![0];
        let with_hole = cage.tessellate(1).triangle_count();
        let without = cube_cage(Vec::new()).tessellate(1).triangle_count();
        assert!(with_hole < without);
    }
}
