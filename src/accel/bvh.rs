//! Binned-SAH bounding volume hierarchy with a compact flat node array and
//! stack-based traversal. Generic over what lives in the leaves: callers
//! provide per-primitive bounds to `build` and a leaf-intersection closure
//! to `traverse`.

use crate::math::{Point3, Vec3};
use crate::raytracer::Ray;

#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn empty() -> Self {
        Self {
            min: Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            max: Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }

    pub fn from_points(points: impl IntoIterator<Item = Point3>) -> Self {
        let mut aabb = Self::empty();
        for p in points {
            aabb.grow_point(&Vec3::new(p.x, p.y, p.z));
        }
        aabb
    }

    pub fn grow_point(&mut self, p: &Vec3) {
        self.min = Vec3::new(self.min.x.min(p.x), self.min.y.min(p.y), self.min.z.min(p.z));
        self.max = Vec3::new(self.max.x.max(p.x), self.max.y.max(p.y), self.max.z.max(p.z));
    }

    pub fn grow(&mut self, other: &Aabb) {
        self.grow_point(&other.min);
        self.grow_point(&other.max);
    }

    pub fn centroid(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn surface_area(&self) -> f64 {
        let d = self.max - self.min;
        if d.x < 0.0 || d.y < 0.0 || d.z < 0.0 {
            return 0.0;
        }
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// Slab test; returns entry distance if the ray hits before `t_max`.
    #[inline]
    pub fn hit(&self, origin: &Vec3, inv_dir: &Vec3, t_max: f64) -> Option<f64> {
        let t0x = (self.min.x - origin.x) * inv_dir.x;
        let t1x = (self.max.x - origin.x) * inv_dir.x;
        let (mut t_near, mut t_far) = (t0x.min(t1x), t0x.max(t1x));
        let t0y = (self.min.y - origin.y) * inv_dir.y;
        let t1y = (self.max.y - origin.y) * inv_dir.y;
        t_near = t_near.max(t0y.min(t1y));
        t_far = t_far.min(t0y.max(t1y));
        let t0z = (self.min.z - origin.z) * inv_dir.z;
        let t1z = (self.max.z - origin.z) * inv_dir.z;
        t_near = t_near.max(t0z.min(t1z));
        t_far = t_far.min(t0z.max(t1z));
        if t_near <= t_far && t_far > 0.0 && t_near < t_max {
            Some(t_near.max(0.0))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
struct Node {
    bounds: Aabb,
    /// Interior: index of the left child (right = left + 1 in `nodes`).
    /// Leaf: first index into `prim_order`.
    first: u32,
    /// 0 for interior nodes; number of primitives for leaves.
    count: u32,
}

pub struct Bvh {
    nodes: Vec<Node>,
    /// Primitive ids in leaf order.
    prim_order: Vec<u32>,
}

const BINS: usize = 16;
const MAX_LEAF: usize = 4;

impl Bvh {
    /// Build over per-primitive bounds. Empty input yields an empty BVH.
    pub fn build(bounds: &[Aabb]) -> Self {
        let mut prim_order: Vec<u32> = (0..bounds.len() as u32).collect();
        let mut nodes = Vec::with_capacity(bounds.len() * 2);
        if bounds.is_empty() {
            return Self { nodes, prim_order };
        }
        let centroids: Vec<Vec3> = bounds.iter().map(Aabb::centroid).collect();
        nodes.push(Node { bounds: Aabb::empty(), first: 0, count: 0 });
        let mut builder = Builder { bounds, centroids: &centroids, nodes, prim_order: &mut prim_order };
        builder.subdivide(0, 0, bounds.len());
        let nodes = builder.nodes;
        Self { nodes, prim_order }
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Leaf-order primitive ids (for exporters that reorder primitive
    /// buffers into leaf order).
    pub fn prim_order(&self) -> &[u32] {
        &self.prim_order
    }

    /// Node data for GPU export: (min, max, first_or_left, count).
    /// Interior nodes: first_or_left = left-child index, count = 0.
    /// Leaves: first_or_left = first slot in prim_order, count = prims.
    pub fn node_views(&self) -> impl Iterator<Item = (Vec3, Vec3, u32, u32)> + '_ {
        self.nodes
            .iter()
            .map(|n| (n.bounds.min, n.bounds.max, n.first, n.count))
    }

    pub fn root_bounds(&self) -> Option<Aabb> {
        self.nodes.first().map(|n| n.bounds)
    }

    /// Find the closest hit. `intersect_prim(prim_id, t_max) -> Option<t>`
    /// must return a hit strictly closer than the given `t_max`; the caller
    /// records its own payload for the winning id. Returns the winning
    /// (prim_id, t).
    pub fn traverse<F>(&self, ray: &Ray, mut t_max: f64, mut intersect_prim: F) -> Option<(u32, f64)>
    where
        F: FnMut(u32, f64) -> Option<f64>,
    {
        if self.nodes.is_empty() {
            return None;
        }
        let origin = Vec3::new(ray.origin.x, ray.origin.y, ray.origin.z);
        let inv_dir = Vec3::new(
            1.0 / ray.direction.x,
            1.0 / ray.direction.y,
            1.0 / ray.direction.z,
        );
        let mut best: Option<(u32, f64)> = None;
        let mut stack: [u32; 64] = [0; 64];
        let mut sp = 0usize;
        stack[sp] = 0;
        sp += 1;

        while sp > 0 {
            sp -= 1;
            let node = &self.nodes[stack[sp] as usize];
            if node.bounds.hit(&origin, &inv_dir, t_max).is_none() {
                continue;
            }
            if node.count > 0 {
                for i in node.first..node.first + node.count {
                    let prim = self.prim_order[i as usize];
                    if let Some(t) = intersect_prim(prim, t_max) {
                        if t < t_max {
                            t_max = t;
                            best = Some((prim, t));
                        }
                    }
                }
            } else {
                // Push the farther child first so the nearer pops first.
                let l = node.first as usize;
                let r = l + 1;
                let dl = self.nodes[l].bounds.hit(&origin, &inv_dir, t_max);
                let dr = self.nodes[r].bounds.hit(&origin, &inv_dir, t_max);
                match (dl, dr) {
                    (Some(a), Some(b)) => {
                        let (near, far) = if a <= b { (l, r) } else { (r, l) };
                        stack[sp] = far as u32;
                        sp += 1;
                        stack[sp] = near as u32;
                        sp += 1;
                    }
                    (Some(_), None) => {
                        stack[sp] = l as u32;
                        sp += 1;
                    }
                    (None, Some(_)) => {
                        stack[sp] = r as u32;
                        sp += 1;
                    }
                    (None, None) => {}
                }
                debug_assert!(sp < 64, "BVH traversal stack overflow");
            }
        }
        best
    }

    /// Any-hit within t_limit (shadow rays). `hit_prim(prim_id, t_limit)`
    /// returns true if the primitive blocks within the limit.
    pub fn any_hit<F>(&self, ray: &Ray, t_limit: f64, mut hit_prim: F) -> bool
    where
        F: FnMut(u32, f64) -> bool,
    {
        if self.nodes.is_empty() {
            return false;
        }
        let origin = Vec3::new(ray.origin.x, ray.origin.y, ray.origin.z);
        let inv_dir = Vec3::new(
            1.0 / ray.direction.x,
            1.0 / ray.direction.y,
            1.0 / ray.direction.z,
        );
        let mut stack: [u32; 64] = [0; 64];
        let mut sp = 0usize;
        stack[sp] = 0;
        sp += 1;
        while sp > 0 {
            sp -= 1;
            let node = &self.nodes[stack[sp] as usize];
            if node.bounds.hit(&origin, &inv_dir, t_limit).is_none() {
                continue;
            }
            if node.count > 0 {
                for i in node.first..node.first + node.count {
                    if hit_prim(self.prim_order[i as usize], t_limit) {
                        return true;
                    }
                }
            } else {
                stack[sp] = node.first;
                sp += 1;
                stack[sp] = node.first + 1;
                sp += 1;
            }
        }
        false
    }
}

struct Builder<'a> {
    bounds: &'a [Aabb],
    centroids: &'a [Vec3],
    nodes: Vec<Node>,
    prim_order: &'a mut Vec<u32>,
}

impl Builder<'_> {
    fn subdivide(&mut self, node_idx: usize, start: usize, count: usize) {
        // Node bounds over its primitives.
        let mut bounds = Aabb::empty();
        let mut cbounds = Aabb::empty();
        for &p in &self.prim_order[start..start + count] {
            bounds.grow(&self.bounds[p as usize]);
            cbounds.grow_point(&self.centroids[p as usize]);
        }
        self.nodes[node_idx].bounds = bounds;

        if count <= MAX_LEAF {
            self.nodes[node_idx].first = start as u32;
            self.nodes[node_idx].count = count as u32;
            return;
        }

        // Binned SAH along the widest centroid axis.
        let extent = cbounds.max - cbounds.min;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };
        let axis_min = get(&cbounds.min, axis);
        let axis_extent = get(&extent, axis);

        let mut split_at = start + count / 2; // fallback: median
        if axis_extent > 1e-12 {
            let mut bin_bounds = [Aabb::empty(); BINS];
            let mut bin_counts = [0usize; BINS];
            let scale = BINS as f64 / axis_extent;
            for &p in &self.prim_order[start..start + count] {
                let b = (((get(&self.centroids[p as usize], axis) - axis_min) * scale) as usize)
                    .min(BINS - 1);
                bin_counts[b] += 1;
                bin_bounds[b].grow(&self.bounds[p as usize]);
            }
            // Sweep to find the cheapest split.
            let mut right_acc = [Aabb::empty(); BINS];
            let mut acc = Aabb::empty();
            for i in (1..BINS).rev() {
                acc.grow(&bin_bounds[i]);
                right_acc[i] = acc;
            }
            let mut best_cost = f64::INFINITY;
            let mut best_bin = 0;
            let mut left = Aabb::empty();
            let mut left_count = 0usize;
            for i in 0..BINS - 1 {
                left.grow(&bin_bounds[i]);
                left_count += bin_counts[i];
                let right_count = count - left_count;
                if left_count == 0 || right_count == 0 {
                    continue;
                }
                let cost = left.surface_area() * left_count as f64
                    + right_acc[i + 1].surface_area() * right_count as f64;
                if cost < best_cost {
                    best_cost = cost;
                    best_bin = i;
                }
            }
            let leaf_cost = bounds.surface_area() * count as f64;
            if best_cost.is_finite() && best_cost < leaf_cost {
                let boundary = axis_min + axis_extent * (best_bin + 1) as f64 / BINS as f64;
                let mid = partition(
                    &mut self.prim_order[start..start + count],
                    |p| get(&self.centroids[p as usize], axis) < boundary,
                );
                if mid > 0 && mid < count {
                    split_at = start + mid;
                }
            } else if count <= MAX_LEAF * 4 {
                // SAH says a leaf is cheaper and it's small enough: stop.
                self.nodes[node_idx].first = start as u32;
                self.nodes[node_idx].count = count as u32;
                return;
            }
        }

        let left_idx = self.nodes.len();
        self.nodes.push(Node { bounds: Aabb::empty(), first: 0, count: 0 });
        self.nodes.push(Node { bounds: Aabb::empty(), first: 0, count: 0 });
        self.nodes[node_idx].first = left_idx as u32;
        self.nodes[node_idx].count = 0;
        self.subdivide(left_idx, start, split_at - start);
        self.subdivide(left_idx + 1, split_at, start + count - split_at);
    }
}

#[inline]
fn get(v: &Vec3, axis: usize) -> f64 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

/// In-place partition; returns the index of the first element for which the
/// predicate is false.
fn partition<F: FnMut(u32) -> bool>(slice: &mut [u32], mut pred: F) -> usize {
    let mut i = 0;
    for j in 0..slice.len() {
        if pred(slice[j]) {
            slice.swap(i, j);
            i += 1;
        }
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri_bounds(tris: &[[Point3; 3]]) -> Vec<Aabb> {
        tris.iter()
            .map(|t| Aabb::from_points(t.iter().copied()))
            .collect()
    }

    fn ray_tri(ray: &Ray, tri: &[Point3; 3]) -> Option<f64> {
        let e1 = tri[1] - tri[0];
        let e2 = tri[2] - tri[0];
        let p = ray.direction.cross(&e2);
        let det = e1.dot(&p);
        if det.abs() < 1e-12 {
            return None;
        }
        let inv = 1.0 / det;
        let tv = ray.origin - tri[0];
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
        (t > 1e-9).then_some(t)
    }

    #[test]
    fn bvh_matches_brute_force() {
        // Deterministic pseudo-random triangle soup.
        let mut state = 0x12345678u64;
        let mut rnd = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 8.0 - 4.0
        };
        let tris: Vec<[Point3; 3]> = (0..500)
            .map(|_| {
                let base = Point3::new(rnd(), rnd(), rnd());
                [
                    base,
                    Point3::new(base.x + rnd() * 0.2, base.y + rnd() * 0.2, base.z + rnd() * 0.2),
                    Point3::new(base.x + rnd() * 0.2, base.y + rnd() * 0.2, base.z + rnd() * 0.2),
                ]
            })
            .collect();
        let bvh = Bvh::build(&tri_bounds(&tris));

        for _ in 0..2000 {
            let ray = Ray::new(
                Point3::new(rnd(), rnd(), rnd()),
                Vec3::new(rnd(), rnd(), rnd()),
            );
            if ray.direction.length() < 1e-6 {
                continue;
            }
            // Brute force.
            let mut brute: Option<(usize, f64)> = None;
            for (i, tri) in tris.iter().enumerate() {
                if let Some(t) = ray_tri(&ray, tri) {
                    if brute.map_or(true, |(_, bt)| t < bt) {
                        brute = Some((i, t));
                    }
                }
            }
            // BVH.
            let bvh_hit = bvh.traverse(&ray, f64::INFINITY, |prim, t_max| {
                ray_tri(&ray, &tris[prim as usize]).filter(|t| *t < t_max)
            });
            match (brute, bvh_hit) {
                (None, None) => {}
                (Some((bi, bt)), Some((vi, vt))) => {
                    assert!((bt - vt).abs() < 1e-9, "t mismatch: {bt} vs {vt}");
                    assert_eq!(bi as u32, vi, "prim mismatch");
                }
                other => panic!("hit disagreement: {other:?}"),
            }
        }
    }

    #[test]
    fn any_hit_agrees() {
        let tris = vec![
            [
                Point3::new(-1.0, -1.0, 5.0),
                Point3::new(1.0, -1.0, 5.0),
                Point3::new(0.0, 1.0, 5.0),
            ],
        ];
        let bvh = Bvh::build(&tri_bounds(&tris));
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(bvh.any_hit(&ray, 10.0, |p, lim| {
            ray_tri(&ray, &tris[p as usize]).is_some_and(|t| t < lim)
        }));
        assert!(!bvh.any_hit(&ray, 4.0, |p, lim| {
            ray_tri(&ray, &tris[p as usize]).is_some_and(|t| t < lim)
        }));
    }
}
