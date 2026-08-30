//! Many-light sampling (roadmap Phase 9): a BVH over finite lights,
//! traversed stochastically by importance (power over squared distance,
//! clamped inside a cluster's own radius), replaces uniform light
//! selection — with 1000+ lights, uniform picking wastes almost every
//! shadow ray on distant lights. Infinite lights (distant sun, dome) sit
//! in a separate power-proportional group.
//!
//! The same structure answers `pmf(p, light)` for MIS on emitter hits, by
//! walking the chosen light's leaf-to-root path and multiplying branch
//! probabilities. Node layout is GPU-shaped: the Metal kernel traverses
//! the identical buffer.

use crate::math::{Point3, Vec3};
use crate::scene::{Light, LightType};

/// One BVH node, #[repr(C)] for byte-identical GPU upload.
/// Interior: `a`/`b` are child node indices. Leaf: `a` is the light index
/// (scene light list), `b` == u32::MAX marks the leaf.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LightBvhNode {
    pub min: [f32; 3],
    pub power: f32,
    pub max: [f32; 3],
    pub parent: u32,
    pub a: u32,
    pub b: u32,
    pub pad: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<LightBvhNode>() == 48);

pub struct LightSampler {
    pub nodes: Vec<LightBvhNode>,
    /// Leaf node index per scene light (u32::MAX for infinite lights).
    pub light_leaf: Vec<u32>,
    /// Scene light indices of the infinite group (distant + dome).
    pub infinite: Vec<u32>,
    /// Power per infinite light and the group total.
    pub infinite_power: Vec<f64>,
    pub infinite_total: f64,
    /// Probability of choosing the infinite group.
    pub p_infinite: f64,
}

fn lum(c: &Vec3) -> f64 {
    (0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z).max(1e-9)
}

/// (center, half-diagonal radius, power) of a finite light.
fn finite_light_info(light: &Light) -> Option<(Point3, f64, f64)> {
    let power = lum(&light.radiance());
    match &light.light_type {
        LightType::Point { position } => Some((*position, 0.0, power * 4.0 * std::f64::consts::PI)),
        LightType::Rect { corner, edge1, edge2, area, .. } => {
            let center = *corner + *edge1 * 0.5 + *edge2 * 0.5;
            let radius = (edge1.length() + edge2.length()) * 0.5;
            Some((center, radius, power * area * std::f64::consts::PI))
        }
        LightType::SphereArea { center, radius } => Some((
            *center,
            *radius,
            power * 4.0 * std::f64::consts::PI * radius * radius * std::f64::consts::PI,
        )),
        LightType::DiskArea { center, e1, area, .. } => {
            Some((*center, e1.length(), power * area * std::f64::consts::PI))
        }
        LightType::Distant { .. } | LightType::Dome => None,
    }
}

/// Importance of a cluster as seen from `p`: power / max(d², r²) — the
/// clamp keeps points inside (or near) a cluster from exploding.
fn node_importance(node: &LightBvhNode, p: &Point3) -> f64 {
    let cx = (node.min[0] as f64 + node.max[0] as f64) * 0.5;
    let cy = (node.min[1] as f64 + node.max[1] as f64) * 0.5;
    let cz = (node.min[2] as f64 + node.max[2] as f64) * 0.5;
    let dx = p.x - cx;
    let dy = p.y - cy;
    let dz = p.z - cz;
    let d2 = dx * dx + dy * dy + dz * dz;
    let rx = (node.max[0] - node.min[0]) as f64 * 0.5;
    let ry = (node.max[1] - node.min[1]) as f64 * 0.5;
    let rz = (node.max[2] - node.min[2]) as f64 * 0.5;
    let r2 = rx * rx + ry * ry + rz * rz;
    node.power as f64 / d2.max(r2).max(1e-6)
}

struct BuildEntry {
    light: u32,
    center: Point3,
    radius: f64,
    power: f64,
}

impl LightSampler {
    pub fn build(lights: &[Light]) -> Self {
        let mut finite = Vec::new();
        let mut infinite = Vec::new();
        let mut infinite_power = Vec::new();
        for (i, light) in lights.iter().enumerate() {
            match finite_light_info(light) {
                Some((center, radius, power)) => finite.push(BuildEntry {
                    light: i as u32,
                    center,
                    radius,
                    power,
                }),
                None => {
                    infinite.push(i as u32);
                    // Dome/distant power heuristic: luminance scaled as an
                    // ambient bath (env maps fold in their mean).
                    let l = &lights[i];
                    let base = match &l.env {
                        Some(env) => env.mean_luminance() * l.intensity,
                        None => lum(&l.radiance()),
                    };
                    infinite_power.push(base * 4.0 * std::f64::consts::PI);
                }
            }
        }
        let infinite_total: f64 = infinite_power.iter().sum();

        let mut nodes = Vec::new();
        let mut light_leaf = vec![u32::MAX; lights.len()];
        if !finite.is_empty() {
            let n = finite.len();
            let mut order: Vec<usize> = (0..n).collect();
            build_recursive(&finite, &mut order, 0, n, u32::MAX, &mut nodes, &mut light_leaf);
        }

        let finite_total: f64 = finite.iter().map(|e| e.power).sum();
        let p_infinite = if infinite.is_empty() {
            0.0
        } else if finite.is_empty() {
            1.0
        } else {
            (infinite_total / (infinite_total + finite_total)).clamp(0.1, 0.9)
        };

        Self { nodes, light_leaf, infinite, infinite_power, infinite_total, p_infinite }
    }

    pub fn is_trivial(&self) -> bool {
        self.nodes.is_empty() && self.infinite.len() <= 1
    }

    /// Pick a light for NEE at shading point `p`. Returns (light index,
    /// pmf). `u` is a uniform sample (consumed adaptively down the tree).
    pub fn sample(&self, p: &Point3, mut u: f64) -> Option<(usize, f64)> {
        let has_finite = !self.nodes.is_empty();
        let has_infinite = !self.infinite.is_empty();
        if !has_finite && !has_infinite {
            return None;
        }
        let mut pmf = 1.0;
        let infinite = if has_finite && has_infinite {
            if u < self.p_infinite {
                u /= self.p_infinite;
                pmf *= self.p_infinite;
                true
            } else {
                u = (u - self.p_infinite) / (1.0 - self.p_infinite);
                pmf *= 1.0 - self.p_infinite;
                false
            }
        } else {
            has_infinite
        };

        if infinite {
            // Power-proportional pick within the infinite group.
            let mut acc = 0.0;
            let target = u * self.infinite_total;
            for (k, w) in self.infinite_power.iter().enumerate() {
                acc += w;
                if target <= acc || k == self.infinite.len() - 1 {
                    return Some((
                        self.infinite[k] as usize,
                        pmf * w / self.infinite_total.max(1e-12),
                    ));
                }
            }
            return None;
        }

        // Stochastic BVH descent.
        let mut node = 0usize;
        loop {
            let nd = &self.nodes[node];
            if nd.b == u32::MAX {
                return Some((nd.a as usize, pmf));
            }
            let ia = node_importance(&self.nodes[nd.a as usize], p);
            let ib = node_importance(&self.nodes[nd.b as usize], p);
            let total = ia + ib;
            let pa = if total > 1e-30 { ia / total } else { 0.5 };
            if u < pa {
                u = (u / pa).min(1.0 - 1e-12);
                pmf *= pa;
                node = nd.a as usize;
            } else {
                u = ((u - pa) / (1.0 - pa)).min(1.0 - 1e-12);
                pmf *= 1.0 - pa;
                node = nd.b as usize;
            }
        }
    }

    /// Probability that `sample(p, ·)` returns `light` — the MIS density
    /// for emitter hits.
    pub fn pmf(&self, p: &Point3, light: usize) -> f64 {
        let has_finite = !self.nodes.is_empty();
        let has_infinite = !self.infinite.is_empty();
        if let Some(k) = self.infinite.iter().position(|&i| i as usize == light) {
            let group = if has_finite { self.p_infinite } else { 1.0 };
            return group * self.infinite_power[k] / self.infinite_total.max(1e-12);
        }
        let leaf = self.light_leaf[light];
        if leaf == u32::MAX {
            return 0.0;
        }
        let mut pmf = if has_infinite { 1.0 - self.p_infinite } else { 1.0 };
        // Walk leaf -> root, multiplying the branch probability at each
        // parent (recomputing sibling importances from p).
        let mut node = leaf;
        while self.nodes[node as usize].parent != u32::MAX {
            let parent = self.nodes[node as usize].parent;
            let pn = &self.nodes[parent as usize];
            let ia = node_importance(&self.nodes[pn.a as usize], p);
            let ib = node_importance(&self.nodes[pn.b as usize], p);
            let total = ia + ib;
            let mine = if node == pn.a { ia } else { ib };
            pmf *= if total > 1e-30 { mine / total } else { 0.5 };
            node = parent;
        }
        pmf
    }
}

/// Median-split build over light centers; returns the node index.
fn build_recursive(
    entries: &[BuildEntry],
    order: &mut [usize],
    start: usize,
    end: usize,
    parent: u32,
    nodes: &mut Vec<LightBvhNode>,
    light_leaf: &mut [u32],
) -> u32 {
    let index = nodes.len() as u32;
    // Bounds + power over the range (light extents included).
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut power = 0.0f64;
    for &i in &order[start..end] {
        let e = &entries[i];
        let c = [e.center.x, e.center.y, e.center.z];
        for a in 0..3 {
            min[a] = min[a].min((c[a] - e.radius) as f32);
            max[a] = max[a].max((c[a] + e.radius) as f32);
        }
        power += e.power;
    }
    nodes.push(LightBvhNode {
        min,
        power: power as f32,
        max,
        parent,
        a: 0,
        b: u32::MAX,
        pad: [0; 2],
    });

    if end - start == 1 {
        let light = entries[order[start]].light;
        nodes[index as usize].a = light;
        light_leaf[light as usize] = index;
        return index;
    }

    // Split along the widest axis at the median.
    let mut axis = 0;
    let mut widest = -1.0f32;
    for a in 0..3 {
        let w = max[a] - min[a];
        if w > widest {
            widest = w;
            axis = a;
        }
    }
    let mid = (start + end) / 2;
    order[start..end].sort_by(|&x, &y| {
        let cx = match axis {
            0 => entries[x].center.x,
            1 => entries[x].center.y,
            _ => entries[x].center.z,
        };
        let cy = match axis {
            0 => entries[y].center.x,
            1 => entries[y].center.y,
            _ => entries[y].center.z,
        };
        cx.partial_cmp(&cy).unwrap_or(std::cmp::Ordering::Equal)
    });
    let a = build_recursive(entries, order, start, mid, index, nodes, light_leaf);
    let b = build_recursive(entries, order, mid, end, index, nodes, light_leaf);
    // Interior: b != u32::MAX distinguishes it from a leaf.
    nodes[index as usize].a = a;
    nodes[index as usize].b = b;
    index
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point_light(x: f64, y: f64, z: f64, intensity: f64) -> Light {
        Light::point(Point3::new(x, y, z), intensity, Vec3::one())
    }

    /// The pmf returned by sample() must equal pmf() recomputed, and sum
    /// to 1 over all lights.
    #[test]
    fn pmf_consistency_and_normalization() {
        let mut lights = Vec::new();
        for i in 0..40 {
            let a = i as f64 * 0.7;
            lights.push(point_light(
                a.cos() * (5.0 + (i % 7) as f64),
                (i % 5) as f64,
                a.sin() * (5.0 + (i % 3) as f64 * 3.0),
                0.5 + (i % 4) as f64,
            ));
        }
        lights.push(Light::dome(0.5, Vec3::one(), None));
        let sampler = LightSampler::build(&lights);
        let p = Point3::new(1.0, 0.5, -2.0);

        let mut total = 0.0;
        for i in 0..lights.len() {
            total += sampler.pmf(&p, i);
        }
        assert!((total - 1.0).abs() < 1e-9, "pmf sums to {total}");

        for k in 0..200 {
            let u = (k as f64 + 0.5) / 200.0;
            let (idx, pmf) = sampler.sample(&p, u).expect("sample");
            let re = sampler.pmf(&p, idx);
            assert!(
                (pmf - re).abs() < 1e-9 * pmf.max(re).max(1e-12),
                "light {idx}: sample pmf {pmf} vs pmf() {re}"
            );
        }
    }

    /// Nearby bright lights must be picked far more often than distant
    /// dim ones.
    #[test]
    fn importance_prefers_near_lights() {
        let mut lights = vec![point_light(0.0, 1.0, 0.0, 10.0)];
        for i in 0..63 {
            lights.push(point_light(500.0 + i as f64, 1.0, 500.0, 10.0));
        }
        let sampler = LightSampler::build(&lights);
        let p = Point3::new(0.0, 0.0, 0.5);
        let near_pmf = sampler.pmf(&p, 0);
        // Median-split without SAOH still concentrates: >20x uniform
        // (uniform would be 1/64 ≈ 0.016).
        assert!(near_pmf > 0.25, "near light pmf = {near_pmf}");
    }
}
