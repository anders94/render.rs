//! True displacement at dice time (roadmap Phase 5): tessellated vertices
//! move along their normals by a procedural field, then normals are
//! rebuilt from the displaced faces. Until the Phase-6 pattern graph
//! arrives, the displacement source is a built-in fBm over classic Perlin
//! gradient noise, driven by the `Displace "noise"` extension parameters.

use crate::geometry::subdiv::smooth_normals;
use crate::geometry::Mesh;

#[derive(Debug, Clone)]
pub struct DisplaceParams {
    pub amplitude: f64,
    pub frequency: f64,
    pub octaves: u32,
    /// Per-octave amplitude falloff.
    pub gain: f64,
    /// Per-octave frequency multiplier.
    pub lacunarity: f64,
    /// Offset added to the noise field (shifts features).
    pub offset: [f64; 3],
}

impl Default for DisplaceParams {
    fn default() -> Self {
        Self {
            amplitude: 0.1,
            frequency: 1.0,
            octaves: 4,
            gain: 0.5,
            lacunarity: 2.0,
            offset: [0.0; 3],
        }
    }
}

// Classic Perlin gradient noise with a fixed permutation (deterministic).
fn hash(mut x: i64) -> u32 {
    x = x.wrapping_mul(0x9e3779b97f4a7c15u64 as i64);
    x ^= x >> 29;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9u64 as i64);
    x ^= x >> 32;
    x as u32
}

fn gradient(ix: i64, iy: i64, iz: i64) -> [f64; 3] {
    let h = hash(ix.wrapping_mul(73856093) ^ iy.wrapping_mul(19349663) ^ iz.wrapping_mul(83492791));
    // 12 cube-edge gradients.
    const G: [[f64; 3]; 12] = [
        [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, -1.0],
        [0.0, 1.0, 1.0], [0.0, -1.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0],
    ];
    G[(h % 12) as usize]
}

fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

pub fn perlin(p: [f64; 3]) -> f64 {
    let cell = [p[0].floor(), p[1].floor(), p[2].floor()];
    let f = [p[0] - cell[0], p[1] - cell[1], p[2] - cell[2]];
    let (ix, iy, iz) = (cell[0] as i64, cell[1] as i64, cell[2] as i64);
    let mut corner = [0.0f64; 8];
    for (k, c) in corner.iter_mut().enumerate() {
        let (dx, dy, dz) = ((k & 1) as i64, ((k >> 1) & 1) as i64, ((k >> 2) & 1) as i64);
        let g = gradient(ix + dx, iy + dy, iz + dz);
        let d = [f[0] - dx as f64, f[1] - dy as f64, f[2] - dz as f64];
        *c = g[0] * d[0] + g[1] * d[1] + g[2] * d[2];
    }
    let (u, v, w) = (fade(f[0]), fade(f[1]), fade(f[2]));
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let x00 = lerp(corner[0], corner[1], u);
    let x10 = lerp(corner[2], corner[3], u);
    let x01 = lerp(corner[4], corner[5], u);
    let x11 = lerp(corner[6], corner[7], u);
    let y0 = lerp(x00, x10, v);
    let y1 = lerp(x01, x11, v);
    lerp(y0, y1, w)
}

pub fn fbm(p: [f64; 3], params: &DisplaceParams) -> f64 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut freq = params.frequency;
    for _ in 0..params.octaves.max(1) {
        sum += amp
            * perlin([
                p[0] * freq + params.offset[0],
                p[1] * freq + params.offset[1],
                p[2] * freq + params.offset[2],
            ]);
        amp *= params.gain;
        freq *= params.lacunarity;
    }
    sum
}

/// Displace mesh vertices along their (smooth) normals and rebuild
/// normals from the displaced surface.
pub fn displace_mesh(mesh: Mesh, params: &DisplaceParams) -> Mesh {
    let base_normals = match &mesh.normals {
        Some(n) => n.clone(),
        None => smooth_normals(&mesh.positions, &mesh.indices),
    };
    let positions: Vec<[f32; 3]> = mesh
        .positions
        .iter()
        .zip(base_normals.iter())
        .map(|(p, n)| {
            let pd = [p[0] as f64, p[1] as f64, p[2] as f64];
            let d = fbm(pd, params) * params.amplitude;
            [
                (pd[0] + n[0] as f64 * d) as f32,
                (pd[1] + n[1] as f64 * d) as f32,
                (pd[2] + n[2] as f64 * d) as f32,
            ]
        })
        .collect();
    let indices = mesh.indices.clone();
    let normals = smooth_normals(&positions, &indices);
    Mesh::new(positions, indices, Some(normals), mesh.st.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::subdiv::SubdivCage;

    #[test]
    fn perlin_is_bounded_and_zero_at_lattice() {
        for i in -3..3 {
            for j in -3..3 {
                assert!(perlin([i as f64, j as f64, 0.0]).abs() < 1e-12);
            }
        }
        let mut mx: f64 = 0.0;
        for i in 0..1000 {
            let t = i as f64 * 0.113;
            mx = mx.max(perlin([t, t * 0.7 + 0.3, t * 1.3 + 0.9]).abs());
        }
        assert!(mx <= 1.3, "perlin out of range: {mx}");
        assert!(mx > 0.05, "perlin suspiciously flat: {mx}");
    }

    #[test]
    fn displacement_moves_vertices_and_rebuilds_normals() {
        let cage = SubdivCage {
            positions: vec![
                [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
            ],
            faces: vec![
                vec![0, 1, 2, 3], vec![5, 4, 7, 6], vec![4, 0, 3, 7],
                vec![1, 5, 6, 2], vec![3, 2, 6, 7], vec![4, 5, 1, 0],
            ],
            crease_edges: Vec::new(),
            corners: Vec::new(),
            holes: Vec::new(),
            interpolate_boundary: true,
        };
        let smooth = cage.tessellate(3);
        let before: Vec<[f32; 3]> = smooth.positions.clone();
        let params = DisplaceParams { amplitude: 0.3, frequency: 2.0, ..Default::default() };
        let displaced = displace_mesh(smooth, &params);
        let moved = displaced
            .positions
            .iter()
            .zip(before.iter())
            .filter(|(a, b)| {
                let d = [(a[0] - b[0]), (a[1] - b[1]), (a[2] - b[2])];
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() > 1e-4
            })
            .count();
        assert!(moved > before.len() / 2, "displacement barely moved: {moved}");
        // Normals are unit length.
        for n in displaced.normals.as_ref().unwrap() {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-3);
        }
    }
}
