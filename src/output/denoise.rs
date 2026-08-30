//! AOV-guided denoising (roadmap Phase 11): edge-avoiding à-trous wavelet
//! filtering after Dammertz et al. 2010, guided by the film's albedo,
//! normal, and depth layers. Beauty is demodulated by albedo before
//! filtering (illumination varies smoothly where texture detail does
//! not) and remodulated after, so texture stays crisp while Monte Carlo
//! noise smooths out.
//!
//! This is the committed backend-agnostic film post-pass; an OIDN FFI
//! path can slot in behind the same call later (the roadmap's staged
//! plan) — OIDN is not installed on this machine.

use crate::math::Vec3;
use crate::output::film::Film;
use crate::output::Image;
use rayon::prelude::*;

/// À-trous 5-tap B3-spline kernel (separable halves, applied 2D).
const KERNEL: [f64; 5] = [1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0];
const ITERATIONS: usize = 5;

fn lum(c: &Vec3) -> f64 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

/// Denoise the film: the DIFFUSE layer is albedo-demodulated, filtered,
/// and remodulated; the SPECULAR layer is added back raw. Specular and
/// refracted detail (glass, mirrors) does not correlate with the
/// first-hit albedo/normal guides — filtering it smears refraction, so
/// the split earns its keep here. Returns beauty' = filter(diffuse) +
/// specular; the film is unchanged.
pub fn denoise(film: &Film) -> Image {
    let h = film.height();
    let w = film.width();
    if h == 0 || w == 0 {
        return film.beauty.clone();
    }

    // Demodulate: illumination = beauty / max(albedo, eps).
    let demod = |b: &Vec3, a: &Vec3| {
        Vec3::new(
            b.x / a.x.max(5e-2),
            b.y / a.y.max(5e-2),
            b.z / a.z.max(5e-2),
        )
    };
    let mut illum: Image = (0..h)
        .map(|y| {
            (0..w)
                .map(|x| demod(&film.diffuse[y][x], &film.albedo[y][x]))
                .collect()
        })
        .collect();

    // Depth scale for the range weight: median positive depth.
    let mut depths: Vec<f64> = film
        .depth
        .iter()
        .flatten()
        .map(|d| d.x)
        .filter(|d| *d > 0.0)
        .collect();
    depths.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let depth_scale = depths.get(depths.len() / 2).copied().unwrap_or(1.0).max(1e-3);

    let mut sigma_c = 8.0f64; // illumination range weight, tightened per pass

    for iter in 0..ITERATIONS {
        let step = 1usize << iter;
        let src = illum;
        let next: Image = (0..h)
            .into_par_iter()
            .map(|y| {
                (0..w)
                    .map(|x| {
                        let c0 = src[y][x];
                        let n0 = film.normal[y][x];
                        let d0 = film.depth[y][x].x;
                        let id0 = film.id[y][x].x;
                        let mut sum = Vec3::zero();
                        let mut wsum = 0.0;
                        for (kj, wj) in KERNEL.iter().enumerate() {
                            let yy = y as isize + (kj as isize - 2) * step as isize;
                            if yy < 0 || yy >= h as isize {
                                continue;
                            }
                            for (ki, wi) in KERNEL.iter().enumerate() {
                                let xx = x as isize + (ki as isize - 2) * step as isize;
                                if xx < 0 || xx >= w as isize {
                                    continue;
                                }
                                let (yy, xx) = (yy as usize, xx as usize);
                                let mut wgt = wj * wi;
                                // Illumination range.
                                let dc = lum(&(src[yy][xx] - c0));
                                wgt *= (-(dc * dc) / (sigma_c * sigma_c)).exp();
                                // Normal.
                                let ndot = film.normal[yy][xx].dot(&n0).max(0.0);
                                wgt *= ndot.powi(32);
                                // Depth (relative to scene scale).
                                let dd = (film.depth[yy][xx].x - d0) / depth_scale;
                                wgt *= (-(dd * dd) / 0.02).exp();
                                // Object id: never blur across objects.
                                if (film.id[yy][xx].x - id0).abs() > 0.5 {
                                    wgt = 0.0;
                                }
                                sum = sum + src[yy][xx] * wgt;
                                wsum += wgt;
                            }
                        }
                        if wsum > 1e-9 { sum / wsum } else { c0 }
                    })
                    .collect()
            })
            .collect();
        illum = next;
        sigma_c *= 0.5;
    }

    // Remodulate and add the raw specular layer back.
    (0..h)
        .map(|y| {
            (0..w)
                .map(|x| {
                    let a = film.albedo[y][x];
                    let i = illum[y][x];
                    let s = film.specular[y][x];
                    Vec3::new(
                        i.x * a.x.max(5e-2) + s.x,
                        i.y * a.y.max(5e-2) + s.y,
                        i.z * a.z.max(5e-2) + s.z,
                    )
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// Noisy flat region: denoising must slash variance while keeping the
    /// mean, and must not blur across an id/normal edge.
    #[test]
    fn denoise_reduces_noise_preserves_edges() {
        let (w, h) = (64, 64);
        let mut state = 42u64;
        let mut rand = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as f64 / (1u64 << 31) as f64
        };
        // Left half: bright object id 1; right half: dark object id 2.
        let mut beauty = vec![vec![Vec3::zero(); w]; h];
        let mut albedo = vec![vec![Vec3::zero(); w]; h];
        let mut normal = vec![vec![Vec3::zero(); w]; h];
        let mut depth = vec![vec![Vec3::zero(); w]; h];
        let mut id = vec![vec![Vec3::zero(); w]; h];
        for y in 0..h {
            for x in 0..w {
                let left = x < w / 2;
                let base = if left { 0.8 } else { 0.1 };
                let noise = (rand() - 0.5) * 0.4;
                beauty[y][x] = Vec3::new(
                    (base + noise).max(0.0),
                    (base + noise).max(0.0),
                    (base + noise).max(0.0),
                );
                albedo[y][x] = Vec3::new(base, base, base);
                normal[y][x] = Vec3::new(0.0, 0.0, 1.0);
                depth[y][x] = Vec3::new(5.0, 0.0, 0.0);
                id[y][x] = Vec3::new(if left { 1.0 } else { 2.0 }, 1.0, 0.0);
            }
        }
        let film = Film {
            beauty: beauty.clone(),
            diffuse: beauty.clone(),
            specular: vec![vec![Vec3::zero(); w]; h],
            albedo,
            normal,
            depth,
            id,
            manifest: BTreeMap::new(),
        };
        let out = denoise(&film);

        // Variance within the left half must fall dramatically.
        let var = |img: &Image| {
            let vals: Vec<f64> = (8..h - 8)
                .flat_map(|y| (8..w / 2 - 8).map(move |x| (y, x)))
                .map(|(y, x)| img[y][x].x)
                .collect();
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            (
                mean,
                vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / vals.len() as f64,
            )
        };
        let (mean_in, var_in) = var(&film.beauty);
        let (mean_out, var_out) = var(&out);
        assert!(var_out < var_in * 0.05, "variance {var_in:.5} -> {var_out:.5}");
        assert!((mean_out - mean_in).abs() < 0.02, "mean {mean_in:.3} -> {mean_out:.3}");
        // The edge must survive: adjacent pixels across the id boundary
        // keep their contrast.
        let l = out[h / 2][w / 2 - 1].x;
        let r = out[h / 2][w / 2].x;
        assert!(l - r > 0.5, "edge blurred: {l:.3} vs {r:.3}");
    }
}
