//! Lat-long environment maps with 2D-CDF importance sampling (dome
//! lights). Loads any format the `image` crate reads (EXR/HDR for real
//! HDRIs). Convention: y-up; u wraps azimuth via atan2(x, -z), v maps
//! polar angle acos(y)/pi.

use crate::math::Vec3;
use anyhow::{Context, Result};
use std::f64::consts::PI;

pub struct EnvMap {
    width: usize,
    height: usize,
    /// Linear radiance, row-major.
    data: Vec<[f32; 3]>,
    /// Row-marginal CDF (height+1 entries) over sin-weighted luminance.
    marginal: Vec<f64>,
    /// Per-row conditional CDFs, (width+1) entries each.
    conditional: Vec<f64>,
    /// Integral of sin-weighted luminance (normalization).
    total: f64,
}

fn lum(c: &[f32; 3]) -> f64 {
    0.2126 * c[0] as f64 + 0.7152 * c[1] as f64 + 0.0722 * c[2] as f64
}

impl EnvMap {
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let img = image::open(path)
            .with_context(|| format!("loading environment map {}", path.display()))?
            .to_rgb32f();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let data: Vec<[f32; 3]> = img.pixels().map(|p| p.0).collect();
        Ok(Self::from_data(w, h, data))
    }

    pub fn from_data(width: usize, height: usize, data: Vec<[f32; 3]>) -> Self {
        assert_eq!(data.len(), width * height);
        let mut marginal = vec![0.0; height + 1];
        let mut conditional = vec![0.0; (width + 1) * height];
        for y in 0..height {
            let sin_theta = (PI * (y as f64 + 0.5) / height as f64).sin();
            let row = &data[y * width..(y + 1) * width];
            let base = y * (width + 1);
            let mut acc = 0.0;
            conditional[base] = 0.0;
            for x in 0..width {
                acc += lum(&row[x]) * sin_theta;
                conditional[base + x + 1] = acc;
            }
            marginal[y + 1] = marginal[y] + acc;
        }
        let total = marginal[height].max(1e-12);
        Self { width, height, data, marginal, conditional, total }
    }

    pub fn is_black(&self) -> bool {
        self.total <= 1e-9
    }

    /// Sin-weighted mean luminance over the sphere (light-power heuristic).
    pub fn mean_luminance(&self) -> f64 {
        // `total` accumulates lum * sin(theta) over all texels; the sin
        // weights sum to ~ width * height * (2/pi).
        let weight_sum = self.width as f64 * self.height as f64 * (2.0 / PI);
        self.total / weight_sum.max(1e-12)
    }

    /// GPU export: (width, height, rgb pixels, marginal CDF, conditional
    /// CDFs, total weight), all f32.
    pub fn export(&self) -> (usize, usize, Vec<f32>, Vec<f32>, Vec<f32>, f64) {
        let pixels: Vec<f32> = self.data.iter().flat_map(|c| c.iter().copied()).collect();
        let marginal: Vec<f32> = self.marginal.iter().map(|v| *v as f32).collect();
        let conditional: Vec<f32> = self.conditional.iter().map(|v| *v as f32).collect();
        (self.width, self.height, pixels, marginal, conditional, self.total)
    }

    fn texel(&self, x: usize, y: usize) -> Vec3 {
        let c = self.data[y.min(self.height - 1) * self.width + x.min(self.width - 1)];
        Vec3::new(c[0] as f64, c[1] as f64, c[2] as f64)
    }

    /// Radiance along a world direction (nearest texel).
    pub fn eval(&self, dir: &Vec3) -> Vec3 {
        let d = dir.normalize();
        let u = (d.x.atan2(-d.z) / (2.0 * PI)).rem_euclid(1.0);
        let v = (d.y.clamp(-1.0, 1.0).acos() / PI).clamp(0.0, 1.0);
        let x = ((u * self.width as f64) as usize).min(self.width - 1);
        let y = ((v * self.height as f64) as usize).min(self.height - 1);
        self.texel(x, y)
    }

    fn dir_from_uv(&self, u: f64, v: f64) -> Vec3 {
        let phi = u * 2.0 * PI;
        let theta = v * PI;
        let sin_t = theta.sin();
        Vec3::new(sin_t * phi.sin(), theta.cos(), -sin_t * phi.cos())
    }

    /// Importance-sample a direction; returns (dir, radiance, pdf_sa).
    pub fn sample(&self, u1: f64, u2: f64) -> (Vec3, Vec3, f64) {
        // Row via marginal CDF.
        let target = u1 * self.total;
        let y = match self
            .marginal
            .binary_search_by(|c| c.partial_cmp(&target).unwrap())
        {
            Ok(i) => i.min(self.height - 1),
            Err(i) => i.saturating_sub(1).min(self.height - 1),
        };
        let row_total = (self.marginal[y + 1] - self.marginal[y]).max(1e-12);

        // Column via conditional CDF.
        let base = y * (self.width + 1);
        let ct = u2 * row_total;
        let cond = &self.conditional[base..base + self.width + 1];
        let x = match cond.binary_search_by(|c| c.partial_cmp(&ct).unwrap()) {
            Ok(i) => i.min(self.width - 1),
            Err(i) => i.saturating_sub(1).min(self.width - 1),
        };

        let u = (x as f64 + 0.5) / self.width as f64;
        let v = (y as f64 + 0.5) / self.height as f64;
        let dir = self.dir_from_uv(u, v);
        let radiance = self.texel(x, y);
        (dir, radiance, self.pdf(&dir))
    }

    /// Solid-angle pdf of sampling `dir`.
    pub fn pdf(&self, dir: &Vec3) -> f64 {
        let d = dir.normalize();
        let u = (d.x.atan2(-d.z) / (2.0 * PI)).rem_euclid(1.0);
        let v = (d.y.clamp(-1.0, 1.0).acos() / PI).clamp(0.0, 1.0);
        let x = ((u * self.width as f64) as usize).min(self.width - 1);
        let y = ((v * self.height as f64) as usize).min(self.height - 1);
        let sin_theta = (PI * (y as f64 + 0.5) / self.height as f64).sin();
        if sin_theta < 1e-9 {
            return 0.0;
        }
        let texel_weight = lum(&self.data[y * self.width + x]) * sin_theta;
        // p(u,v) = texel_weight * (w*h) / total; ω-Jacobian = 2π² sinθ.
        texel_weight * self.width as f64 * self.height as f64
            / (self.total * 2.0 * PI * PI * sin_theta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_pdf_consistency() {
        // A map with a bright patch: sampled directions must land there,
        // and pdf must integrate to ~1 over the sphere.
        let w = 32;
        let h = 16;
        let mut data = vec![[0.05f32; 3]; w * h];
        for y in 4..8 {
            for x in 10..16 {
                data[y * w + x] = [5.0, 5.0, 5.0];
            }
        }
        let env = EnvMap::from_data(w, h, data);

        // pdf integral over sphere via uniform grid.
        let mut integral = 0.0;
        let nt = 128;
        let np = 256;
        for it in 0..nt {
            let theta = (it as f64 + 0.5) / nt as f64 * PI;
            for ip in 0..np {
                let phi = (ip as f64 + 0.5) / np as f64 * 2.0 * PI;
                let d = Vec3::new(
                    theta.sin() * phi.sin(),
                    theta.cos(),
                    -theta.sin() * phi.cos(),
                );
                integral += env.pdf(&d) * theta.sin() * (PI / nt as f64) * (2.0 * PI / np as f64);
            }
        }
        assert!((integral - 1.0).abs() < 0.03, "pdf integral {integral:.4}");

        // Direction round-trip: sample → eval at that dir is bright.
        let mut rng_state = 0x1234u64;
        let mut next = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng_state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut bright = 0;
        for _ in 0..200 {
            let (dir, rad, pdf) = env.sample(next(), next());
            assert!(pdf > 0.0);
            assert!((env.eval(&dir) - rad).length() < 1e-9);
            if rad.x > 1.0 {
                bright += 1;
            }
        }
        assert!(bright > 150, "importance sampling missed the bright patch: {bright}/200");
    }
}
