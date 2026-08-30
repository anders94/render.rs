//! IES photometric profiles (LM-63), roadmap P12: measured luminaire
//! distributions applied to point lights. The candela grid over
//! (vertical, horizontal) angles becomes a direction-dependent intensity
//! multiplier, normalized by the sphere-averaged candela so a profiled
//! light emits the same total power as the bare light — the profile
//! reshapes, never brightens.
//!
//! Convention: vertical angle 0° points along the light's local "down"
//! (-y before the light's transform); the light's CTM at declaration
//! orients the profile.

use anyhow::{bail, Result};

#[derive(Debug, Clone)]
pub struct IesProfile {
    /// Vertical angles in degrees, ascending (0 = nadir).
    pub v_angles: Vec<f64>,
    /// Horizontal angles in degrees, ascending.
    pub h_angles: Vec<f64>,
    /// candela[h][v].
    pub candela: Vec<Vec<f64>>,
    /// Sphere-averaged candela (sin-weighted) for normalization.
    pub average: f64,
}

impl IesProfile {
    pub fn parse(text: &str) -> Result<Self> {
        // Skip header lines until TILT=...; numbers follow, whitespace-
        // separated across lines.
        let tilt_pos = text
            .find("TILT=")
            .ok_or_else(|| anyhow::anyhow!("IES: no TILT= line"))?;
        let after = &text[tilt_pos..];
        let tilt_line_end = after.find('\n').unwrap_or(after.len());
        let tilt = after[..tilt_line_end].trim();
        if tilt != "TILT=NONE" {
            bail!("IES: only TILT=NONE is supported (got {tilt})");
        }
        let numbers: Vec<f64> = after[tilt_line_end..]
            .split_whitespace()
            .filter_map(|t| t.parse::<f64>().ok())
            .collect();
        if numbers.len() < 13 {
            bail!("IES: truncated numeric block");
        }
        let n_v = numbers[3] as usize;
        let n_h = numbers[4] as usize;
        let multiplier = numbers[2];
        if n_v == 0 || n_h == 0 || n_v > 4096 || n_h > 4096 {
            bail!("IES: implausible angle counts {n_v}x{n_h}");
        }
        // numbers[0..10] = counts/geometry, [10..13] = ballast block.
        let mut it = numbers[13..].iter().copied();
        let v_angles: Vec<f64> = (&mut it).take(n_v).collect();
        let h_angles: Vec<f64> = (&mut it).take(n_h).collect();
        if v_angles.len() < n_v || h_angles.len() < n_h {
            bail!("IES: missing angle values");
        }
        let mut candela = Vec::with_capacity(n_h);
        for _ in 0..n_h {
            let row: Vec<f64> = (&mut it).take(n_v).map(|c| c * multiplier).collect();
            if row.len() < n_v {
                bail!("IES: missing candela values");
            }
            candela.push(row);
        }

        // Sphere-averaged candela: integrate over vertical bands
        // (candela averaged over horizontal) with sin weighting.
        let mut num = 0.0;
        let mut den = 0.0;
        for vi in 0..n_v {
            let theta = v_angles[vi].to_radians();
            let w = theta.sin().max(1e-4);
            let mean_h: f64 =
                candela.iter().map(|row| row[vi]).sum::<f64>() / n_h as f64;
            num += mean_h * w;
            den += w;
        }
        let average = (num / den.max(1e-9)).max(1e-9);
        Ok(Self { v_angles, h_angles, candela, average })
    }

    pub fn load(path: &std::path::Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::parse(&text)
    }

    fn interp_axis(angles: &[f64], a: f64) -> (usize, usize, f64) {
        if a <= angles[0] {
            return (0, 0, 0.0);
        }
        if a >= *angles.last().unwrap() {
            let last = angles.len() - 1;
            return (last, last, 0.0);
        }
        for i in 0..angles.len() - 1 {
            if a <= angles[i + 1] {
                let span = (angles[i + 1] - angles[i]).max(1e-9);
                return (i, i + 1, (a - angles[i]) / span);
            }
        }
        let last = angles.len() - 1;
        (last, last, 0.0)
    }

    /// Raw candela toward a local direction (0° vertical = local -y).
    pub fn candela_toward(&self, dir_local: &crate::math::Vec3) -> f64 {
        let d = dir_local.normalize();
        // theta from nadir; phi around the vertical axis.
        let theta = (-d.y).clamp(-1.0, 1.0).acos().to_degrees();
        let mut phi = d.z.atan2(d.x).to_degrees();
        if phi < 0.0 {
            phi += 360.0;
        }
        // Fold phi into the profile's symmetry range.
        let h_max = *self.h_angles.last().unwrap();
        let phi = if h_max <= 0.0 {
            0.0
        } else if h_max <= 90.0 {
            let p = phi % 180.0;
            if p > 90.0 { 180.0 - p } else { p }
        } else if h_max <= 180.0 {
            let p = phi % 360.0;
            if p > 180.0 { 360.0 - p } else { p }
        } else {
            phi
        };
        let (v0, v1, tv) = Self::interp_axis(&self.v_angles, theta);
        let (h0, h1, th) = Self::interp_axis(&self.h_angles, phi);
        let c00 = self.candela[h0][v0];
        let c01 = self.candela[h0][v1];
        let c10 = self.candela[h1][v0];
        let c11 = self.candela[h1][v1];
        (c00 * (1.0 - tv) + c01 * tv) * (1.0 - th) + (c10 * (1.0 - tv) + c11 * tv) * th
    }

    /// Normalized intensity multiplier toward a local direction.
    pub fn factor(&self, dir_local: &crate::math::Vec3) -> f64 {
        self.candela_toward(dir_local) / self.average
    }

    /// Export as a dense (v, h) table for the GPU: fixed 32x16 grid over
    /// [0,180] x [0,360] degrees, already normalized.
    pub fn export_grid(&self) -> Vec<f32> {
        const NV: usize = 32;
        const NH: usize = 16;
        let mut out = Vec::with_capacity(NV * NH);
        for hi in 0..NH {
            for vi in 0..NV {
                let theta = 180.0 * vi as f64 / (NV - 1) as f64;
                let phi = 360.0 * hi as f64 / NH as f64;
                let tr = theta.to_radians();
                let pr = phi.to_radians();
                let d = crate::math::Vec3::new(
                    tr.sin() * pr.cos(),
                    -tr.cos(),
                    tr.sin() * pr.sin(),
                );
                out.push(self.factor(&d) as f32);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;

    /// A narrow downlight: bright at nadir, dark past 40°.
    const DOWNLIGHT: &str = "IESNA:LM-63-2002\n\
        [TEST] hand-written narrow downlight\n\
        TILT=NONE\n\
        1 1000 1 5 1 1 2 0.1 0.1 0.1\n\
        1.0 1.0 100\n\
        0 20 40 60 90\n\
        0\n\
        1000 800 200 20 0\n";

    #[test]
    fn parse_and_evaluate() {
        let p = IesProfile::parse(DOWNLIGHT).unwrap();
        assert_eq!(p.v_angles.len(), 5);
        assert_eq!(p.h_angles.len(), 1);
        // Straight down: candela 1000; sideways (90°): 0.
        let down = p.candela_toward(&Vec3::new(0.0, -1.0, 0.0));
        let side = p.candela_toward(&Vec3::new(1.0, 0.0, 0.0));
        let up = p.candela_toward(&Vec3::new(0.0, 1.0, 0.0));
        assert!((down - 1000.0).abs() < 1e-6, "down {down}");
        assert!(side < 1.0, "side {side}");
        assert!(up < 1.0, "up {up}");
        // 30° = between 20° (800) and 40° (200) -> 500.
        let t30 = 30.0f64.to_radians();
        let d30 = Vec3::new(t30.sin(), -t30.cos(), 0.0);
        let c30 = p.candela_toward(&d30);
        assert!((c30 - 500.0).abs() < 1.0, "30deg {c30}");
        // Normalization: factor at nadir > 1 (narrow beam concentrates).
        assert!(p.factor(&Vec3::new(0.0, -1.0, 0.0)) > 2.0);
        // Grid export is finite and non-negative.
        let g = p.export_grid();
        assert_eq!(g.len(), 32 * 16);
        assert!(g.iter().all(|v| v.is_finite() && *v >= 0.0));
    }
}
