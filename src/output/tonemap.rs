//! Display output transforms (roadmap Phase 11). The film is linear; PNG
//! and PPM outputs pass through one of these before quantization:
//! - linear: clamp only (debugging)
//! - srgb: the IEC sRGB electro-optical encoding (the old default was a
//!   plain 2.2 gamma; the exact curve replaces it)
//! - aces: the Narkowicz ACES filmic fit, then sRGB encode — the shoulder
//!   rolls highlights off smoothly instead of clipping.

use crate::math::Vec3;
use crate::output::Image;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tonemap {
    Linear,
    Srgb,
    Aces,
}

impl Tonemap {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "linear" => Some(Tonemap::Linear),
            "srgb" => Some(Tonemap::Srgb),
            "aces" => Some(Tonemap::Aces),
            _ => None,
        }
    }
}

fn srgb_encode(c: f64) -> f64 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn aces_fit(x: f64) -> f64 {
    // Narkowicz 2015 ACES filmic approximation.
    ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0)
}

/// Apply the transform to one linear pixel, producing display-referred
/// values in [0, 1].
pub fn apply(tonemap: Tonemap, c: &Vec3) -> Vec3 {
    match tonemap {
        Tonemap::Linear => Vec3::new(
            c.x.clamp(0.0, 1.0),
            c.y.clamp(0.0, 1.0),
            c.z.clamp(0.0, 1.0),
        ),
        Tonemap::Srgb => Vec3::new(srgb_encode(c.x), srgb_encode(c.y), srgb_encode(c.z)),
        Tonemap::Aces => Vec3::new(
            srgb_encode(aces_fit(c.x)),
            srgb_encode(aces_fit(c.y)),
            srgb_encode(aces_fit(c.z)),
        ),
    }
}

/// Transform a whole (linear) image to display space.
pub fn apply_image(tonemap: Tonemap, image: &Image) -> Image {
    image
        .iter()
        .map(|row| row.iter().map(|p| apply(tonemap, p)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transforms_behave() {
        // sRGB: 0 -> 0, 1 -> 1, monotone.
        assert!(srgb_encode(0.0) < 1e-9);
        assert!((srgb_encode(1.0) - 1.0).abs() < 1e-9);
        // ACES: rolls off — bright values compress below clamp, and the
        // shoulder approaches (but never exceeds) 1.
        assert!(aces_fit(0.18) > 0.18); // filmic lifts mid-grey slightly
        assert!(aces_fit(4.0) > 0.95 && aces_fit(4.0) <= 1.0);
        assert!(aces_fit(1.0) < 1.0);
        // Monotonicity over the working range.
        let mut prev = -1.0;
        for i in 0..100 {
            let v = aces_fit(i as f64 * 0.1);
            assert!(v >= prev);
            prev = v;
        }
    }
}
