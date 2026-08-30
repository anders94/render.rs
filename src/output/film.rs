//! The production film (roadmap Phase 11): beauty plus auxiliary AOVs.
//! Every layer is a full-resolution Vec<Vec<Vec3>> plane; scalar AOVs
//! (depth, id) live in x.

use crate::math::Vec3;
use crate::output::Image;
use std::collections::BTreeMap;

/// Per-sample auxiliary shading data, filled at the first "real" hit
/// (presence skips and volume hulls are looked through; volume scatter
/// points count as hits for depth but have no albedo/normal).
#[derive(Debug, Clone, Copy)]
pub struct AuxSample {
    pub albedo: Vec3,
    pub normal: Vec3,
    /// Euclidean distance from the camera (0 = background).
    pub depth: f64,
    /// Object id (from Attribute "identifier" or auto-assigned), 0 = none.
    pub id: u32,
    /// Beauty split: radiance whose FIRST bounce was diffuse-like vs
    /// specular-like (glass/mirror). diffuse + specular = beauty.
    pub diffuse: Vec3,
    pub specular: Vec3,
}

impl Default for AuxSample {
    fn default() -> Self {
        Self {
            albedo: Vec3::zero(),
            normal: Vec3::zero(),
            depth: 0.0,
            id: 0,
            diffuse: Vec3::zero(),
            specular: Vec3::zero(),
        }
    }
}

/// A rendered frame with all its layers.
pub struct Film {
    pub beauty: Image,
    pub diffuse: Image,
    pub specular: Image,
    pub albedo: Image,
    pub normal: Image,
    /// x = depth (mean over samples that hit; background stays 0).
    pub depth: Image,
    /// x = dominant object id, y = coverage of that id in [0,1].
    pub id: Image,
    /// id -> name manifest (cryptomatte-style, stored in the EXR header).
    pub manifest: BTreeMap<u32, String>,
}

impl Film {
    pub fn width(&self) -> usize {
        self.beauty.first().map(|r| r.len()).unwrap_or(0)
    }

    pub fn height(&self) -> usize {
        self.beauty.len()
    }
}
