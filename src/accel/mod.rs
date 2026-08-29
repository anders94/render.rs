//! Acceleration structures (roadmap Phase 2): a binned-SAH BVH used at two
//! levels — per-mesh over triangles (BLAS) and scene-wide over instances
//! (TLAS).

mod bvh;

pub use bvh::{Aabb, Bvh};
