//! Native Metal compute backend (macOS only, always available — no build
//! flags). One MSL megakernel, one thread per pixel: the per-ray object
//! loop, shadow rays, and the 5-bounce reflection loop all run in
//! registers, avoiding the per-op intermediate materialization that makes
//! array-programming backends memory-bandwidth-bound. The kernel is compiled at
//! runtime from the embedded kernel.metal via the OS compiler service.

pub mod gpu_scene;
pub mod pattern_codegen;
mod renderer;
pub mod scene_buffers;

pub use renderer::{intersect_probe, render, render_pt, render_pt_checkpointed, render_pt_film};
