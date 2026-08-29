//! Native Metal compute backend (macOS only, always available — no build
//! flags). One MSL megakernel, one thread per pixel: the per-ray object
//! loop, shadow rays, and the 5-bounce reflection loop all run in
//! registers, avoiding the per-op intermediate materialization that makes
//! the MLX backend memory-bandwidth-bound. The kernel is compiled at
//! runtime from the embedded kernel.metal via the OS compiler service.

mod renderer;
pub mod scene_buffers;

pub use renderer::{intersect_probe, render};
