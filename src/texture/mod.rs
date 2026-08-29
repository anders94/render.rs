//! Textures & patterns (roadmap Phase 6): the tiled-mip .tex format,
//! the process-global sharded-LRU tile cache, and (soon) the pattern
//! node graph.

pub mod cache;
pub mod pattern;
pub mod tex;

pub use cache::{global as global_cache, TexId, TextureCache, Wrap};
pub use tex::{LinearImage, TexHeader, TILE_SIZE};
