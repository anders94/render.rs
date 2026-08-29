//! Sharded-LRU texture tile cache (roadmap Phase 6). One process-global
//! cache serves every texture lookup; tiles are read on demand from .tex
//! files and evicted least-recently-used per shard under a hard byte
//! budget (RENDER_TEX_CACHE_MB overrides the 256 MB default). Non-.tex
//! images are auto-converted once into a temp-dir .tex keyed by path +
//! mtime — the renderer's txmake-on-demand.

use super::tex::{self, TexHeader, TILE_SIZE};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

pub type TexId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wrap {
    Periodic,
    Clamp,
    Black,
}

impl Wrap {
    pub fn from_name(name: &str) -> Self {
        match name {
            "clamp" => Wrap::Clamp,
            "black" => Wrap::Black,
            _ => Wrap::Periodic,
        }
    }
}

struct TexFile {
    file: File,
    header: TexHeader,
}

struct Entry {
    data: Arc<Vec<f32>>,
    last_used: u64,
}

#[derive(Default)]
struct Shard {
    map: HashMap<u64, Entry>,
    bytes: usize,
    tick: u64,
}

#[derive(Default)]
pub struct CacheStats {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub evictions: AtomicU64,
    pub bytes_read: AtomicU64,
}

const SHARDS: usize = 16;

pub struct TextureCache {
    files: RwLock<Vec<TexFile>>,
    by_path: RwLock<HashMap<PathBuf, TexId>>,
    shards: Vec<Mutex<Shard>>,
    shard_budget: usize,
    pub stats: CacheStats,
}

static GLOBAL: OnceLock<TextureCache> = OnceLock::new();

/// The process-global cache (256 MB budget; RENDER_TEX_CACHE_MB overrides).
pub fn global() -> &'static TextureCache {
    GLOBAL.get_or_init(|| {
        let mb = std::env::var("RENDER_TEX_CACHE_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(256);
        TextureCache::new(mb * 1024 * 1024)
    })
}

impl TextureCache {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            files: RwLock::new(Vec::new()),
            by_path: RwLock::new(HashMap::new()),
            shards: (0..SHARDS).map(|_| Mutex::new(Shard::default())).collect(),
            shard_budget: (budget_bytes / SHARDS).max(TILE_SIZE * TILE_SIZE * 3 * 4),
            stats: CacheStats::default(),
        }
    }

    /// Open (or reuse) a texture. Non-.tex inputs are converted to a
    /// cached temp .tex on first use.
    pub fn open(&self, path: &Path) -> Result<TexId, String> {
        let canonical = path
            .canonicalize()
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if let Some(id) = self.by_path.read().unwrap().get(&canonical) {
            return Ok(*id);
        }

        let mut file = File::open(&canonical).map_err(|e| e.to_string())?;
        let header = match tex::read_header(&mut file) {
            Ok(h) => h,
            Err(_) => {
                // Auto-txmake into the temp dir, keyed by path + mtime.
                let mtime = file
                    .metadata()
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                let mut h = DefaultHasher::new();
                (&canonical, mtime).hash(&mut h);
                let tmp = std::env::temp_dir()
                    .join(format!("render_rs_tex_{:016x}.tex", h.finish()));
                if tex::read_header(&mut File::open(&tmp).map_err(|e| e.to_string())?)
                    .is_err()
                {
                    tex::txmake(&canonical, &tmp)?;
                }
                file = File::open(&tmp).map_err(|e| e.to_string())?;
                tex::read_header(&mut file)?
            }
        };

        let mut files = self.files.write().unwrap();
        let id = files.len() as TexId;
        files.push(TexFile { file, header });
        self.by_path.write().unwrap().insert(canonical, id);
        Ok(id)
    }

    pub fn header(&self, id: TexId) -> TexHeader {
        self.files.read().unwrap()[id as usize].header.clone()
    }

    fn tile(&self, id: TexId, mip: usize, tx: u32, ty: u32) -> Arc<Vec<f32>> {
        let key = ((id as u64) << 44) | ((mip as u64) << 36) | ((ty as u64) << 18) | tx as u64;
        let shard = &self.shards[(key as usize) % SHARDS];
        {
            let mut s = shard.lock().unwrap();
            s.tick += 1;
            let tick = s.tick;
            if let Some(e) = s.map.get_mut(&key) {
                e.last_used = tick;
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                return e.data.clone();
            }
        }
        // Miss: read outside the shard lock (pread; no seek state).
        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        let data = {
            let files = self.files.read().unwrap();
            let tf = &files[id as usize];
            Arc::new(
                tex::read_tile(&tf.file, &tf.header, mip, tx, ty)
                    .unwrap_or_else(|_| vec![0.0; TILE_SIZE * TILE_SIZE * 3]),
            )
        };
        let bytes = data.len() * 4;
        self.stats.bytes_read.fetch_add(bytes as u64, Ordering::Relaxed);
        let mut s = shard.lock().unwrap();
        while s.bytes + bytes > self.shard_budget && !s.map.is_empty() {
            // LRU eviction: scan for the stalest entry (shards stay small).
            if let Some((&victim, _)) = s.map.iter().min_by_key(|(_, e)| e.last_used) {
                if let Some(e) = s.map.remove(&victim) {
                    s.bytes -= e.data.len() * 4;
                    self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        s.tick += 1;
        let tick = s.tick;
        s.bytes += bytes;
        s.map.insert(key, Entry { data: data.clone(), last_used: tick });
        data
    }

    /// One texel from a mip level, wrap-resolved. Coordinates are integer
    /// texel indices possibly out of range.
    fn texel(&self, id: TexId, header: &TexHeader, mip: usize, x: i64, y: i64, wrap: Wrap) -> [f32; 3] {
        let info = &header.mips[mip];
        let (w, h) = (info.width as i64, info.height as i64);
        let (x, y) = match wrap {
            Wrap::Periodic => (x.rem_euclid(w), y.rem_euclid(h)),
            Wrap::Clamp => (x.clamp(0, w - 1), y.clamp(0, h - 1)),
            Wrap::Black => {
                if x < 0 || x >= w || y < 0 || y >= h {
                    return [0.0; 3];
                }
                (x, y)
            }
        };
        let ts = TILE_SIZE as i64;
        let tile = self.tile(id, mip, (x / ts) as u32, (y / ts) as u32);
        let i = ((y % ts) as usize * TILE_SIZE + (x % ts) as usize) * 3;
        [tile[i], tile[i + 1], tile[i + 2]]
    }

    /// Bilinear sample of one mip level at (s, t) in [0,1] (t = 0 at the
    /// top row, matching image ingest).
    fn bilinear(&self, id: TexId, header: &TexHeader, mip: usize, s: f64, t: f64, wrap: Wrap) -> [f32; 3] {
        let info = &header.mips[mip];
        let fx = s * info.width as f64 - 0.5;
        let fy = t * info.height as f64 - 0.5;
        let x0 = fx.floor();
        let y0 = fy.floor();
        let (ax, ay) = (fx - x0, fy - y0);
        let (x0, y0) = (x0 as i64, y0 as i64);
        let mut out = [0.0f32; 3];
        for (dx, dy, w) in [
            (0, 0, (1.0 - ax) * (1.0 - ay)),
            (1, 0, ax * (1.0 - ay)),
            (0, 1, (1.0 - ax) * ay),
            (1, 1, ax * ay),
        ] {
            let px = self.texel(id, header, mip, x0 + dx, y0 + dy, wrap);
            for c in 0..3 {
                out[c] += px[c] * w as f32;
            }
        }
        out
    }

    /// Trilinear lookup. `footprint` is the pixel footprint's diameter in
    /// st units (0 = sharpest mip).
    pub fn sample(&self, id: TexId, s: f64, t: f64, footprint: f64, wrap: Wrap) -> [f32; 3] {
        let header = self.header(id);
        let base = header.width.max(header.height) as f64;
        let texels = (footprint * base).max(1e-9);
        let mip_f = texels.log2().clamp(0.0, (header.mips.len() - 1) as f64);
        let m0 = mip_f.floor() as usize;
        let m1 = (m0 + 1).min(header.mips.len() - 1);
        let a = mip_f - m0 as f64;
        let lo = self.bilinear(id, &header, m0, s, t, wrap);
        if a < 1e-6 || m1 == m0 {
            return lo;
        }
        let hi = self.bilinear(id, &header, m1, s, t, wrap);
        let mut out = [0.0f32; 3];
        for c in 0..3 {
            out[c] = lo[c] * (1.0 - a) as f32 + hi[c] * a as f32;
        }
        out
    }

    /// Read an entire mip level as row-major RGB f32 (de-tiled). Used to
    /// build GPU-resident texture tables.
    pub fn read_mip(&self, id: TexId, mip: usize) -> (u32, u32, Vec<f32>) {
        let header = self.header(id);
        let info = &header.mips[mip];
        let (w, h) = (info.width as usize, info.height as usize);
        let mut out = vec![0.0f32; w * h * 3];
        for ty in 0..info.tiles_y {
            for tx in 0..info.tiles_x {
                let tile = self.tile(id, mip, tx, ty);
                for py in 0..TILE_SIZE {
                    let y = ty as usize * TILE_SIZE + py;
                    if y >= h {
                        break;
                    }
                    for px in 0..TILE_SIZE {
                        let x = tx as usize * TILE_SIZE + px;
                        if x >= w {
                            break;
                        }
                        let src = (py * TILE_SIZE + px) * 3;
                        let dst = (y * w + x) * 3;
                        out[dst..dst + 3].copy_from_slice(&tile[src..src + 3]);
                    }
                }
            }
        }
        (info.width, info.height, out)
    }

    pub fn stats_line(&self) -> String {
        let h = self.stats.hits.load(Ordering::Relaxed);
        let m = self.stats.misses.load(Ordering::Relaxed);
        let e = self.stats.evictions.load(Ordering::Relaxed);
        let b = self.stats.bytes_read.load(Ordering::Relaxed);
        let rate = if h + m > 0 { 100.0 * h as f64 / (h + m) as f64 } else { 0.0 };
        format!(
            "texture cache: {h} hits / {m} misses ({rate:.1}% hit rate), {e} evictions, {:.1} MB read",
            b as f64 / (1024.0 * 1024.0)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture::tex::{write_tex, LinearImage};

    fn gradient_tex(path: &Path, w: usize, h: usize) {
        let mut pixels = vec![0.0f32; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                pixels[i] = x as f32 / (w - 1) as f32;
                pixels[i + 1] = y as f32 / (h - 1) as f32;
                pixels[i + 2] = 0.5;
            }
        }
        write_tex(path, &LinearImage { width: w, height: h, pixels }).unwrap();
    }

    #[test]
    fn sample_center_and_wrap() {
        let path = std::env::temp_dir().join("render_rs_cache_test.tex");
        gradient_tex(&path, 128, 128);
        let cache = TextureCache::new(64 * 1024 * 1024);
        let id = cache.open(&path).unwrap();
        // Center texel: red ~ 0.5, green ~ 0.5.
        let c = cache.sample(id, 0.5, 0.5, 0.0, Wrap::Periodic);
        assert!((c[0] - 0.5).abs() < 0.02 && (c[1] - 0.5).abs() < 0.02, "{c:?}");
        // s just past 1 wraps to the left edge (red ~ 0).
        let wrapped = cache.sample(id, 1.02, 0.5, 0.0, Wrap::Periodic);
        assert!(wrapped[0] < 0.1, "{wrapped:?}");
        // Black wrap outside is black.
        let black = cache.sample(id, 1.5, 0.5, 0.0, Wrap::Black);
        assert_eq!(black, [0.0; 3]);
        // Huge footprint hits the 1x1 mip: overall mean ~ (0.5, 0.5, 0.5).
        let top = cache.sample(id, 0.25, 0.75, 10.0, Wrap::Periodic);
        assert!((top[0] - 0.5).abs() < 0.05 && (top[2] - 0.5).abs() < 0.01, "{top:?}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn eviction_respects_budget() {
        let path = std::env::temp_dir().join("render_rs_cache_evict.tex");
        gradient_tex(&path, 512, 512);
        // Budget of ~24 tiles total across 16 shards.
        let tile_bytes = TILE_SIZE * TILE_SIZE * 3 * 4;
        let cache = TextureCache::new(tile_bytes * 24);
        let id = cache.open(&path).unwrap();
        for i in 0..16 {
            for j in 0..16 {
                cache.sample(
                    id,
                    i as f64 / 15.0 * 0.99,
                    j as f64 / 15.0 * 0.99,
                    0.0,
                    Wrap::Clamp,
                );
            }
        }
        assert!(cache.stats.evictions.load(Ordering::Relaxed) > 0);
        let held: usize = cache.shards.iter().map(|s| s.lock().unwrap().bytes).sum();
        assert!(held <= tile_bytes * 24 + tile_bytes * SHARDS, "held {held}");
        std::fs::remove_file(&path).ok();
    }
}
