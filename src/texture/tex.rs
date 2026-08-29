//! The renderer's own tiled, mip-mapped texture file format (roadmap
//! Phase 6) plus the txmake-equivalent converter. Layout:
//!
//! ```text
//! magic  b"RTEX"            4 bytes
//! version u32               1
//! width  u32 / height u32   base resolution
//! channels u32              3 (RGB, linear f32)
//! tile_size u32             32
//! mip_count u32
//! per mip: width u32, height u32, tiles_x u32, tiles_y u32, offset u64
//! tile data                 tiles row-major per mip; each tile is
//!                           tile_size^2 * channels f32 (little-endian),
//!                           edge tiles padded by edge replication
//! ```
//!
//! Fixed-size tiles keep offset math trivial and make edge filtering safe
//! without per-tile bounds checks. All pixel data is linear; 8-bit source
//! images are sRGB-decoded at txmake time, float sources pass through.

use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub const TILE_SIZE: usize = 32;
pub const MAGIC: &[u8; 4] = b"RTEX";

#[derive(Debug, Clone)]
pub struct MipInfo {
    pub width: u32,
    pub height: u32,
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub offset: u64,
}

#[derive(Debug, Clone)]
pub struct TexHeader {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub tile_size: u32,
    pub mips: Vec<MipInfo>,
}

impl TexHeader {
    pub fn tile_bytes(&self) -> usize {
        (self.tile_size * self.tile_size * self.channels) as usize * 4
    }
}

/// A base image in linear RGB f32.
pub struct LinearImage {
    pub width: usize,
    pub height: usize,
    /// RGB triples, row-major from the top-left.
    pub pixels: Vec<f32>,
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Load any supported image (PNG/JPEG/TIFF/EXR/HDR) as linear RGB.
/// 8-/16-bit integer sources are assumed sRGB-encoded and are decoded;
/// float sources are taken as already linear.
pub fn load_linear(path: &Path) -> Result<LinearImage, String> {
    let img = image::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let ldr = !matches!(
        img.color(),
        image::ColorType::Rgb32F | image::ColorType::Rgba32F
    );
    let rgb = img.to_rgb32f();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let mut pixels = rgb.into_raw();
    if ldr {
        for p in pixels.iter_mut() {
            *p = srgb_to_linear(*p);
        }
    }
    Ok(LinearImage { width: w, height: h, pixels })
}

/// One 2x2 box-filter downsample (odd dimensions clamp the second tap).
fn downsample(img: &LinearImage) -> LinearImage {
    let w = (img.width / 2).max(1);
    let h = (img.height / 2).max(1);
    let mut pixels = vec![0.0f32; w * h * 3];
    for y in 0..h {
        let y0 = (y * 2).min(img.height - 1);
        let y1 = (y * 2 + 1).min(img.height - 1);
        for x in 0..w {
            let x0 = (x * 2).min(img.width - 1);
            let x1 = (x * 2 + 1).min(img.width - 1);
            for c in 0..3 {
                let s = img.pixels[(y0 * img.width + x0) * 3 + c]
                    + img.pixels[(y0 * img.width + x1) * 3 + c]
                    + img.pixels[(y1 * img.width + x0) * 3 + c]
                    + img.pixels[(y1 * img.width + x1) * 3 + c];
                pixels[(y * w + x) * 3 + c] = s * 0.25;
            }
        }
    }
    LinearImage { width: w, height: h, pixels }
}

/// Write a .tex file: full mip chain down to 1x1, 32x32 tiles.
pub fn write_tex(path: &Path, base: &LinearImage) -> Result<(), String> {
    let channels = 3u32;
    let tile = TILE_SIZE;

    // Build the mip chain.
    let mut mips_img = vec![LinearImage {
        width: base.width,
        height: base.height,
        pixels: base.pixels.clone(),
    }];
    while mips_img.last().unwrap().width > 1 || mips_img.last().unwrap().height > 1 {
        mips_img.push(downsample(mips_img.last().unwrap()));
    }

    // Header size: magic + 6 u32s (version, w, h, channels, tile, mips)
    // + per-mip (4 u32s + u64 offset).
    let header_bytes = 4 + 24 + mips_img.len() * 24;
    let tile_bytes = tile * tile * channels as usize * 4;

    let mut infos = Vec::with_capacity(mips_img.len());
    let mut offset = header_bytes as u64;
    for m in &mips_img {
        let tiles_x = m.width.div_ceil(tile) as u32;
        let tiles_y = m.height.div_ceil(tile) as u32;
        infos.push(MipInfo {
            width: m.width as u32,
            height: m.height as u32,
            tiles_x,
            tiles_y,
            offset,
        });
        offset += (tiles_x * tiles_y) as u64 * tile_bytes as u64;
    }

    let f = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut w = BufWriter::new(f);
    let put_u32 = |w: &mut BufWriter<File>, v: u32| w.write_all(&v.to_le_bytes());
    w.write_all(MAGIC).map_err(|e| e.to_string())?;
    for v in [1u32, base.width as u32, base.height as u32, channels, tile as u32] {
        put_u32(&mut w, v).map_err(|e| e.to_string())?;
    }
    put_u32(&mut w, mips_img.len() as u32).map_err(|e| e.to_string())?;
    for info in &infos {
        for v in [info.width, info.height, info.tiles_x, info.tiles_y] {
            put_u32(&mut w, v).map_err(|e| e.to_string())?;
        }
        w.write_all(&info.offset.to_le_bytes()).map_err(|e| e.to_string())?;
    }

    // Tile data, edge-replicated to full tiles.
    let mut buf = vec![0.0f32; tile * tile * 3];
    for (m, info) in mips_img.iter().zip(&infos) {
        for ty in 0..info.tiles_y as usize {
            for tx in 0..info.tiles_x as usize {
                for py in 0..tile {
                    let sy = (ty * tile + py).min(m.height - 1);
                    for px in 0..tile {
                        let sx = (tx * tile + px).min(m.width - 1);
                        let src = (sy * m.width + sx) * 3;
                        let dst = (py * tile + px) * 3;
                        buf[dst..dst + 3].copy_from_slice(&m.pixels[src..src + 3]);
                    }
                }
                let bytes: Vec<u8> = buf.iter().flat_map(|v| v.to_le_bytes()).collect();
                w.write_all(&bytes).map_err(|e| e.to_string())?;
            }
        }
    }
    w.flush().map_err(|e| e.to_string())
}

fn get_u32(r: &mut impl Read) -> Result<u32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(b))
}

pub fn read_header(f: &mut File) -> Result<TexHeader, String> {
    f.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != MAGIC {
        return Err("not a .tex file (bad magic)".into());
    }
    let version = get_u32(f)?;
    if version != 1 {
        return Err(format!("unsupported .tex version {version}"));
    }
    let width = get_u32(f)?;
    let height = get_u32(f)?;
    let channels = get_u32(f)?;
    let tile_size = get_u32(f)?;
    let mip_count = get_u32(f)?;
    if channels != 3 || tile_size as usize != TILE_SIZE || mip_count > 32 {
        return Err("unsupported .tex layout".into());
    }
    let mut mips = Vec::with_capacity(mip_count as usize);
    for _ in 0..mip_count {
        let width = get_u32(f)?;
        let height = get_u32(f)?;
        let tiles_x = get_u32(f)?;
        let tiles_y = get_u32(f)?;
        let mut b = [0u8; 8];
        f.read_exact(&mut b).map_err(|e| e.to_string())?;
        mips.push(MipInfo { width, height, tiles_x, tiles_y, offset: u64::from_le_bytes(b) });
    }
    Ok(TexHeader { width, height, channels, tile_size, mips })
}

/// Read one tile's pixel block (TILE_SIZE^2 RGB f32s).
pub fn read_tile(
    f: &File,
    header: &TexHeader,
    mip: usize,
    tx: u32,
    ty: u32,
) -> Result<Vec<f32>, String> {
    use std::os::unix::fs::FileExt;
    let info = header.mips.get(mip).ok_or("mip out of range")?;
    if tx >= info.tiles_x || ty >= info.tiles_y {
        return Err("tile out of range".into());
    }
    let tile_bytes = header.tile_bytes();
    let offset = info.offset + (ty as u64 * info.tiles_x as u64 + tx as u64) * tile_bytes as u64;
    let mut bytes = vec![0u8; tile_bytes];
    f.read_exact_at(&mut bytes, offset).map_err(|e| e.to_string())?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// The txmake entry point: any supported image in, .tex out.
pub fn txmake(input: &Path, output: &Path) -> Result<TexHeader, String> {
    let img = load_linear(input)?;
    write_tex(output, &img)?;
    let mut f = File::open(output).map_err(|e| e.to_string())?;
    read_header(&mut f)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checker(width: usize, height: usize) -> LinearImage {
        let mut pixels = vec![0.0f32; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let v = if (x / 8 + y / 8) % 2 == 0 { 1.0 } else { 0.25 };
                let i = (y * width + x) * 3;
                pixels[i] = v;
                pixels[i + 1] = v * 0.5;
                pixels[i + 2] = 0.1;
            }
        }
        LinearImage { width, height, pixels }
    }

    #[test]
    fn tex_round_trip_and_mips() {
        let dir = std::env::temp_dir();
        let path = dir.join("render_rs_test_roundtrip.tex");
        let img = checker(70, 50); // deliberately non-tile-aligned, non-pow2
        write_tex(&path, &img).unwrap();
        let mut f = File::open(&path).unwrap();
        let header = read_header(&mut f).unwrap();
        assert_eq!((header.width, header.height), (70, 50));
        // 70x50 -> 35x25 -> 17x12 -> 8x6 -> 4x3 -> 2x1 -> 1x1
        assert_eq!(header.mips.len(), 7);
        assert_eq!(header.mips[0].tiles_x, 3);
        assert_eq!(header.mips[0].tiles_y, 2);
        // Base mip pixel (40, 20) via its tile.
        let tile = read_tile(&f, &header, 0, 1, 0).unwrap();
        let (px, py) = (40 - TILE_SIZE, 20);
        let got = tile[(py * TILE_SIZE + px) * 3];
        assert_eq!(got, img.pixels[(20 * 70 + 40) * 3]);
        // Last mip is 1x1 and equals the image mean (box chain approximates;
        // just check it is within the value range).
        let last = header.mips.len() - 1;
        assert_eq!(header.mips[last].width, 1);
        let top = read_tile(&f, &header, last, 0, 0).unwrap();
        assert!(top[0] > 0.25 && top[0] < 1.0, "1x1 mip = {}", top[0]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn edge_tiles_replicate() {
        let dir = std::env::temp_dir();
        let path = dir.join("render_rs_test_edge.tex");
        let img = checker(40, 40); // right/bottom tiles are partial
        write_tex(&path, &img).unwrap();
        let mut f = File::open(&path).unwrap();
        let header = read_header(&mut f).unwrap();
        let tile = read_tile(&f, &header, 0, 1, 1).unwrap();
        // Pixel (39,39) is the last valid one; padded region repeats it.
        let valid = img.pixels[(39 * 40 + 39) * 3];
        let padded = tile[(20 * TILE_SIZE + 20) * 3]; // (52,52) -> clamped
        assert_eq!(padded, valid);
        std::fs::remove_file(&path).ok();
    }
}
