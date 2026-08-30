//! Accumulation images for distributed rendering (roadmap P12): a node
//! renders a *sample range* and writes the raw radiance SUM plus a
//! per-pixel sample count; `render merge` averages any number of them.
//! Deterministic per-(pixel, sample) seeding makes the partition exact —
//! merging ranges [0,a) and [a,b) equals rendering [0,b) on one machine.
//!
//! On disk: an EXR with channels R, G, B (sums) and `count`.

use crate::math::Vec3;
use crate::output::Image;
use anyhow::{bail, Result};
use exr::prelude::*;

pub fn write_accum_exr(
    filename: &str,
    sum: &Image,
    count: &[Vec<f64>],
) -> Result<()> {
    let h = sum.len();
    let w = sum.first().map(|r| r.len()).unwrap_or(0);
    let plane = |f: &dyn Fn(usize, usize) -> f32| -> FlatSamples {
        let mut v = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                v.push(f(y, x));
            }
        }
        FlatSamples::F32(v)
    };
    let mut channels = vec![
        AnyChannel::new("R", plane(&|y, x| sum[y][x].x as f32)),
        AnyChannel::new("G", plane(&|y, x| sum[y][x].y as f32)),
        AnyChannel::new("B", plane(&|y, x| sum[y][x].z as f32)),
        AnyChannel::new("count", plane(&|y, x| count[y][x] as f32)),
    ];
    channels.sort_by(|a, b| a.name.cmp(&b.name));
    let layer = Layer::new(
        (w, h),
        LayerAttributes::named("accum"),
        Encoding::FAST_LOSSLESS,
        AnyChannels::sort(SmallVec::from_vec(channels)),
    );
    exr::prelude::Image::from_layer(layer)
        .write()
        .to_file(filename)
        .map_err(|e| anyhow::anyhow!("accum EXR write failed: {e}"))?;
    Ok(())
}

pub fn read_accum_exr(filename: &str) -> Result<(Image, Vec<Vec<f64>>)> {
    let img = exr::prelude::read()
        .no_deep_data()
        .largest_resolution_level()
        .all_channels()
        .all_layers()
        .all_attributes()
        .from_file(filename)
        .map_err(|e| anyhow::anyhow!("accum EXR read failed: {e}"))?;
    let layer = &img.layer_data[0];
    let (w, h) = (layer.size.x(), layer.size.y());
    let chan = |name: &str| -> Result<Vec<f32>> {
        let c = layer
            .channel_data
            .list
            .iter()
            .find(|c| c.name.to_string() == name)
            .ok_or_else(|| anyhow::anyhow!("accum EXR missing channel {name}"))?;
        match &c.sample_data {
            FlatSamples::F32(v) => Ok(v.clone()),
            FlatSamples::F16(v) => Ok(v.iter().map(|x| x.to_f32()).collect()),
            FlatSamples::U32(_) => bail!("unexpected u32 channel {name}"),
        }
    };
    let (r, g, b, n) = (chan("R")?, chan("G")?, chan("B")?, chan("count")?);
    let mut sum = vec![vec![Vec3::zero(); w]; h];
    let mut count = vec![vec![0.0f64; w]; h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            sum[y][x] = Vec3::new(r[i] as f64, g[i] as f64, b[i] as f64);
            count[y][x] = n[i] as f64;
        }
    }
    Ok((sum, count))
}

/// Merge accumulation files into a mean image.
pub fn merge(files: &[String]) -> Result<Image> {
    if files.is_empty() {
        bail!("merge needs at least one accum file");
    }
    let (mut sum, mut count) = read_accum_exr(&files[0])?;
    for f in &files[1..] {
        let (s2, c2) = read_accum_exr(f)?;
        if s2.len() != sum.len() || s2[0].len() != sum[0].len() {
            bail!("{f}: resolution mismatch");
        }
        for (rs, r2) in sum.iter_mut().zip(&s2) {
            for (ps, p2) in rs.iter_mut().zip(r2) {
                *ps = *ps + *p2;
            }
        }
        for (rc, r2) in count.iter_mut().zip(&c2) {
            for (pc, p2) in rc.iter_mut().zip(r2) {
                *pc += *p2;
            }
        }
    }
    Ok(sum
        .iter()
        .zip(&count)
        .map(|(rs, rc)| {
            rs.iter()
                .zip(rc)
                .map(|(p, c)| if *c > 0.0 { *p / *c } else { Vec3::zero() })
                .collect()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accum_round_trip_and_merge() {
        let (w, h) = (8, 6);
        let img_a: Image = (0..h)
            .map(|y| (0..w).map(|x| Vec3::new(x as f64, y as f64, 1.0) * 4.0).collect())
            .collect();
        let img_b: Image = (0..h)
            .map(|y| (0..w).map(|x| Vec3::new(x as f64, y as f64, 3.0) * 4.0).collect())
            .collect();
        let counts = vec![vec![4.0f64; w]; h];
        let pa = std::env::temp_dir().join("render_rs_accum_a.exr");
        let pb = std::env::temp_dir().join("render_rs_accum_b.exr");
        write_accum_exr(pa.to_str().unwrap(), &img_a, &counts).unwrap();
        write_accum_exr(pb.to_str().unwrap(), &img_b, &counts).unwrap();
        let merged = merge(&[
            pa.to_str().unwrap().to_string(),
            pb.to_str().unwrap().to_string(),
        ])
        .unwrap();
        // mean = (a + b) / 8 samples
        let expect = (img_a[3][5] + img_b[3][5]) / 8.0;
        assert!((merged[3][5] - expect).length() < 1e-5);
        std::fs::remove_file(&pa).ok();
        std::fs::remove_file(&pb).ok();
    }
}
