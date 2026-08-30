//! objc2-metal host code: compile the embedded MSL kernel at runtime,
//! upload the flattened scene, dispatch one thread per pixel in row bands,
//! and read the result back into an Image.

use super::scene_buffers::{GpuUniforms, SceneBuffers};
use crate::math::Vec3;
use crate::output::Image;
use crate::raytracer::flatten::FlatScene;
use crate::scene::Scene;
use anyhow::{anyhow, Result};
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2_foundation::{ns_string, NSString};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;

// MTLCreateSystemDefaultDevice lives in CoreGraphics.
#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}

const ISECT_COMMON_SRC: &str = include_str!("isect_common.metal");
const WHITTED_SRC: &str = include_str!("kernel.metal");
const PT_SRC: &str = include_str!("kernel_pt.metal");
const PATTERN_PRELUDE_SRC: &str = include_str!("pattern_prelude.metal");
/// GPU-watchdog insurance: bounded work per command buffer.
const ROWS_PER_BAND: usize = 256;
/// The path tracer keeps each command buffer small — one sample over a
/// bounded row band — so heavy scenes never trip the macOS GPU watchdog
/// ("Impacting Interactivity" kills buffers that run too long). The band
/// height scales inversely with image width (bounding pixels per buffer);
/// RENDER_PT_BAND_ROWS overrides for very heavy or very light scenes.
fn pt_rows_per_band(width: usize) -> usize {
    if let Ok(v) = std::env::var("RENDER_PT_BAND_ROWS") {
        if let Ok(n) = v.parse::<usize>() {
            return n.clamp(8, 4096);
        }
    }
    (250_000 / width.max(1)).clamp(16, 512)
}

fn whitted_source() -> String {
    format!("{ISECT_COMMON_SRC}\n{WHITTED_SRC}")
}

/// PT kernel source: common intersectors, the pattern runtime, the
/// scene-specific generated pattern functions, then the kernel itself.
fn pt_source(pattern_msl: &str) -> String {
    format!("{ISECT_COMMON_SRC}\n{PATTERN_PRELUDE_SRC}\n{pattern_msl}\n{PT_SRC}")
}

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

pub fn render(scene: &Scene) -> Result<Image> {
    let flat = FlatScene::from_scene(scene)?;
    let bufs = SceneBuffers::build(&flat);
    autoreleasepool(|_| render_impl(&flat, &bufs))
}

fn render_impl(flat: &FlatScene, bufs: &SceneBuffers) -> Result<Image> {
    let device = MTLCreateSystemDefaultDevice()
        .ok_or_else(|| anyhow!("no Metal device available"))?;

    // Runtime MSL compile. Fast math OFF: the kernel's closest-hit and
    // distant-light shadow logic rely on IEEE INFINITY semantics.
    let options = MTLCompileOptions::new();
    #[allow(deprecated)]
    options.setFastMathEnabled(false);
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(&whitted_source()), Some(&options))
        .map_err(|e| anyhow!("MSL compilation failed:\n{}", e.localizedDescription()))?;
    let function = library
        .newFunctionWithName(ns_string!("render_pixels"))
        .ok_or_else(|| anyhow!("kernel entry point `render_pixels` not found"))?;
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| anyhow!("compute pipeline creation failed: {}", e.localizedDescription()))?;
    let queue = device
        .newCommandQueue()
        .ok_or_else(|| anyhow!("failed to create Metal command queue"))?;

    let obj_buf = upload(&device, bufs.objects_bytes())?;
    let mat_buf = upload(&device, bufs.materials_bytes())?;
    let light_buf = upload(&device, bufs.lights_bytes())?;

    let (w, h) = (flat.width as usize, flat.height as usize);
    let out_len = w * h * 4 * std::mem::size_of::<f32>();
    let out_buf = device
        .newBufferWithLength_options(out_len, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| anyhow!("output buffer allocation failed ({out_len} bytes)"))?;

    let max_tg = pipeline.maxTotalThreadsPerThreadgroup();
    let tg_side = if max_tg >= 256 { 16 } else { 8 };
    let tg = MTLSize { width: tg_side, height: tg_side, depth: 1 };

    let mut y0 = 0usize;
    while y0 < h {
        let band = (h - y0).min(ROWS_PER_BAND);
        let mut uniforms: GpuUniforms = bufs.uniforms;
        uniforms.y_offset = y0 as u32;

        let cmd = queue
            .commandBuffer()
            .ok_or_else(|| anyhow!("failed to create command buffer"))?;
        let enc = cmd
            .computeCommandEncoder()
            .ok_or_else(|| anyhow!("failed to create compute encoder"))?;
        enc.setComputePipelineState(&pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&obj_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&mat_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&light_buf), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(&uniforms as *const GpuUniforms as *mut c_void).unwrap(),
                std::mem::size_of::<GpuUniforms>(),
                3,
            );
            enc.setBuffer_offset_atIndex(Some(&out_buf), 0, 4);
        }
        let grid = MTLSize { width: w, height: band, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if cmd.status() != MTLCommandBufferStatus::Completed {
            let detail = cmd
                .error()
                .map(|e| e.localizedDescription().to_string())
                .unwrap_or_else(|| "no error detail".to_string());
            return Err(anyhow!("GPU command buffer failed: {detail}"));
        }
        y0 += band;
    }

    // StorageModeShared: contents() is host-visible after waitUntilCompleted.
    let ptr = out_buf.contents().as_ptr() as *const f32;
    let data = unsafe { std::slice::from_raw_parts(ptr, w * h * 4) };
    let mut image = vec![vec![Vec3::zero(); w]; h];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            image[y][x] = Vec3::new(data[i] as f64, data[i + 1] as f64, data[i + 2] as f64);
        }
    }
    Ok(image)
}

/// Path-traced render on the GPU: same light transport as the CPU
/// reference (`raytracer::pt`), f32, statistically convergent to the same
/// image. Samples run in batches of PT_SAMPLE_BATCH per command buffer.
pub fn render_pt(scene: &Scene, spp: u32) -> Result<Image> {
    render_pt_checkpointed(scene, spp, None)
}

/// Path tracing with optional checkpoint/resume: the accumulation buffer
/// and completed-sample counter persist to `checkpoint` every
/// CHECKPOINT_EVERY samples (atomic tmp+rename), and a matching file is
/// loaded on start so interrupted renders continue where they stopped.
pub fn render_pt_checkpointed(
    scene: &Scene,
    spp: u32,
    checkpoint: Option<&std::path::Path>,
) -> Result<Image> {
    let gpu = super::gpu_scene::GpuPtScene::build(scene)?;
    autoreleasepool(|_| render_pt_impl(&gpu, spp, checkpoint))
}

const CHECKPOINT_MAGIC: &[u8; 4] = b"RCKP";
const CHECKPOINT_EVERY: u32 = 4;

fn save_checkpoint(path: &std::path::Path, w: u32, h: u32, done: u32, accum: &[f32]) {
    let mut bytes = Vec::with_capacity(16 + accum.len() * 4);
    bytes.extend_from_slice(CHECKPOINT_MAGIC);
    bytes.extend_from_slice(&w.to_le_bytes());
    bytes.extend_from_slice(&h.to_le_bytes());
    bytes.extend_from_slice(&done.to_le_bytes());
    for v in accum {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let tmp = path.with_extension("tmp");
    if std::fs::write(&tmp, &bytes).and_then(|_| std::fs::rename(&tmp, path)).is_err() {
        eprintln!("warning: checkpoint write to {} failed", path.display());
    }
}

fn load_checkpoint(path: &std::path::Path, w: u32, h: u32) -> Option<(u32, Vec<f32>)> {
    let bytes = std::fs::read(path).ok()?;
    let expect = 16 + (w as usize * h as usize * 4) * 4;
    if bytes.len() != expect || &bytes[0..4] != CHECKPOINT_MAGIC {
        return None;
    }
    let rw = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
    let rh = u32::from_le_bytes(bytes[8..12].try_into().ok()?);
    let done = u32::from_le_bytes(bytes[12..16].try_into().ok()?);
    if rw != w || rh != h {
        return None;
    }
    let accum: Vec<f32> = bytes[16..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Some((done, accum))
}

fn render_pt_impl(
    gpu: &super::gpu_scene::GpuPtScene,
    spp: u32,
    checkpoint: Option<&std::path::Path>,
) -> Result<Image> {
    use super::gpu_scene::GpuPtUniforms;

    let device = MTLCreateSystemDefaultDevice()
        .ok_or_else(|| anyhow!("no Metal device available"))?;
    // Unlike the whitted kernel, the PT kernel is written infinity-free
    // (PT_BIG sentinels, safe_inv) precisely so fast math can stay ON —
    // it is a large win on this ALU-heavy kernel.
    let options = MTLCompileOptions::new();
    #[allow(deprecated)]
    options.setFastMathEnabled(true);
    let library = device
        .newLibraryWithSource_options_error(
            &NSString::from_str(&pt_source(&gpu.pattern_msl)),
            Some(&options),
        )
        .map_err(|e| anyhow!("PT MSL compilation failed:\n{}", e.localizedDescription()))?;
    let function = library
        .newFunctionWithName(ns_string!("render_pt"))
        .ok_or_else(|| anyhow!("kernel entry point `render_pt` not found"))?;
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| anyhow!("PT pipeline creation failed: {}", e.localizedDescription()))?;
    let queue = device
        .newCommandQueue()
        .ok_or_else(|| anyhow!("failed to create Metal command queue"))?;

    let buffers: Vec<Buffer> = [
        gpu.objects_bytes(),
        gpu.object_materials_bytes(),
        gpu.materials_bytes(),
        gpu.lights_bytes(),
        gpu.tlas_bytes(),
        gpu.instances_bytes(),
        gpu.blas_bytes(),
        gpu.tri_indices_bytes(),
        gpu.vertices_bytes(),
        gpu.normals_bytes(),
        gpu.mesh_infos_bytes(),
        gpu.env_pixels_bytes(),
        gpu.env_marginal_bytes(),
        gpu.env_conditional_bytes(),
        gpu.st_bytes(),
        gpu.tex_data_bytes(),
        gpu.tex_mips_bytes(),
        gpu.vertices1_bytes(),
        gpu.curve_segs_bytes(),
        gpu.curve_infos_bytes(),
        gpu.light_bvh_bytes(),
        gpu.light_aux_bytes(),
    ]
    .into_iter()
    .map(|bytes| upload(&device, bytes))
    .collect::<Result<_>>()?;

    let (w, h) = (gpu.uniforms.width as usize, gpu.uniforms.height as usize);
    let accum_len = w * h * 4 * std::mem::size_of::<f32>();
    let accum_buf = device
        .newBufferWithLength_options(accum_len, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| anyhow!("accumulation buffer allocation failed"))?;
    unsafe {
        std::ptr::write_bytes(accum_buf.contents().as_ptr() as *mut u8, 0, accum_len);
    }

    // Resume from a matching checkpoint.
    let mut sample_start = 0u32;
    if let Some(path) = checkpoint {
        if let Some((done, accum)) =
            load_checkpoint(path, gpu.uniforms.width, gpu.uniforms.height)
        {
            let done = done.min(spp);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    accum.as_ptr(),
                    accum_buf.contents().as_ptr() as *mut f32,
                    accum.len(),
                );
            }
            sample_start = done;
            println!("Resuming from checkpoint: {done}/{spp} samples done");
        }
    }

    // 8x8 threadgroups: the PT kernel is register-heavy (traversal
    // stacks), and smaller groups schedule better under that pressure.
    let _ = pipeline.maxTotalThreadsPerThreadgroup();
    let tg = MTLSize { width: 8, height: 8, depth: 1 };

    let band_rows = pt_rows_per_band(w);
    for sample in sample_start..spp {
        let mut y0 = 0usize;
        while y0 < h {
            let band = (h - y0).min(band_rows);
            let mut uniforms: GpuPtUniforms = gpu.uniforms;
            uniforms.sample_start = sample;
            uniforms.sample_count = 1;
            uniforms.y_offset = y0 as u32;

            let cmd = queue
                .commandBuffer()
                .ok_or_else(|| anyhow!("failed to create command buffer"))?;
            let enc = cmd
                .computeCommandEncoder()
                .ok_or_else(|| anyhow!("failed to create compute encoder"))?;
            enc.setComputePipelineState(&pipeline);
            unsafe {
                for (i, buf) in buffers.iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(buf), 0, i);
                }
                enc.setBytes_length_atIndex(
                    NonNull::new(&uniforms as *const GpuPtUniforms as *mut c_void).unwrap(),
                    std::mem::size_of::<GpuPtUniforms>(),
                    22,
                );
                enc.setBuffer_offset_atIndex(Some(&accum_buf), 0, 23);
            }
            let grid = MTLSize { width: w, height: band, depth: 1 };
            enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            if cmd.status() != MTLCommandBufferStatus::Completed {
                let detail = cmd
                    .error()
                    .map(|e| e.localizedDescription().to_string())
                    .unwrap_or_else(|| "no error detail".to_string());
                // The macOS GPU watchdog kills buffers that run long
                // ("Impacting Interactivity") — transient and load-
                // dependent, not a real failure. Retry in small slices.
                // Pixels the killed buffer already finished get their
                // sample re-added, but color and weight (w channel) double
                // together, so the mean stays correct — only the per-pixel
                // effective sample count shifts slightly.
                if detail.contains("Interactivity") {
                    eprintln!(
                        "GPU watchdog killed a band at sample {sample}, y0={y0};                          retrying in 8-row slices"
                    );
                    let mut ry = y0;
                    let band_end = y0 + band;
                    while ry < band_end {
                        let rband = (band_end - ry).min(8);
                        let mut runi: GpuPtUniforms = gpu.uniforms;
                        runi.sample_start = sample;
                        runi.sample_count = 1;
                        runi.y_offset = ry as u32;
                        let cmd2 = queue
                            .commandBuffer()
                            .ok_or_else(|| anyhow!("failed to create command buffer"))?;
                        let enc2 = cmd2
                            .computeCommandEncoder()
                            .ok_or_else(|| anyhow!("failed to create compute encoder"))?;
                        enc2.setComputePipelineState(&pipeline);
                        unsafe {
                            for (i, buf) in buffers.iter().enumerate() {
                                enc2.setBuffer_offset_atIndex(Some(buf), 0, i);
                            }
                            enc2.setBytes_length_atIndex(
                                NonNull::new(&runi as *const GpuPtUniforms as *mut c_void)
                                    .unwrap(),
                                std::mem::size_of::<GpuPtUniforms>(),
                                22,
                            );
                            enc2.setBuffer_offset_atIndex(Some(&accum_buf), 0, 23);
                        }
                        let grid2 = MTLSize { width: w, height: rband, depth: 1 };
                        enc2.dispatchThreads_threadsPerThreadgroup(grid2, tg);
                        enc2.endEncoding();
                        cmd2.commit();
                        cmd2.waitUntilCompleted();
                        if cmd2.status() != MTLCommandBufferStatus::Completed {
                            return Err(anyhow!(
                                "PT command buffer failed even at 8 rows: {detail}"
                            ));
                        }
                        ry += rband;
                    }
                    y0 += band;
                    continue;
                }
                return Err(anyhow!("PT command buffer failed: {detail}"));
            }
            y0 += band;
        }
        if let Some(path) = checkpoint {
            let done = sample + 1;
            if done % CHECKPOINT_EVERY == 0 || done == spp {
                let ptr = accum_buf.contents().as_ptr() as *const f32;
                let accum = unsafe { std::slice::from_raw_parts(ptr, w * h * 4) };
                save_checkpoint(path, gpu.uniforms.width, gpu.uniforms.height, done, accum);
            }
        }
    }

    let ptr = accum_buf.contents().as_ptr() as *const f32;
    let data = unsafe { std::slice::from_raw_parts(ptr, w * h * 4) };
    let mut image = vec![vec![Vec3::zero(); w]; h];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            let weight = data[i + 3].max(1.0);
            image[y][x] = Vec3::new(
                (data[i] / weight) as f64,
                (data[i + 1] / weight) as f64,
                (data[i + 2] / weight) as f64,
            );
        }
    }
    Ok(image)
}

/// Test-support entry point: intersect one object with a batch of rays via
/// the `intersect_probe` kernel; returns (valid, t) per ray. Used by the
/// per-primitive parity tests.
pub fn intersect_probe(
    object: &crate::raytracer::flatten::FlatObject,
    rays: &[[f32; 6]],
) -> Result<Vec<(bool, f32)>> {
    autoreleasepool(|_| {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| anyhow!("no Metal device available"))?;
        let options = MTLCompileOptions::new();
        #[allow(deprecated)]
        options.setFastMathEnabled(false);
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(&whitted_source()), Some(&options))
            .map_err(|e| anyhow!("MSL compilation failed:\n{}", e.localizedDescription()))?;
        let function = library
            .newFunctionWithName(ns_string!("intersect_probe"))
            .ok_or_else(|| anyhow!("kernel entry point `intersect_probe` not found"))?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| anyhow!("pipeline creation failed: {}", e.localizedDescription()))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| anyhow!("failed to create Metal command queue"))?;

        let gpu_obj = [super::scene_buffers::gpu_object(object)];
        let obj_buf = upload(&device, super::scene_buffers::as_bytes(&gpu_obj))?;
        let ray_buf = upload(&device, super::scene_buffers::as_bytes(rays))?;
        let n = rays.len();
        let out_len = n * 2 * std::mem::size_of::<f32>();
        let out_buf = device
            .newBufferWithLength_options(out_len, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow!("probe output buffer allocation failed"))?;
        let ray_count = n as u32;

        let cmd = queue
            .commandBuffer()
            .ok_or_else(|| anyhow!("failed to create command buffer"))?;
        let enc = cmd
            .computeCommandEncoder()
            .ok_or_else(|| anyhow!("failed to create compute encoder"))?;
        enc.setComputePipelineState(&pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&obj_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&ray_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&out_buf), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(&ray_count as *const u32 as *mut c_void).unwrap(),
                std::mem::size_of::<u32>(),
                3,
            );
        }
        let grid = MTLSize { width: n, height: 1, depth: 1 };
        let tg = MTLSize { width: 64.min(n.max(1)), height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if cmd.status() != MTLCommandBufferStatus::Completed {
            return Err(anyhow!("probe command buffer failed"));
        }

        let ptr = out_buf.contents().as_ptr() as *const f32;
        let data = unsafe { std::slice::from_raw_parts(ptr, n * 2) };
        Ok((0..n).map(|i| (data[i * 2] != 0.0, data[i * 2 + 1])).collect())
    })
}

fn upload(device: &ProtocolObject<dyn MTLDevice>, bytes: &[u8]) -> Result<Buffer> {
    unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
            bytes.len(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or_else(|| anyhow!("Metal buffer upload failed ({} bytes)", bytes.len()))
}
