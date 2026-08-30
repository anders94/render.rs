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
const WF_SRC: &str = include_str!("kernel_wf.metal");
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

/// Wavefront source: everything the megakernel has plus the scheduler
/// kernels (they share pt_shade_step and the whole device library).
fn wf_source(pattern_msl: &str) -> String {
    format!("{}\n{WF_SRC}", pt_source(pattern_msl))
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
    autoreleasepool(|_| render_pt_impl(&gpu, spp, checkpoint).map(|(img, _)| img))
}

/// Path tracing with the full AOV stack (roadmap Phase 11).
pub fn render_pt_film(scene: &Scene, spp: u32) -> Result<crate::output::Film> {
    let gpu = super::gpu_scene::GpuPtScene::build(scene)?;
    let (beauty, aux) = autoreleasepool(|_| render_pt_impl(&gpu, spp, None))?;
    let (w, h) = (gpu.uniforms.width as usize, gpu.uniforms.height as usize);
    let mut film = crate::output::Film {
        beauty,
        diffuse: vec![vec![Vec3::zero(); w]; h],
        specular: vec![vec![Vec3::zero(); w]; h],
        albedo: vec![vec![Vec3::zero(); w]; h],
        normal: vec![vec![Vec3::zero(); w]; h],
        depth: vec![vec![Vec3::zero(); w]; h],
        id: vec![vec![Vec3::zero(); w]; h],
        manifest: scene.id_manifest.clone(),
    };
    let inv = 1.0 / spp as f64;
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 12;
            let hits = aux[i + 11] as f64;
            film.albedo[y][x] =
                Vec3::new(aux[i] as f64, aux[i + 1] as f64, aux[i + 2] as f64) * inv;
            film.normal[y][x] =
                Vec3::new(aux[i + 3] as f64, aux[i + 4] as f64, aux[i + 5] as f64) * inv;
            film.depth[y][x] = Vec3::new(
                if hits > 0.0 { aux[i + 6] as f64 / hits } else { 0.0 },
                0.0,
                0.0,
            );
            film.id[y][x] = Vec3::new(aux[i + 7] as f64, (hits * inv).min(1.0), 0.0);
            let d = Vec3::new(aux[i + 8] as f64, aux[i + 9] as f64, aux[i + 10] as f64) * inv;
            film.diffuse[y][x] = d;
            let b = film.beauty[y][x];
            film.specular[y][x] =
                Vec3::new((b.x - d.x).max(0.0), (b.y - d.y).max(0.0), (b.z - d.z).max(0.0));
        }
    }
    Ok(film)
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

/// A persistent GPU path-tracing session: compiled pipeline + uploaded
/// scene + accumulation state, able to add samples incrementally (the
/// interactive preview and the batch renderer share this). NOT Send —
/// create and use it on one thread.
pub struct PtSession {
    #[allow(dead_code)]
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn objc2_metal::MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    buffers: Vec<Buffer>,
    accum_buf: Buffer,
    aux_buf: Buffer,
    uniforms: super::gpu_scene::GpuPtUniforms,
    pub width: usize,
    pub height: usize,
    band_rows: usize,
}

impl PtSession {
    pub fn new(scene: &Scene) -> Result<Self> {
        let gpu = super::gpu_scene::GpuPtScene::build(scene)?;
        Self::from_gpu(&gpu)
    }

    fn from_gpu(gpu: &super::gpu_scene::GpuPtScene) -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| anyhow!("no Metal device available"))?;
        // Unlike the whitted kernel, the PT kernel is written
        // infinity-free (PT_BIG sentinels, safe_inv) precisely so fast
        // math can stay ON — a large win on this ALU-heavy kernel.
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
            gpu.media_bytes(),
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
        let aux_len = w * h * 12 * std::mem::size_of::<f32>();
        let aux_buf = device
            .newBufferWithLength_options(aux_len, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow!("aux buffer allocation failed"))?;
        unsafe {
            std::ptr::write_bytes(aux_buf.contents().as_ptr() as *mut u8, 0, aux_len);
        }
        let band_rows = pt_rows_per_band(w);
        Ok(Self {
            device,
            queue,
            pipeline,
            buffers,
            accum_buf,
            aux_buf,
            uniforms: gpu.uniforms,
            width: w,
            height: h,
            band_rows,
        })
    }

    fn dispatch_band(&self, sample: u32, y0: usize, band: usize) -> Result<()> {
        use super::gpu_scene::GpuPtUniforms;
        let mut uniforms: GpuPtUniforms = self.uniforms;
        uniforms.sample_start = sample;
        uniforms.sample_count = 1;
        uniforms.y_offset = y0 as u32;
        let cmd = self
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow!("failed to create command buffer"))?;
        let enc = cmd
            .computeCommandEncoder()
            .ok_or_else(|| anyhow!("failed to create compute encoder"))?;
        enc.setComputePipelineState(&self.pipeline);
        unsafe {
            for (i, buf) in self.buffers.iter().enumerate() {
                enc.setBuffer_offset_atIndex(Some(buf), 0, i);
            }
            enc.setBytes_length_atIndex(
                NonNull::new(&uniforms as *const GpuPtUniforms as *mut c_void).unwrap(),
                std::mem::size_of::<GpuPtUniforms>(),
                23,
            );
            enc.setBuffer_offset_atIndex(Some(&self.accum_buf), 0, 24);
            enc.setBuffer_offset_atIndex(Some(&self.aux_buf), 0, 25);
        }
        let tg = MTLSize { width: 8, height: 8, depth: 1 };
        let grid = MTLSize { width: self.width, height: band, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if cmd.status() != MTLCommandBufferStatus::Completed {
            let detail = cmd
                .error()
                .map(|e| e.localizedDescription().to_string())
                .unwrap_or_else(|| "no error detail".to_string());
            return Err(anyhow!("{detail}"));
        }
        Ok(())
    }

    /// Add samples [start, start+count) to the accumulation buffer.
    /// GPU-watchdog kills ("Impacting Interactivity") retry in 8-row
    /// slices: color and weight double together on re-added rows, so the
    /// mean stays correct.
    pub fn render_samples(&self, start: u32, count: u32) -> Result<()> {
        for sample in start..start + count {
            let mut y0 = 0usize;
            while y0 < self.height {
                let band = (self.height - y0).min(self.band_rows);
                if let Err(e) = self.dispatch_band(sample, y0, band) {
                    let detail = e.to_string();
                    if !detail.contains("Interactivity") {
                        return Err(anyhow!("PT command buffer failed: {detail}"));
                    }
                    eprintln!(
                        "GPU watchdog killed a band at sample {sample}, y0={y0}; \
                         retrying in 8-row slices"
                    );
                    let mut ry = y0;
                    while ry < y0 + band {
                        let rband = (y0 + band - ry).min(8);
                        self.dispatch_band(sample, ry, rband).map_err(|e2| {
                            anyhow!("PT command buffer failed even at 8 rows: {e2}")
                        })?;
                        ry += rband;
                    }
                }
                y0 += band;
            }
        }
        Ok(())
    }

    fn accum_slice(&self) -> &[f32] {
        let ptr = self.accum_buf.contents().as_ptr() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, self.width * self.height * 4) }
    }

    fn load_accum(&self, accum: &[f32]) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                accum.as_ptr(),
                self.accum_buf.contents().as_ptr() as *mut f32,
                accum.len().min(self.width * self.height * 4),
            );
        }
    }

    /// The accumulated image so far.
    pub fn image(&self) -> Image {
        let data = self.accum_slice();
        let (w, h) = (self.width, self.height);
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
        image
    }

    /// Raw accumulation: per-pixel radiance sums and sample weights
    /// (weights differ from the nominal count only on watchdog-retried
    /// rows). For distributed accumulation output.
    pub fn sum_and_weight(&self) -> (Image, Vec<Vec<f64>>) {
        let data = self.accum_slice();
        let (w, h) = (self.width, self.height);
        let mut sum = vec![vec![Vec3::zero(); w]; h];
        let mut weight = vec![vec![0.0f64; w]; h];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                sum[y][x] =
                    Vec3::new(data[i] as f64, data[i + 1] as f64, data[i + 2] as f64);
                weight[y][x] = data[i + 3] as f64;
            }
        }
        (sum, weight)
    }

    fn aux_vec(&self) -> Vec<f32> {
        let ptr = self.aux_buf.contents().as_ptr() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, self.width * self.height * 12) }.to_vec()
    }
}

/// Wavefront scheduler session: shares the scene buffers with the
/// megakernel but drives raygen/extend/shade kernels over compacted
/// path queues. Slab-based so 4K frames stay within memory.
pub struct WfSession {
    base: PtSession,
    p_raygen: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    p_extend: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    p_shade: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    paths_buf: Buffer,
    hits_buf: Buffer,
    q_a: Buffer,
    q_b: Buffer,
    stats_buf: Buffer,
    slab: usize,
    adaptive_tol: f32,
}

const WF_SLAB: usize = 1 << 21; // 2M paths per slab

impl WfSession {
    pub fn new(scene: &Scene) -> Result<Self> {
        let gpu = super::gpu_scene::GpuPtScene::build(scene)?;
        // Base session compiles the megakernel pipeline; we compile the
        // combined source for the wavefront entry points.
        let base = PtSession::from_gpu(&gpu)?;
        let options = MTLCompileOptions::new();
        #[allow(deprecated)]
        options.setFastMathEnabled(true);
        let library = base
            .device
            .newLibraryWithSource_options_error(
                &NSString::from_str(&wf_source(&gpu.pattern_msl)),
                Some(&options),
            )
            .map_err(|e| anyhow!("WF MSL compilation failed:\n{}", e.localizedDescription()))?;
        let pipeline_of = |name: &str| -> Result<_> {
            let f = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| anyhow!("kernel `{name}` not found"))?;
            base.device
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| anyhow!("{name} pipeline: {}", e.localizedDescription()))
        };
        let p_raygen = pipeline_of("wf_raygen")?;
        let p_extend = pipeline_of("wf_extend")?;
        let p_shade = pipeline_of("wf_shade")?;

        let slab = (base.width * base.height).min(WF_SLAB);
        let alloc = |bytes: usize| -> Result<Buffer> {
            base.device
                .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| anyhow!("wavefront buffer allocation failed ({bytes} B)"))
        };
        let paths_buf = alloc(slab * 128)?;
        let hits_buf = alloc(slab * 64)?;
        // Queues: 16-byte counter block + entries.
        let q_a = alloc(16 + slab * 4)?;
        let q_b = alloc(16 + slab * 4)?;
        // Per-pixel luminance sum + sum-of-squares (adaptive stopping).
        let stats_buf = alloc(base.width * base.height * 8)?;
        unsafe {
            std::ptr::write_bytes(
                stats_buf.contents().as_ptr() as *mut u8,
                0,
                base.width * base.height * 8,
            );
        }
        Ok(Self {
            base,
            p_raygen,
            p_extend,
            p_shade,
            paths_buf,
            hits_buf,
            q_a,
            q_b,
            stats_buf,
            slab,
            adaptive_tol: 0.0,
        })
    }

    /// Enable adaptive sampling: pixels stop once their 95% CI relative
    /// error drops below `tol` (checked after 32 samples).
    pub fn set_adaptive(&mut self, tol: f64) {
        self.adaptive_tol = tol as f32;
    }

    /// Average samples actually taken per pixel (from the weight channel).
    pub fn average_spp(&self) -> f64 {
        let data = self.base.accum_slice();
        let n = self.base.width * self.base.height;
        (0..n).map(|i| data[i * 4 + 3] as f64).sum::<f64>() / n as f64
    }

    pub fn width(&self) -> usize {
        self.base.width
    }

    pub fn height(&self) -> usize {
        self.base.height
    }

    fn dispatch_1d(
        &self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        uniforms: &super::gpu_scene::GpuPtUniforms,
        q_in: &Buffer,
        q_in_offset: usize,
        q_out: &Buffer,
        threads: usize,
        with_accum: bool,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        let cmd = self
            .base
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow!("wf command buffer"))?;
        let enc = cmd
            .computeCommandEncoder()
            .ok_or_else(|| anyhow!("wf encoder"))?;
        enc.setComputePipelineState(pipeline);
        unsafe {
            for (i, buf) in self.base.buffers.iter().enumerate() {
                enc.setBuffer_offset_atIndex(Some(buf), 0, i);
            }
            enc.setBytes_length_atIndex(
                NonNull::new(uniforms as *const _ as *mut c_void).unwrap(),
                std::mem::size_of::<super::gpu_scene::GpuPtUniforms>(),
                23,
            );
            enc.setBuffer_offset_atIndex(Some(&self.paths_buf), 0, 24);
            enc.setBuffer_offset_atIndex(Some(&self.hits_buf), 0, 25);
            enc.setBuffer_offset_atIndex(Some(q_in), q_in_offset, 26);
            enc.setBuffer_offset_atIndex(Some(q_out), 0, 27);
            let _ = with_accum;
            enc.setBuffer_offset_atIndex(Some(&self.base.accum_buf), 0, 28);
            enc.setBuffer_offset_atIndex(Some(&self.stats_buf), 0, 29);
        }
        let tgs = MTLSize { width: 256, height: 1, depth: 1 };
        let grid = MTLSize { width: threads, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid, tgs);
        enc.endEncoding();
        cmd.commit();
        Ok(cmd)
    }

    /// Wait for the queue's committed buffers and verify each completed.
    fn drain(cmds: Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>) -> Result<()> {
        for cmd in cmds {
            cmd.waitUntilCompleted();
            if cmd.status() != MTLCommandBufferStatus::Completed {
                let detail = cmd
                    .error()
                    .map(|e| e.localizedDescription().to_string())
                    .unwrap_or_else(|| "no error detail".to_string());
                return Err(anyhow!("wavefront dispatch failed: {detail}"));
            }
        }
        Ok(())
    }

    fn reset_counter(&self, q: &Buffer) {
        unsafe {
            std::ptr::write_bytes(q.contents().as_ptr() as *mut u8, 0, 16);
        }
    }

    fn read_counter(&self, q: &Buffer) -> usize {
        unsafe { *(q.contents().as_ptr() as *const u32) as usize }
    }

    /// Add samples [start, start+count) — same estimator and sampling
    /// streams as the megakernel, wavefront-scheduled.
    pub fn render_samples(&self, start: u32, count: u32) -> Result<()> {
        let total = self.base.width * self.base.height;
        // Chunk dispatches to dodge the GPU watchdog on heavy scenes.
        const CHUNK: usize = 1 << 18;
        for sample in start..start + count {
            let mut slab_start = 0usize;
            while slab_start < total {
                let slab_n = (total - slab_start).min(self.slab);
                let mut uni = self.base.uniforms;
                uni.sample_start = sample;
                uni.wf_slab_base = slab_start as u32;
                uni.adaptive_tol = self.adaptive_tol;
                self.reset_counter(&self.q_a);
                // Raygen fills paths + identity queue (entries at offset
                // 16 unused for wave 0: raygen writes from index 0).
                let mut cmds = Vec::new();
                let mut off = 0usize;
                while off < slab_n {
                    let n = (slab_n - off).min(CHUNK);
                    let mut cu = uni;
                    cu.y_offset = (slab_start + off) as u32;
                    cmds.push(self.dispatch_raygen(&cu, n)?);
                    off += n;
                }
                Self::drain(cmds)?;

                let mut live = self.read_counter(&self.q_a).min(self.slab);
                let mut q_in_is_a = true;
                let mut wave = 0u32;
                while live > 0 && wave < 128 {
                    let (q_in, q_out) = if q_in_is_a {
                        (&self.q_a, &self.q_b)
                    } else {
                        (&self.q_b, &self.q_a)
                    };
                    // Entries always live after the 16-byte counter block
                    // (raygen writes through the same offset binding).
                    let in_off = 16;
                    self.reset_counter(q_out);
                    let mut uni2 = self.base.uniforms;
                    uni2.sample_start = sample;
                    uni2.sample_count = live as u32;
                    uni2.wf_slab_base = slab_start as u32;
                    // Queue every chunk of extend then shade and sync once
                    // per wave: command buffers on one queue run in order,
                    // and shade[i] only needs extend[i]'s hits (both cover
                    // the same queue slice, and extends all precede shades).
                    let mut cmds = Vec::new();
                    let mut off = 0usize;
                    while off < live {
                        let n = (live - off).min(CHUNK);
                        let mut cu = uni2;
                        cu.y_offset = off as u32;
                        cmds.push(self.dispatch_1d(
                            &self.p_extend, &cu, q_in, in_off, q_out, n, false,
                        )?);
                        off += n;
                    }
                    let mut off = 0usize;
                    while off < live {
                        let n = (live - off).min(CHUNK);
                        let mut cu = uni2;
                        cu.y_offset = off as u32;
                        cmds.push(self.dispatch_1d(
                            &self.p_shade, &cu, q_in, in_off, q_out, n, true,
                        )?);
                        off += n;
                    }
                    Self::drain(cmds)?;
                    live = self.read_counter(q_out).min(self.slab);
                    q_in_is_a = !q_in_is_a;
                    wave += 1;
                }
                slab_start += slab_n;
            }
        }
        Ok(())
    }

    fn dispatch_raygen(
        &self,
        uniforms: &super::gpu_scene::GpuPtUniforms,
        n: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        // Raygen pushes through q_a's atomic counter (offset 0).
        self.dispatch_1d(&self.p_raygen, uniforms, &self.q_a, 0, &self.q_b, n, false)
    }

    pub fn image(&self) -> Image {
        self.base.image()
    }
}

fn render_pt_impl(
    gpu: &super::gpu_scene::GpuPtScene,
    spp: u32,
    checkpoint: Option<&std::path::Path>,
) -> Result<(Image, Vec<f32>)> {
    let session = PtSession::from_gpu(gpu)?;

    // Resume from a matching checkpoint.
    let mut sample_start = 0u32;
    if let Some(path) = checkpoint {
        if let Some((done, accum)) =
            load_checkpoint(path, gpu.uniforms.width, gpu.uniforms.height)
        {
            let done = done.min(spp);
            session.load_accum(&accum);
            sample_start = done;
            println!("Resuming from checkpoint: {done}/{spp} samples done");
        }
    }

    let mut sample = sample_start;
    while sample < spp {
        let step = if checkpoint.is_some() {
            CHECKPOINT_EVERY.min(spp - sample)
        } else {
            spp - sample
        };
        session.render_samples(sample, step)?;
        sample += step;
        if let Some(path) = checkpoint {
            save_checkpoint(
                path,
                gpu.uniforms.width,
                gpu.uniforms.height,
                sample,
                session.accum_slice(),
            );
        }
    }

    let image = session.image();
    let aux_owned = session.aux_vec();
    Ok((image, aux_owned))
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
