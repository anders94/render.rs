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

const KERNEL_SRC: &str = include_str!("kernel.metal");
/// GPU-watchdog insurance: bounded work per command buffer.
const ROWS_PER_BAND: usize = 256;

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
        .newLibraryWithSource_options_error(&NSString::from_str(KERNEL_SRC), Some(&options))
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
            .newLibraryWithSource_options_error(&NSString::from_str(KERNEL_SRC), Some(&options))
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
