# Roadmap: render.rs → a modern-PRMan-class renderer

The goal: evolve render.rs into a renderer in modern PRMan's class — full RIB
scene-description compliance, physically-based path tracing at RIS/XPU
quality, movie-order scene complexity. Every phase ends with a demo image
that was impossible before.

**Locked decisions**
- *Modern target*: no RSL/REYES (PRMan 27 itself is XPU path tracing; RIS is
  in maintenance). OSL for programmable shading, staged via C++ FFI.
- *Backends*: CPU is the always-correct reference; Metal is the
  production/performance hero. MLX is frozen and will be deleted in Phase 3.
- *USD/Hydra*: late-phase stretch goal; RIB is the native format throughout.
- *Compliance philosophy*: "fully compliant" means **accepting** every RIB
  request. COMPLIANCE.md tracks each as implemented / planned /
  skipped-forever (CSG, Blobby, LoD are parse-and-warn).

## Phases

| # | Theme | Key contents | Demo gate | Backends |
|---|---|---|---|---|
| P0 | Speak RIB properly | Generalized request/typed-param AST, Declare, ConcatTransform/CoordinateSystem, Orientation/Sides/Basis, Torus/Disk/Paraboloid/Hyperboloid, simple Polygon, option/attribute passthrough, parse-and-warn | Quadric zoo; classic PRMan RIBs parse | CPU+Metal |
| P1 | **The path-tracing pivot** | Progressive MC integrator, f32 render core + HDR film, NEE+MIS, Russian roulette, samplers, rect area lights, EXR out, linear pipeline | Cornell box with GI | CPU only |
| P2 | Triangles at scale | PointsPolygons, watertight ray-tri, SAH BVH (BLAS/TLAS), ObjectInstance instancing, ReadArchive, CompiledScene (#[repr(C)], grown from flatten.rs) | Instanced dragon glade (1M tris × 10k instances) | CPU only |
| P3 | Metal catches up | PT+BVH in iterative Metal megakernel, byte-shared CompiledScene, statistical parity harness, delete MLX | Dragon glade at 4K in minutes | CPU+Metal |
| P4 | PBR materials + physical lights | Bxdf trait, GGX/VNDF, conductor/dielectric fresnel, PxrSurface-lite über-material, `Bxdf`/`Light` requests, dome light + HDRI importance sampling, furnace + chi-square validation tier | Shaderball lineup under HDRI | CPU+Metal |
| P5 | Subdiv + displacement | PatchMesh, Catmull-Clark w/ creases (uniform first; OpenSubdiv-FFI plan B), NuPatch, true displacement at dice time | Displaced creature close-up | CPU+Metal |
| P6 | Textures & patterns | Tiled-mip .tex + txmake tool, LRU tile cache w/ byte budget, EWA, UDIM, ray differentials, Rust pattern node graph (GPU via MSL codegen) | UDIM still life, no shimmer at 16 spp | CPU+Metal |
| P7 | Camera & motion | MotionBegin/End (xform+deform), time-sampled BVH, shutter, thin-lens DoF, pixel filters w/ importance sampling, adaptive sampling | Motion-blurred pan with bokeh | CPU+Metal |
| P8 | Hair & curves | Curves (linear/cubic, ribbon/round), curve BVH, Marschner hair BSDF, Points | Furry teapot / 500k-strand groom | CPU+Metal |
| P9 | Movie scale | Metal wavefront refactor (measured triggers), light BVH many-light sampling, binary RIB, DelayedReadArchive, Procedural, buckets + checkpointing, stress-scene generators, fuzzing | Forest flythrough: 10M+ instances, 1M+ curves, 1000+ lights, 4K | CPU ref + Metal wavefront |
| P10 | Volumes + SSS | Null-scattering heterogeneous volumes, HG phase, volume MIS, NanoVDB, Burley then random-walk SSS | VDB cloudscape + skin bust | CPU→Metal |
| P11 | Programmable shading + pipeline | OSL (C++ FFI, CPU, feature-flag), LPE engine, AOVs, cryptomatte, multilayer EXR, OIDN denoise, interactive preview, ACES | Production shot compositing in Nuke | CPU (+GPU native graph) |
| P12 | Stretch | USD/Hydra delegate, deep EXR, ptex, Lama layering, path guiding, distributed rendering | — | — |

Dependency spine: P0→P1→P2→P3→{P4→P5, P6, P8}→P7→P9→{P10, P11}→P12.

## Architectural decisions

1. **f32 render core, f64 scene composition** (flip during P1); camera-relative positioning + ulp ray offsets, not magic epsilons.
2. **RGB radiance behind a `Spectrum` newtype** — hero-wavelength spectral remains a possible later swap.
3. **Megakernel until P9's measured wavefront triggers** (shade-switch dominance, texture-miss queueing, volumes); megakernel stays selectable after.
4. **OSL staged with an off-ramp**: Rust pattern nodes are the committed workhorse; C++ OSL FFI is a feature flag; a Rust OSL interpreter is explicitly out.
5. **One CompiledScene**: #[repr(C)]/Pod buffers traversed by CPU and uploaded byte-identical to Metal; layout locked by size/offset asserts on both sides.
6. **Validation as a test tier**: furnace tests, chi-square sampler tests, energy asserts, PBRT cross-renders, statistical CPU/GPU parity.
7. **Own tiled `.tex` format** + sharded LRU cache with a hard byte budget.
8. **Denoise via OIDN FFI** on the host-side film.

## Movie-scale acceptance targets (P9–P11, Max/Ultra-class Mac)

≥100M resident post-tess triangles · ≥10M TLAS instances (≥10B effective
prims) · ≥1M hair strands · ≥500M active VDB voxels · ≥1,000 lights via
light BVH at <2× noise penalty · ≥100GB texture set from a ≤8GB cache at
<15% overhead · 4K at ≤256 avg spp + denoise · ≤60 min/frame Metal ·
≤2s to first pixel interactive · ≤90s scene build · ≤64GB peak RSS ·
furnace within 0.5% · CPU/GPU RMSE ≤1%.

## Top risks

P1 pivot stalling (mitigate: keep `--integrator whitted` until P3; Cornell
box is the merge gate) · OSL rabbit hole (staged, never blocks) · subdiv
crack correctness (uniform first, OpenSubdiv plan B) · wavefront complexity
(measured triggers, fallback path) · texture-cache thrash (budget + stats
from day one) · f32 precision at scale (camera-relative early) · RIB arcana
scope creep (parse-and-warn IS compliance) · parity-harness ossification
(deliberate P3 migration to statistical parity).
