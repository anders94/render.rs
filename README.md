# render.rs

A high-performance RenderMan-compatible RIB (RenderMan Interface Bytestream) renderer implemented in Rust.

## Features

### Rendering Engine
- **Multi-threaded raytracer** using rayon for parallel pixel rendering
- **Ray-primitive intersection** for Sphere, Cone, and Cylinder
- **Blinn-Phong shading** with ambient, diffuse, and specular components
- **Reflection support** for shiny materials
- **Gamma correction** for accurate color output

### Geometric Primitives
- **Sphere** - `Sphere radius zmin zmax thetamax`
- **Cone** - `Cone height radius thetamax`
- **Cylinder** - `Cylinder radius zmin zmax thetamax`

All primitives support:
- Arbitrary transformations (translate, rotate, scale)
- Partial geometry (zmin/zmax clipping, thetamax < 360)

### Materials
- **Matte** - Pure Lambertian diffuse (rough, non-reflective)
- **Plastic** - Diffuse + specular (moderate shininess)
- **Metal** - High specular, reflective

### Transformations
- Translate, Rotate, Scale
- Hierarchical transform stack
- Matrix-based transformations with proper normal handling

### Output Formats
- **PPM** - Simple text format for debugging
- **PNG** - Compressed format for production

## Performance

- Whitted on Metal: 1080p, 64 objects, 4spp in 0.23s
- Path-traced glade (10k instances, 10B effective triangles) at 720p/96spp
  in 84s on Metal, 178s on CPU
- 79 tests including GPU parity, statistical PT parity, and BVH
  brute-force cross-checks

## Usage

```bash
# Render a RIB file to PNG
cargo run --release -- scene.rib -o output.png -f png

# Render to PPM
cargo run --release -- scene.rib -o output.ppm

# Override resolution
cargo run --release -- scene.rib -w 1920 -h 1080 -o hd.png -f png

# Set thread count
cargo run --release -- scene.rib -t 8 -o output.png -f png

# Render on the Apple GPU (macOS, no build flags needed)
cargo run --release -- scene.rib --backend metal -o output.png -f png
```

## GPU Rendering

Backends share one scene model and are cross-validated by the test suite:

| | CPU (default) | `--backend metal` |
|---|---|---|
| Whitted integrator | rayon, f64 | native MSL megakernel, pixel-exact parity tests |
| Path integrator | f64 reference | f32 megakernel with TLAS/BLAS traversal, statistical parity tests |
| Cornell 300px 512spp (path) | 10.9s | 3.4s |
| Glade: 10k instances / 10B tris, 720p 96spp (path) | 178s | 84s |

The Metal path tracer is a megakernel; incoherent traversal on huge scenes
limits it to a few× CPU until the wavefront refactor (roadmap Phase 9).
The MLX backend was removed in Phase 3 (it was memory-bandwidth-bound by
design; see ROADMAP.md history).

### Metal backend (recommended)

A single Metal compute megakernel — one GPU thread per pixel, with the
object loop, shadow rays, and 5-bounce reflections all in registers. Built
on [objc2-metal](https://crates.io/crates/objc2-metal); always available on
macOS builds with no extra flags or tooling (the kernel source is embedded
and compiled at runtime by the OS). Output is deterministic (no atomics) and
parity-tested against the CPU backend — `cargo test` runs the suite on
macOS.

The GPU computes in f32 (Metal GPUs have no f64): whitted output can
differ from the CPU by about one 8-bit step on silhouette edges; the path
integrator's f32/f64 difference is far below its Monte Carlo noise.

## Example RIB Files

### Minimal Sphere
```rib
Display "test.ppm" "file" "rgb"
Format 320 240 1.0
WorldBegin
    Color 1 0 0
    Sphere 1 -1 1 360
WorldEnd
```

### Multiple Primitives
```rib
Display "scene.ppm" "file" "rgb"
Format 640 480 1.0
Projection "perspective" "fov" [45]

WorldBegin
    # Red sphere
    Color 1 0 0
    Translate -2 0 8
    Sphere 1 -1 1 360

    # Green cylinder
    Color 0 1 0
    Translate 0 0 8
    Cylinder 0.8 -1.5 1.5 360

    # Blue cone
    Color 0 0 1
    Translate 2 0 8
    Rotate -90 1 0 0
    Cone 2 0.8 360
WorldEnd
```

### Materials
```rib
WorldBegin
    # Matte (diffuse)
    Color 1 0.2 0.2
    Surface "matte"
    Sphere 1 -1 1 360

    # Plastic (shiny)
    Color 0.2 1 0.2
    Surface "plastic"
    Translate 2 0 0
    Sphere 1 -1 1 360

    # Metal (reflective)
    Color 0.2 0.2 1
    Surface "metal"
    Translate 4 0 0
    Sphere 1 -1 1 360
WorldEnd
```

## Supported RIB Commands

See **COMPLIANCE.md** for the full request matrix. Policy: every
syntactically valid RIB request is accepted — implemented requests take
effect, state-only requests are recorded, everything else warns once and
is skipped. Highlights: all seven quadrics, Polygon, PointsPolygons
meshes, ObjectInstance instancing, ReadArchive, AreaLightSource,
ConcatTransform/named coordinate systems, Declare/inline declarations.

## Test Scenes

The `tests/fixtures/` directory contains several test scenes:

- `minimal.rib` - Single red sphere
- `three_primitives.rib` - Sphere, cylinder, and cone
- `materials.rib` - Three materials comparison
- `transforms.rib` - Complex scene with 5 objects and transformations

Render them with:
```bash
cargo run --release -- tests/fixtures/transforms.rib -o output.png -f png
```

## Architecture

```
render.rs/
├── src/
│   ├── math/           # Vec3, Point3, Matrix4
│   ├── raytracer/      # Ray, Intersection, Renderer
│   ├── geometry/       # Sphere, Cone, Cylinder
│   ├── scene/          # Camera, Light, Material
│   ├── shading/        # Blinn-Phong shading
│   ├── parser/         # RIB parser (nom-based)
│   └── output/         # PPM and PNG writers
└── tests/
    └── fixtures/       # Example RIB files
```

## Building

```bash
# Debug build
cargo build

# Release build (much faster)
cargo build --release

# Run tests
cargo test

# Run specific scene
cargo run --release -- tests/fixtures/materials.rib -o test.png -f png
```

## Dependencies

- **clap** - CLI argument parsing
- **rayon** - Multi-threaded rendering
- **image** - PNG output
- **nom** - RIB parser
- **anyhow** - Error handling

## Integrators

- `--integrator whitted` (default): direct lighting + hard shadows + mirror
  reflections. Fast, deterministic, what the GPU backends speak.
- `--integrator path`: progressive Monte Carlo **global illumination**
  with a physically-based material system (`Bxdf "PxrSurface"`: Oren-Nayar
  diffuse, GGX specular with VNDF sampling, clearcoat, fuzz, rough glass
  with true refraction, glow, presence cutouts) and physical lights
  (`Light "PxrRectLight" / PxrSphereLight / PxrDiskLight / PxrDistantLight
  / PxrDomeLight` with HDRI importance sampling), plus the legacy
  `AreaLightSource`/`LightSource` forms. NEE + MIS, Russian roulette,
  energy validated by white-furnace tests. `--spp N` controls samples per
  pixel; `-f exr` writes linear HDR. Runs on the CPU (f64 reference) or
  `--backend metal` (f32, statistically identical, faster). See
  `tests/fixtures/cornell.rib` and `tests/fixtures/shaderball.rib`.

## Features (rendering)

- **Shadows** - Shadow rays are cast to every light; occluded lights contribute nothing.
- **Reflections** - Recursive ray tracing (depth 5). Metal reflects strongly and tints by its color; plastic has a subtle clear-coat reflection.
- **Anti-aliasing** - `PixelSamples x y` performs stratified supersampling (e.g. `PixelSamples 2 2` = 4 rays/pixel).
- **Output formats** - Binary PPM (P6, default), ASCII PPM (`--format ppm-ascii`), PNG (`--format png`).

## Known Limitations

1. **Camera** - Fixed at origin looking down +Z (use object/world transforms); perspective and orthographic projections, thin-lens depth of field (`DepthOfField`), motion blur (`MotionBegin`/`Shutter` — transform motion on any instanced geometry plus PointsPolygons deformation), box/triangle/gaussian pixel filters, and adaptive sampling (`--adaptive <tol>` on the CPU path tracer).

2. **Primitives** - Implemented: all seven RiSpec quadrics, Polygon/GeneralPolygon (ear-clipped holes), PointsPolygons meshes with SAH BVH + TLAS instancing, Catmull-Clark SubdivisionMesh with creases, bilinear/bicubic PatchMesh with basis matrices, NURBS NuPatch, fBm displacement at dice time (`Displace "noise"`), Curves (linear/cubic hair as rounded-cone capsule chains, shaded by an energy-conserving Marschner hair BSDF via `Bxdf "PxrMarschnerHair"`), and Points particles. COMPLIANCE.md tracks the full RIB request matrix.

3. **Textures & patterns** - Implemented (roadmap Phase 6): a tiled-mip `.tex` format with a `render txmake` converter, a sharded-LRU tile cache with a byte budget, trilinear filtering driven by ray-cone mip selection, UDIM (`<UDIM>` filenames), and a pattern node graph (PxrTexture/PxrChecker/PxrFractal/PxrMix/PxrColorCorrect/PxrRamp/triplanar) connected to Bxdf params with `reference` declarations — on CPU and Metal (runtime MSL codegen). Not yet: EWA anisotropic filtering, pattern-driven displacement, `Pattern` st on GPU quadrics (use triplanar there).

4. **Acceleration structure** - Quadrics and loose polygons still use a linear scan; meshes (where the polygon counts live) go through a binned-SAH BVH with TLAS instancing.

5. **Global illumination** - The `--integrator path` path tracer (CPU and Metal) does full GI with NEE+MIS; the default Whitted integrator remains the fast direct-light preview. Scenes with many lights (tested to 1200+) are sampled through a light BVH (power/distance importance) on both backends. Long Metal renders can be checkpointed and resumed with `--checkpoint <file>`.

6. **Movie-scale I/O** - Binary RIB reads transparently (mixed ASCII+binary per RISpec Appendix C); `render catrib --binary` converts archives (~45% smaller). `Procedural "DelayedReadArchive"` (eager) and `"RunProgram"` (spawned generators) work; ReadArchive nests both text and binary.

## Future Enhancements

- Implement missing primitives
- Texture mapping support
- BVH acceleration structure
- Progressive rendering
- Motion blur and depth of field
- RenderMan Shading Language (RSL) support

## License

MIT

## Author

Built with Claude Code as a RenderMan renderer reimplementation project.
