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

- **800x600 scene with 5 objects**: 0.13s (multi-threaded)
- **29/29 unit tests passing**
- **~2,110 lines of Rust code**

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

Three backends share the same scene model and shading math:

| Backend | How | 1080p 4spp, 5 objects | 1080p 4spp, 64 objects | 4K 4spp, 64 objects |
|---|---|---|---|---|
| `--backend cpu` (default) | rayon, f64 | 0.84s | 1.15s | 4.55s |
| `--backend metal` **(recommended GPU)** | native MSL compute kernel | **0.13s** | **0.23s** | **0.74s** |
| `--backend mlx` | MLX array ops (wavefront) | 5.0s | 36s | — |

*(Apple Silicon Mac; wall time including scene parse, image write, and for
Metal the one-time ~100ms runtime shader compile)*

### Metal backend (recommended)

A single Metal compute megakernel — one GPU thread per pixel, with the
object loop, shadow rays, and 5-bounce reflections all in registers. Built
on [objc2-metal](https://crates.io/crates/objc2-metal); always available on
macOS builds with no extra flags or tooling (the kernel source is embedded
and compiled at runtime by the OS). Output is deterministic (no atomics) and
parity-tested against the CPU backend — `cargo test` runs the suite on
macOS.

### MLX backend (experimental, slower)

Built on [mlx-rs](https://github.com/oxideai/mlx-rs); requires
`cargo build --release --features mlx` (compiles MLX's C++ core — needs
`cmake` and the Metal toolchain: `xcodebuild -downloadComponent MetalToolchain`).
Kept as a correct, parity-tested reference (`cargo test --features mlx`),
but MLX's array model materializes an intermediate array per op per object,
so it is memory-bandwidth-bound and slower than the CPU; mlx-rs exposes no
custom-kernel API, and MLX's `compile` fusion fails on graphs this deep.

Both GPU backends compute in f32 (Metal GPUs have no f64), so output can
differ from the CPU backend by about one 8-bit quantization step on
silhouette edges.

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

| Command | Status | Description |
|---------|--------|-------------|
| Display | ✅ | Output file specification |
| Format | ✅ | Image resolution |
| Projection | ✅ | Camera projection (perspective) |
| WorldBegin/End | ✅ | Scene definition |
| Color | ✅ | RGB color |
| Surface | ✅ | Material type |
| Translate | ✅ | Translation transform |
| Rotate | ✅ | Rotation transform |
| Scale | ✅ | Scaling transform |
| Sphere | ✅ | Sphere primitive |
| Cone | ✅ | Cone primitive |
| Cylinder | ✅ | Cylinder primitive |
| AttributeBegin/End | ⚠️ | Parsing issue (use direct commands) |
| LightSource | ⚠️ | Parameter parsing issue |

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
- `--integrator path` (CPU): progressive Monte Carlo **global illumination**
  — soft shadows, color bleeding, area lights (via `AreaLightSource` on quad
  polygons), physically-based lobes (Lambert / GGX), next-event estimation
  with multiple importance sampling, Russian roulette. `--spp N` controls
  samples per pixel. Use `-f exr` for linear HDR output. See
  `tests/fixtures/cornell.rib`.

## Features (rendering)

- **Shadows** - Shadow rays are cast to every light; occluded lights contribute nothing.
- **Reflections** - Recursive ray tracing (depth 5). Metal reflects strongly and tints by its color; plastic has a subtle clear-coat reflection.
- **Anti-aliasing** - `PixelSamples x y` performs stratified supersampling (e.g. `PixelSamples 2 2` = 4 rays/pixel).
- **Output formats** - Binary PPM (P6, default), ASCII PPM (`--format ppm-ascii`), PNG (`--format png`).

## Known Limitations

1. **Camera positioning** - Camera is currently fixed at origin looking down +Z axis. Use object translations instead.

2. **Missing primitives** - Subdivision surfaces, NURBS, curves, and points are on the roadmap (see ROADMAP.md); all seven RiSpec quadrics, Polygons, and PointsPolygons triangle meshes (with SAH BVH + TLAS instancing via ObjectInstance — billions of effective triangles) are implemented. COMPLIANCE.md tracks the full RIB request matrix. Mesh geometry is CPU-only until roadmap Phase 3.

3. **No textures** - Texture mapping not implemented.

4. **No acceleration structure** - Every ray tests every object; fine for small scenes, slow for large ones.

5. **No global illumination** - Direct lighting plus mirror reflections only; no ambient occlusion or path tracing. Point lights have no distance falloff.

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
