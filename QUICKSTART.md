# render.rs Quick Start Guide

## Installation

```bash
cd render.rs
cargo build --release
```

## Basic Usage

```bash
# Render a scene to PNG
./target/release/render scene.rib -o output.png -f png

# Render to PPM (faster, larger files)
./target/release/render scene.rib -o output.ppm

# Custom resolution
./target/release/render scene.rib -w 1920 -h 1080 -o hd.png -f png

# Set thread count
./target/release/render scene.rib -t 4 -o output.png -f png
```

## Try the Example Scenes

```bash
# Simple red sphere (fastest)
./target/release/render tests/fixtures/minimal.rib -o test1.png -f png

# Three colored primitives
./target/release/render tests/fixtures/three_primitives.rib -o test2.png -f png

# Materials comparison (matte, plastic, metal)
./target/release/render tests/fixtures/materials.rib -o test3.png -f png

# Complex scene with transformations
./target/release/render tests/fixtures/transforms.rib -o test4.png -f png

# Showcase - 9 objects with all features
./target/release/render tests/fixtures/showcase.rib -o showcase.png -f png
```

## Create Your Own Scene

Create `my_scene.rib`:

```rib
Display "my_render.ppm" "file" "rgb"
Format 640 480 1.0
Projection "perspective" "fov" [50]

WorldBegin
    # Red sphere
    Color 1 0 0
    Surface "plastic"
    Translate -2 0 6
    Sphere 1 -1 1 360

    # Green cylinder
    Color 0 1 0
    Surface "matte"
    Translate 0 0 6
    Cylinder 0.8 -1.5 1.5 360

    # Blue cone
    Color 0 0 1
    Surface "metal"
    Translate 2 0 6
    Rotate -90 1 0 0
    Cone 2 0.8 360
WorldEnd
```

Then render it:

```bash
./target/release/render my_scene.rib -o my_render.png -f png
```

## RIB Command Reference

### Scene Setup
- `Display "file.ppm" "file" "rgb"` - Output file
- `Format width height aspect` - Resolution (e.g., 640 480 1.0)
- `Projection "perspective" "fov" [angle]` - Camera FOV

### Scene Content
- `WorldBegin` / `WorldEnd` - Scene boundaries
- `Color r g b` - Set color (0.0-1.0 range)
- `Surface "type"` - Material: "matte", "plastic", or "metal"

### Transformations
- `Translate x y z` - Move object
- `Rotate angle x y z` - Rotate around axis
- `Scale x y z` - Scale object

### Primitives
- `Sphere radius zmin zmax thetamax`
- `Cylinder radius zmin zmax thetamax`
- `Cone height radius thetamax`

## Performance Tips

1. **Use release mode**: `cargo run --release` is 10-100x faster than debug
2. **Adjust threads**: Use `-t` flag to match your CPU cores
3. **Start small**: Test at 320x240, then increase resolution
4. **PPM for speed**: PPM output is faster than PNG during development
5. **Object count**: Each object adds ~2-5ms render time at 640x480

## Troubleshooting

**Black image?**
- Make sure objects are at positive Z (e.g., 5-10)
- Camera looks down +Z axis from origin
- Check color values are 0.0-1.0 range

**Parse error?**
- Don't use AttributeBegin/End blocks (known issue)
- Don't use LightSource parameters (known issue)
- Keep scene structure flat for now

**Slow rendering?**
- Use `--release` flag
- Reduce resolution for testing
- Decrease object count

## What's Working

✅ Sphere, Cone, Cylinder primitives
✅ Matte, Plastic, Metal materials
✅ Translate, Rotate, Scale transforms
✅ Multi-threaded rendering
✅ PNG and PPM output
✅ Custom resolutions

## What's Not Working Yet

⚠️ AttributeBegin/End with nested objects
⚠️ LightSource parameters
⚠️ Texture mapping
⚠️ Shadows

## Get Help

```bash
./target/release/render --help
```

## Examples Output

All example renders are in the `renders/` directory after running the tests.
