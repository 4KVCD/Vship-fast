# Vship, 4KVCD fork: faster Vulkan metrics that match their references

**This fork of [Vship](https://codeberg.org/Line-fr/Vship) makes the Vulkan backend 1.5-2.2x faster than upstream
for Butteraugli, 1.6-2.5x for CVVDP and up to 1.6x for SSIMULACRA2, and makes SSIMULACRA2 match libjxl and CVVDP
match pycvvdp. The speed changes leave the output bit-identical on NVIDIA ([details](#how-the-output-is-checked)).**

**Only the Vulkan backend is optimized and changed.** The HIP/CUDA backend is upstream's, with upstream's speed
and scores.

The original Vship README follows this section, without its Credits and Support sections, from
[Vship : Fast Metric Computation on GPU](#vship--fast-metric-computation-on-gpu).

## Speed against upstream

Vulkan backend, RTX 3070 (NVIDIA driver 616.92, Windows 11), frames per second, median of 3 runs alternating the
two builds. Upstream is Vship's `main` at e99aac2, the fork is `perf-metrics` at 587194c. Clips: Top Gun Maverick
at 1080p (8 frames) and HoneyBee at 4K (6 frames), 8-bit 4:2:0. Threads: frames computed at once.

| Metric | Threads | 1080p upstream | 1080p fork | | 4K upstream | 4K fork | |
|---|---|---|---|---|---|---|---|
| Butteraugli | 1 | 80.5 | 123.1 | **1.53x** | 21.0 | 34.8 | **1.66x** |
| Butteraugli | 3 | 72.0 | 155.9 | **2.17x** | 18.4 | 39.9 | **2.17x** |
| SSIMULACRA2 | 1 | 270.0 | 285.8 | **1.06x** | 74.7 | 76.9 | **1.03x** |
| SSIMULACRA2 | 3 | 278.7 | 438.7 | **1.57x** | 72.6 | 111.8 | **1.54x** |
| CVVDP | 1 | 117.1 | 199.4 | **1.70x** | 33.0 | 53.9 | **1.63x** |
| CVVDP | 3 | 110.8 | 274.6 | **2.48x** | 29.1 | 65.3 | **2.24x** |

SSIMULACRA2 does more work than upstream: matching libjxl means running libjxl's recursive blur, which is serial
along every row and column. That alone made it 2.8-4.4x slower; the speed work won it back and more.

## Accuracy against the reference implementations

| Metric | Reference | Upstream | Fork |
|---|---|---|---|
| SSIMULACRA2 | libjxl 0.12.0 `ssimulacra2` | 0.130 points off on average (worst frame 0.569) | 0.0000 points (to 4 decimals) |
| CVVDP | pycvvdp 0.5.7 | 0.0135 JOD off on average (worst pair 0.0139) | 0.00000 JOD (to 5 decimals) |
| Butteraugli | libjxl 0.12.0 `butteraugli_main` | 1.235% off on average (max norm, worst frame 2.97%) | unchanged |

Measured on the RTX 3070 for e99aac2 and 587194c, both given the same decoded frames: 11 frames of three video pairs
(1080p and 4K) for SSIMULACRA2 and Butteraugli, 24 frames of three pairs for CVVDP. SSIMULACRA2 and CVVDP scores
therefore differ from upstream Vship's, on purpose. On a Radeon 8060S, SSIMULACRA2 measured about 0.007 points from
libjxl, because AMD's shader compiler fuses multiply-adds that libjxl rounds separately.

Butteraugli keeps upstream's definition. The fixes that bring it to within 0.001% of libjxl are upstream pull
requests [#28](https://codeberg.org/Line-fr/Vship/pulls/28)-[#31](https://codeberg.org/Line-fr/Vship/pulls/31),
not merged here.

## What changed

### Accuracy (scores change on purpose)

- **SSIMULACRA2** computes like libjxl: its recursive Gaussian blur (instead of a 17-tap blur renormalised at the
  borders), the variance from blurred squares, a true cube root and libjxl's XYB constants, every map with all its
  weights, libjxl's rounding (unfused products, correctly rounded division) and the final score in double.
- **CVVDP** computes like pycvvdp in six places: the temporal filters, the masking model's blur and `safe_pow`, the
  Laplacian pyramid's borders, the baseband's CSF frequency, the pooling, and the sRGB-to-XYZ matrix.
- **Input conversion** for SSIMULACRA2 and Butteraugli uses libjxl's arithmetic.

### Speed (output unchanged)

- **All metrics:** one queue per GPU for all its work (1.20-1.23x with 3 threads); uploads on a transfer queue;
  the chroma is upsampled inside the YUV-to-RGB step, which reads integer luma straight from the upload. For
  SSIMULACRA2 and CVVDP that step is part of the metric's first kernel.
- **SSIMULACRA2:** the score kernels spread libjxl's serial blur over the whole GPU and compute the error maps as
  each pixel is blurred (3.4x at 1080p); one kernel turns the YUV input into all 6 scales.
- **Butteraugli:** the blurs of windows 3, 4 and 18 run in one fused, tiled kernel (1.22-1.27x on the RTX 3070,
  up to 1.87x on the Radeon 8060S); the opsin dynamics and its blurs are one kernel; most of the L2 differences
  are one pass; the operations that followed several blurs are folded into them.
- **CVVDP:** the display model and colour conversion are one kernel; the distortion map is computed only for frames
  that ask for it; the pooling runs as one batch; the pyramid-to-CSF kernel handles all 4 channels of a tile at
  once, and it and the masking kernel issue their tile loads together.

Each commit message gives its own speedup and how it was checked.

### Bug fixes

- CVVDP's distortion map was rebuilt with the band sizes in reverse order, so the maps were wrong (not the scores).
- Resize with crop used overlapping scratch buffers, so its results could change from run to run.
- Reported upstream and fixed there: the Butteraugli B channel computed from the wrong plane
  ([#22](https://codeberg.org/Line-fr/Vship/pulls/22)) and a distortion-map overflow with padded strides
  ([#26](https://codeberg.org/Line-fr/Vship/issues/26)), both included here; an out-of-bounds read on tiny images
  ([#25](https://codeberg.org/Line-fr/Vship/issues/25)) and failing pipelines on macOS
  ([#27](https://codeberg.org/Line-fr/Vship/issues/27)), fixed upstream after e99aac2 and not included here.

## How the output is checked

Each speed commit was compared with the commit before it on the RTX 3070: every frame's scores as exact hex floats
and a hash of each distortion map, for all three metrics at 1080p and 4K. Later commits were also checked on a sweep
of 27 input formats (bit depths, ranges, chroma layouts, odd sizes, matrices, transfer functions, crop and resize),
and the last ones with padded input lines. All were identical. From Butteraugli's fused blur onwards, each commit
was also identical on a Radeon 8060S; four earlier commits round slightly differently there, because AMD's
compiler fuses multiply-adds that NVIDIA's does not. The tools are in `perf-tools/` on the `perf-test-tools` branch.

## Branches

- `perf-metrics` (default): the fork's main line, based on upstream's e99aac2. Later upstream commits, including
  upstream's own versions of some of these fixes, are not merged.
- `perf-test-tools`: `perf-metrics` plus the speed and exactness tools.
- `main`: upstream Vship at e99aac2.
- `amd-opt*`: work in progress, tested on AMD and NVIDIA before being merged.

## Building and using

As for upstream Vship: see the [original README](#vship--fast-metric-computation-on-gpu) below and the
[docs](doc/). The C API is unchanged.

The work was done by 4KVCD with [Claude Code](https://claude.com/claude-code). The fork keeps Vship's
[license](LICENSE), which forbids using the code to train machine learning models.

---

# Vship : Fast Metric Computation on GPU

An easy to use high-performance Library for GPU-accelerated visual fidelity
metrics with SSIMULACRA2, Butteraugli & CVVDP.

## Overview

`vship` provides hardware-accelerated implementations of:

- **SSIMULACRA2**: A perceptual image quality metric from Cloudinary's
  SSIMULACRA2
- **Butteraugli**: Google's psychovisual image difference metric from libjxl
- **CVVDP**: University of Cambridge's psychovisual video quality metric

The plugin uses HIP/CUDA or Vulkan for GPU acceleration, providing significant performance
improvements over CPU implementations. It can be used with a simple binary (FFVship), as a vapoursynth plugin and has a C API.

There are precompiled binaries ready to be used in the release section.

## Projects Featuring Vship

If you want to use Vship with a pre-defined workflow, here are some projects
featuring Vship:

- [NVEnc](https://github.com/rigaya/NVEnc), [QSVEnc](https://github.com/rigaya/QSVEnc) : Frameworks to test HW encoding quality from [Rigaya](https://github.com/rigaya)
- [Av1an](https://github.com/rust-av/Av1an): A Cross-platform command-line AV1 / VP9 / HEVC / H264 encoding framework with per scene quality encoding 
- [xav](https://github.com/emrakyz/xav): A simpler alternative to Av1an dedicated to Target Quality Encoding trying to be as fast as possible.
- [Auto-boost-essential](https://github.com/nekotrix/auto-boost-algorithm) : A python script improving efficiency of video encoding by varying parameters on scenes automatically.
- [SSIMULACRApy](https://codeberg.org/Kosaka/ssimulacrapy): A Python script to
  compare videos and output their SSIMU2 scores using various metrics (by
  [Kosaka](https://codeberg.org/Kosaka))
- [`metrics`](https://github.com/psy-ex/metrics): A perceptual video metrics
  toolkit for video encoder developers (by the
  [Psychovisual Experts Group](https://github.com/psy-ex/metrics))
- [`vs_align`](https://github.com/pifroggi/vs_align): A vapoursynth plugin
  measuring temporal offset between two videos
- [`chunknorris`](https://github.com/Boulder08/chunknorris): A python script
  to adjust the quality encoding parameters at each scene of a video base on objective metrics

## Installation

The steps to build `vship` from source are provided below.
See [Compilation and Usage](./doc/VshipAPI.md#compilation-and-usage) for more details.
For compiling on Windows, see [FFVship Windows Compilation](./doc/FFVship-Windows-Compilation.md).

### Dependencies
For all build options the following are required:

- `make`

To build the HIP/CUDA version

- `hipcc` (AMD HIP SDK) or `nvcc` (NVIDIA CUDA SDK)

To build the Vulkan version

- Vulkan SDK (vulkan header + lib) (ideally [LunarG](https://vulkan.lunarg.com/sdk/home))
- `slangc` if rebuilding spv code is intended

Additionally, to build the FFvship cli tool:

- [ffms2](https://github.com/FFMS/ffms2) (and libavutil header to compile)
- [pkg-config](https://gitlab.freedesktop.org/pkg-config/pkg-config)

### Build Instructions

Windows users wishing to compile FFVship by themselves may first refer to [this](doc/FFVship-Windows-Compilation.md)

1. Use the appropriate target for your gpu or use case.

```bash
#libvship Build
make build BACKEND=Cuda              # Build for the current systems Nvidia gpu
make build BACKEND=Cuda ARCH=ALL     # Build for all supported Nvidia gpus
make build BACKEND=HIP               # Build for the current systems AMD gpu
make build BACKEND=HIP ARCH=ALL   # Build for all supported AMD gpus

make build MITIGATE_MALLOC_ASYNC=on #build for AMD gpu and add a workaround to current amd driver issues if you have them (will be slower than a normal build)

make shaderBuild   # Force shader rebuild, requires slangc (optional)
make build BACKEND=Vulkan   # Build release vulkan vship
make build BACKEND=Vulkan DEBUG=1   # Build Vulkan Vship with debugging symbols and validation layers

#FFVship CLI linux tool build (requires libvship built before FFVSHIP)
make buildFFVSHIP
```

2. Install libvship and eventually the FFVship executable.

The `install` target automatically detects and installs only the components that were built.
If libvship is built, it will also create a pkgconfig file and install it directly.
```bash
make install
#for arch/fedora, you need to use another prefix:
make install PREFIX=/usr
```

If pkgconfig is installed, you should be able to do: `pkg-config --modversion libvship` to get vship version.

## Library Usage

### FFVship

To control the performance-to-VRAM trade-off, set the `-g` argument in FFVship to control the
number of GPU threads to use. You can also control the number of decoder threads with `-t`.
I recommend 4 GPU threads as a good compromise between performance and VRAM usage.

This contains only some of the numerous options offered by Vship
I recommend checking the doc or using -h to get the full list.
```
usage: ./FFVship [-h] [--source SOURCE] [--encoded ENCODED]
                    [-m {SSIMULACRA2, Butteraugli, CVVDP}]
                    [--start start] [--end end] [-e --every every]
                    [-t THREADS] [-g gpuThreads] [--gpu-id gpu_id]
                    [--json OUTPUT]
                    [--list-gpu]
```

FFVship allows to get image output for your metric computations like this:

<img src="/attachments/21abb35e-6c15-4631-8164-0de5ee79627d" alt="drawing" width="200"/>

more details may be found in the [doc](doc/FFVship.md)

### Vapoursynth

You can check examples of scripts [here](examples/vapoursynthExample.py) or the doc here [this](doc/Home.md)

### Vship C API

If you intend to use vship C api for your application, I recommend reading the doc and more specifically [this](doc/VshipAPI.md)

## Performance

Testing Hardware: Ryzen 7940HS + RTX 4050 Mobile (strong CPU - weak GPU configuration)  
Testing Clip: 1080p 1339 frames

SSIMU2 Implementation | HW Type | Time
--- | --- | ---
JXL | CPU | 115s
ssimulacra2_rs | CPU | X s
fmetrics | CPU | 38.084s
VSZIP | CPU | 29.978s
--- **CUDA** --- | --- | ---
Vapoursynth Vship | GPU | 10.351s
FFVship | GPU | 7.146s
--- **Vulkan** --- | --- | ---
Vapoursynth Vship | GPU | 16.046s
FFVship | GPU | 10.085s

Butteraugli Implementation | HW Type | Time
--- | --- | ---
fmetrics | CPU | 275s
JXL | CPU | 239s
--- **CUDA** --- | --- | ---
Vapoursynth Vship | GPU | 39.528s
FFVship | GPU | 38.710s
--- **Vulkan** --- | --- | ---
Vapoursynth Vship | GPU | 47.961s
FFVship | GPU | 41.484s

CVVDP Implementation | HW Type | Time
--- | --- | ---
Original Repo | CPU | 1632s
FFVship (Vulkan llvmPipe) | CPU | 499s
fmetrics | CPU | 221s
Original Repo | GPU | 162s
--- **CUDA** --- | ---
Vapoursynth Vship | GPU | 28.558s
FFVship | GPU | 16.184s
--- **Vulkan** --- | ---
Vapoursynth Vship | GPU | 45.402s
FFVship | GPU | 25.922s


`vship` dramatically outperforms CPU-based and GPU-based implementations of these metrics
while preserving a high degree of accuracy.

## References

- Butteraugli Source Code:
  [libjxl/libjxl](https://github.com/libjxl/libjxl/tree/main/lib/jxl/butteraugli)
- SSIMULACRA2 Source Code:
  [cloudinary/ssimulacra2](https://github.com/cloudinary/ssimulacra2)
- CVVDP Source Code: 
  [gfxdisp/ColorVideoVDP](https://github.com/gfxdisp/ColorVideoVDP/tree/main?tab=readme-ov-file)

## License

This project is licensed under the MIT license. License information is provided
by the [LICENSE](LICENSE).
