# Vship-fast: faster Vulkan metrics that match their references

**Vship-fast, a fork of [Vship](https://codeberg.org/Line-fr/Vship), makes the Vulkan backend 1.75-2.5x faster
than upstream for Butteraugli, 1.6-2.4x for CVVDP and up to 1.6x for SSIMULACRA2, and makes all three match their
reference implementations: Butteraugli bit-identical to libjxl, SSIMULACRA2 to libjxl within 1e-7 points, CVVDP to
pycvvdp to 5 decimals. The speed changes leave the output bit-identical, on NVIDIA and AMD
([details](#how-the-output-is-checked)).**

**Only the Vulkan backend is optimized and changed.** The HIP/CUDA backend is upstream's, with upstream's speed
and scores.

The original Vship README follows this section, without its Credits and Support sections, from
[Vship : Fast Metric Computation on GPU](#vship--fast-metric-computation-on-gpu).

## Speed against upstream

Vulkan backend, RTX 3070 (NVIDIA driver 616.92, Windows 11), frames per second, median of 3 runs alternating the
two builds. Upstream is Vship's `main` at e99aac2, the fork is `perf-metrics` at 1941cbb. Clips: Top Gun Maverick
at 1080p (8 frames) and HoneyBee at 4K (6 frames), 8-bit 4:2:0. Threads: frames computed at once.

| Metric | Threads | 1080p upstream | 1080p fork | | 4K upstream | 4K fork | |
|---|---|---|---|---|---|---|---|
| Butteraugli | 1 | 82.0 | 143.8 | **1.75x** | 21.3 | 38.2 | **1.79x** |
| Butteraugli | 3 | 73.3 | 177.6 | **2.42x** | 18.4 | 46.4 | **2.52x** |
| SSIMULACRA2 | 1 | 273.7 | 274.1 | **1.00x** | 74.3 | 76.3 | **1.03x** |
| SSIMULACRA2 | 3 | 272.8 | 433.6 | **1.59x** | 72.7 | 110.8 | **1.52x** |
| CVVDP | 1 | 119.3 | 202.1 | **1.69x** | 33.4 | 52.4 | **1.57x** |
| CVVDP | 3 | 114.5 | 275.5 | **2.41x** | 29.5 | 65.9 | **2.23x** |

SSIMULACRA2 does more work than upstream: matching libjxl means running libjxl's recursive blur, which is serial
along every row and column. That alone made it 2.8-4.4x slower; the speed work won it back and more.

## Accuracy against the reference implementations

| Metric | Reference | Upstream | Fork |
|---|---|---|---|
| Butteraugli | libjxl 0.12.0 `butteraugli_main` | 1.235% off on average (max norm, worst frame 2.97%) | **bit-identical**: every distortion-map pixel equal |
| SSIMULACRA2 | libjxl 0.12.0 `ssimulacra2` | 0.130 points off on average (worst frame 0.569) | 5e-8 points off on average (worst 1e-7) |
| CVVDP | pycvvdp 0.5.7 | 0.0135 JOD off on average (worst pair 0.0139) | 0.00000 JOD (to 5 decimals) |

Measured for e99aac2 and 1941cbb, both given the same decoded 16-bit RGB frames: 11 frames of three video pairs
(1080p and 4K) for Butteraugli and SSIMULACRA2 (all 35,112,960 Butteraugli distortion-map pixels equal to libjxl's
on an RTX 3070; 6,220,800 of 6,220,800 on 3 frames on a Radeon 8060S), 24 frames of three pairs for CVVDP. The fork's scores
therefore differ from upstream Vship's, on purpose.

Limits: Butteraugli is not bit-identical for images under 8x8 (libjxl pads them) or widths that are not a multiple
of 8 (libjxl computes the last columns of one blur with other arithmetic). On GPUs without 64-bit floats a few
pixels in tens of millions can be 1 ulp off. YUV input with the BT.709, PQ or HLG transfer goes through the GPU's own
`pow()`, so its conversion can differ by a few ulps between GPU vendors, and the scores with it; RGB input converted
with libjxl's arithmetic does not.

## What changed

### Accuracy (scores change on purpose)

- **Butteraugli** computes exactly what libjxl computes: its blurs (kernel sizes, border renormalisation, its 5-tap
  path with mirrored borders), its hf_asymmetry, Malta lines, MaximumClamp and per-pixel arithmetic, its FastLog2f,
  and each operation rounded as libjxl's compiled code rounds it. libjxl's double-precision steps run in double-float
  arithmetic, or in integers, with a fallback to double for the rare undecided values.
- **SSIMULACRA2** computes like libjxl: its recursive Gaussian blur (instead of a 17-tap blur renormalised at the
  borders), the variance from blurred squares, a true cube root and libjxl's XYB constants, every map with all its
  weights, libjxl's rounding (unfused products, correctly rounded division) and the final score in double.
- **CVVDP** computes like pycvvdp in six places: the temporal filters, the masking model's blur and `safe_pow`, the
  Laplacian pyramid's borders, the baseband's CSF frequency, the pooling, and the sRGB-to-XYZ matrix.
- **Input conversion** for SSIMULACRA2 and Butteraugli uses libjxl's arithmetic.
- **Rounding:** every float operation of every shader is marked NoContraction when the library is built, so no GPU
  compiler fuses multiply-adds on its own. The fused multiply-adds a metric needs are explicit. This costs 1-3% on
  an RTX 3070 and nothing measurable on a Radeon 8060S.

### Speed (output unchanged)

- **All metrics:** one queue per GPU for all its work (1.20-1.23x with 3 threads); uploads on a transfer queue;
  the chroma is upsampled inside the YUV-to-RGB step, which reads integer luma straight from the upload. For
  SSIMULACRA2 and CVVDP that step is part of the metric's first kernel.
- **SSIMULACRA2:** the score kernels spread libjxl's serial blur over the whole GPU and compute the error maps as
  each pixel is blurred (3.4x at 1080p); one kernel turns the YUV input into all 6 scales.
- **Butteraugli:** every blur runs as one fused, tiled kernel per kernel size (1.22-1.27x on the RTX 3070, up to
  1.87x on the Radeon 8060S); the opsin dynamics and its blurs are one kernel; the L2 differences of the frequency bands are one pass; the
  operations that followed several blurs are folded into them.
- **CVVDP:** the display model and colour conversion are one kernel; the distortion map is computed only for frames
  that ask for it; the pooling runs as one batch; the pyramid-to-CSF kernel handles all 4 channels of a tile at
  once, and it and the masking kernel issue their tile loads together.

Each commit message gives its own speedup and how it was checked.

### Bug fixes

- CVVDP's distortion map was rebuilt with the band sizes in reverse order, so the maps were wrong (not the scores).
- Resize with crop used overlapping scratch buffers, so its results could change from run to run.
- Butteraugli's fuzzy erosion read out of bounds on images under 3 pixels wide or tall.
- Reported upstream and fixed there: the Butteraugli B channel computed from the wrong plane
  ([#22](https://codeberg.org/Line-fr/Vship/pulls/22)), a distortion-map overflow with padded strides
  ([#26](https://codeberg.org/Line-fr/Vship/issues/26)) and the out-of-bounds read above
  ([#25](https://codeberg.org/Line-fr/Vship/issues/25)), all included here; failing pipelines on macOS
  ([#27](https://codeberg.org/Line-fr/Vship/issues/27)), fixed upstream after e99aac2 and not included here.

## How the output is checked

Each speed commit was compared with the commit before it on the RTX 3070: every frame's scores as exact hex floats
and a hash of each distortion map, for all three metrics at 1080p and 4K. Later commits were also checked on a sweep
of 27 input formats (bit depths, ranges, chroma layouts, odd sizes, matrices, transfer functions, crop and resize),
and the last ones with padded input lines. All were identical. On a Radeon 8060S, the accuracy baseline before any
speed work and the last speed commit give identical output, apart from the bug fixes above, once both are built
with NoContraction (before it, four early commits rounded slightly differently on AMD). The Butteraugli exactness
work that followed was checked pixel by pixel against libjxl on both GPUs. The tools are in `perf-tools/` on the
`perf-test-tools` branch.

## Branches

- `perf-metrics` (default): the fork's main line, based on upstream's e99aac2. Later upstream commits, including
  upstream's own versions of some of these fixes, are not merged.
- `perf-test-tools`: `perf-metrics` plus the speed and output-check tools.
- `main`: upstream Vship at e99aac2.
- `amd-opt*`, `exact-*`: work in progress, tested on AMD and NVIDIA before being merged.

## Building and using

As for upstream Vship: see the [original README](#vship--fast-metric-computation-on-gpu) below and the
[docs](doc/). The C API is unchanged. On GPUs with 64-bit floats Vship now enables them (shaderFloat64) for
Butteraugli's rare double-precision fallback.

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
