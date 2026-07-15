# Vship : Fast Metric Computation on GPU

An easy to use high-performance Library for GPU-accelerated visual fidelity
metrics with SSIMULACRA2, Butteraugli & CVVDP.

This project has been 100% made by a **human** (no AI involved).

## Overview

`vship` provides hardware-accelerated implementations of:

- **SSIMULACRA2**: A perceptual image quality metric from Cloudinary's
  SSIMULACRA2
- **Butteraugli**: Google's psychovisual image difference metric from libjxl
- **CVVDP**: University of Cambridge's psychovisual video quality metric

The plugin uses HIP/CUDA or Vulkan for GPU acceleration, providing significant performance
improvements over CPU implementations. It can be used with a simple binary (FFVship), as a vapoursynth plugin and has a C API.

There are precompiled binaries ready to be used in the release section.

/!\ VulkanVship is very new and experimental, it tends to be slower and may be less reliable.

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
Vapoursynth Vship | GPU | 53.750s
FFVship | GPU | 41.297s


`vship` dramatically outperforms CPU-based and GPU-based implementations of these metrics
while preserving a high degree of accuracy.

## References

- Butteraugli Source Code:
  [libjxl/libjxl](https://github.com/libjxl/libjxl/tree/main/lib/jxl/butteraugli)
- SSIMULACRA2 Source Code:
  [cloudinary/ssimulacra2](https://github.com/cloudinary/ssimulacra2)
- CVVDP Source Code: 
  [gfxdisp/ColorVideoVDP](https://github.com/gfxdisp/ColorVideoVDP/tree/main?tab=readme-ov-file)

## Credits

Special thanks to dnjulek for the Zig-based SSIMULACRA2 implementation in
[vszip](https://github.com/dnjulek/vapoursynth-zip).

## License

This project is licensed under the MIT license. License information is provided
by the [LICENSE](LICENSE).
