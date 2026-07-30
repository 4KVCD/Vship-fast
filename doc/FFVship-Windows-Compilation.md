## Introduction

Compiling FFVship on windows has been relatively complex due to lib placement not being known to compiler. As such, I upload below a dev kit containing pre compiled dependencies.

[Windows_devkit.zip](https://codeberg.org/Line-fr/Vship/releases/download/v4.0.0/Windows_devkit.zip)

## Steps

- Unzip the devkit to get a folder containing various files

- Copy these files and dump them in vship folder

- build libvship using `make build BACKEND=<your backend>` (follow the readme)

- build FFVship using `make buildFFVSHIP`

The following binary is going to need to be used next to libvship.dll and ffms2.dll (can be found either in the ffms2 repository or stolen from a pre-compiled FFVship release).

## static libvship into FFVship (new, for vship v5.1.1)

if you wish to avoid having a "libvship.dll" next to your newly compiled FFVship.exe, you can statically link against libvship.lib instead. The commandline varies depending on your Backend:

Vulkan: `make build BACKEND=Vulkan ENABLE_STATIC=1; make buildFFVSHIP CXXFLAGS="-DVSHIP_STATIC" LDFLAGS="-L $env:Vulkan_SDK\Lib -lvulkan-1"`

HIP: `make build BACKEND=HIP ENABLE_STATIC=1; make buildFFVSHIP CXX="hipcc" CXXFLAGS="-DVSHIP_STATIC"`

Cuda: `make build BACKEND=Cuda ENABLE_STATIC=1; make buildFFVSHIP CXX="nvcc" CXXFLAGS="-DVSHIP_STATIC"`