## Introduction

Compiling FFVship on windows has been relatively complex due to lib placement not being known to compiler. As such, I upload below a dev kit containing pre compiled dependencies.

[Windows_devkit.zip](https://codeberg.org/Line-fr/Vship/releases/download/v4.0.0/Windows_devkit.zip)

## Steps

- Unzip the devkit to get a folder containing various files

- Copy these files and dump them in vship folder

- build libvship using `make build BACKEND=<your backend>` (follow the readme)

- build FFVship using `make buildFFVSHIP`

The following binary is going to need to be used next to libvship.dll and ffms2.dll (can be found either in the ffms2 repository or stolen from a pre-compiled FFVship release).
