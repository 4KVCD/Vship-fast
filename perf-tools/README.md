# Speed and exactness tools for the perf-metrics commits

Test tools only, not part of the library. Each tool links against a build of libvship (`-I <repo>/src`, the
library's import library or shared object).

- `ssimu2bench`, `cvvdpbench`, `butterbench`: `W H ref.yuv dist.yuv frames [-r repeats] [-g threads] [-q]`.
  Inputs are raw 8-bit yuv420p (BT.709, limited range). Without `-q` they first print every frame's scores as
  exact hex floats (cvvdpbench also scores every frame with its distortion map requested and prints a hash of
  the map; butterbench prints a hash of its map), then time `repeats` passes over all frames with `threads`
  threads and print `... fps`. cvvdpbench also takes `-m <display model>`, `-z` (resize to the display) and
  `-pq` (PQ BT.2020 input). With the environment variable `BENCH_PAD` set, the three tools pass the frames with
  padded lines (w+64 bytes for luma, w/2+32 for chroma, copied in row by row), as a caller with padded lines does;
  the scores must be the same as without it.
- `formatsweep W H ref.yuv dist.yuv`: builds 2 frames of 27 input formats (8 to 16-bit and float, limited and
  full range, every chroma subsampling and location, odd sizes, the YUV matrices, the transfer functions, crop
  and resize) from the 8-bit clip and prints the 3 metrics' scores as exact hex floats, one line per format.
- `bench.ps1 -Tool <bench> -Builds <dir1>,<dir2>,... -Rounds 3 -Threads 1`: paired speed runs (Windows
  PowerShell): it alternates the builds so that drift affects them alike, and prints each build's median fps.
  It expects the clips as `$env:TEMP\vship-bench\tg1080_{ref,dis}.yuv` (1920x1080, 8 frames) and
  `bee4k_{ref,dis}.yuv` (3840x2160, 6 frames).

A step's output is unchanged when the hex lines of two builds are identical. Outputs are only comparable
between builds run on the same GPU and driver.
