// libjxl's Butteraugli masking (lib/jxl/butteraugli/butteraugli.cc). libjxl compiles these functions for the CPU it
// runs on (AVX2 here), and its compiler fuses each multiplication and addition written in one C++ expression (the
// left product when both sides of an addition are products): those are the fma() calls below. Every other
// operation is rounded on its own, and square roots are correctly rounded like the CPU's.

import exactMathModule;

// libjxl's CombineChannelsForMasking then DiffPrecompute (mul 6.19424080439, bias 12.61050594197), with a, b and c
// its muls 2.5, 0.4 and 0.4
[shader("compute")]
[numthreads(256, 1, 1)]
void maskinit(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* hfx, uniform float* uhfx, uniform float* hfy, uniform float* uhfy, uniform uint64_t width, uniform float a, uniform float b, uniform float c){
    uint64_t x = threadIdx.x + blockIdx.x*256;

    if (x >= width) return;

    const float xdiff = (uhfx[x] + hfx[x])*a;
    const float ydiff = fma(uhfy[x], b, hfy[x]*c);
    const float combined = sqrtRounded(fma(xdiff, xdiff, ydiff*ydiff));

    const float mul = 6.19424080439f;
    const float bias_arg = 12.61050594197f;
    const float bias = mul*bias_arg;
    hfx[x] = sqrtRounded(fma(mul, abs(combined), bias)) - sqrtRounded(bias);
}

float3 StoreMin3(const float v, float3 mins) {
  if (v < mins.z) {
    if (v < mins.x) {
      mins.z = mins.y;
      mins.y = mins.x;
      mins.x = v;
    } else if (v < mins.y) {
      mins.z = mins.y;
      mins.y = v;
    } else {
      mins.z = v;
    }
  }
  return mins;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fuzzyerrosion(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* src, uniform float* dst, uniform uint64_t width, uniform uint64_t height){
    uint64_t ux = threadIdx.x + blockIdx.x*256;

    uint64_t x = ux%width;
    uint64_t y = ux/width;

    if (y >= height) return;

    const uint kStep = 3;
    float3 mins;
    mins.x = src[ux];
    mins.y = 2.0f * mins.x;
    mins.z = mins.y;
    if (x >= kStep) {
        float v = src[y*width + x - kStep];
        mins = StoreMin3(v, mins);
        if (y >= kStep) {
            float v = src[(y-kStep)*width + x - kStep];
            mins = StoreMin3(v, mins);
        }
        if (y + kStep < height) {
            float v = src[(y+kStep)*width + x - kStep];
            mins = StoreMin3(v, mins);
        }
    }
    if (x + kStep < width) {
        float v = src[y*width + x + kStep];
        mins = StoreMin3(v, mins);
        if (y >= kStep) {
            float v = src[(y-kStep)*width + x + kStep];
            mins = StoreMin3(v, mins);
        }
        if (y + kStep < height) {
            float v = src[(y+kStep)*width + x + kStep];
            mins = StoreMin3(v, mins);
        }
    }
    if (y >= kStep) {
        float v = src[(y-kStep)*width + x];
        mins = StoreMin3(v, mins);
    }
    if (y + kStep < height) {
        float v = src[(y+kStep)*width + x];
        mins = StoreMin3(v, mins);
    }
    // libjxl: 0.45f * min0 + 0.3f * min1 + 0.25f * min2
    dst[ux] = fma(0.25f, mins.z, fma(0.45f, mins.x, 0.3f * mins.y));
}
