[shader("compute")]
[numthreads(256, 1, 1)]
void maskinit(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* hfx, uniform float* uhfx, uniform float* hfy, uniform float* uhfy, uniform uint64_t width, uniform float a, uniform float b, uniform float c){
    uint64_t x = threadIdx.x + blockIdx.x*256;

    if (x >= width) return;

    float xdiff = (uhfx[x]+hfx[x])*a;
    float ydiff = uhfy[x]*b + hfy[x]*c;
    float res = sqrt(xdiff*xdiff + ydiff*ydiff);

    const float mul = 6.19424080439f;
    const float bias_arg = 12.61050594197f;
    const float bias = mul*bias_arg;
    res = sqrt(mul * abs(res) + bias) - sqrt(bias);
    hfx[x] = res;
    //if (x == 10000) printf("maskxyb : %f from hfx %f, uhfx %f, hfy %f, uhfy %f\n", dst[x], hfx[x], uhfx[x], hfy[x], uhfy[x]);
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
        if (y < height - kStep) {
            float v = src[(y+kStep)*width + x - kStep];
            mins = StoreMin3(v, mins);
        }
    }
    if (x < width - kStep) {
        float v = src[y*width + x + kStep];
        mins = StoreMin3(v, mins);
        if (y >= kStep) {
            float v = src[(y-kStep)*width + x + kStep];
            mins = StoreMin3(v, mins);
        }
        if (y < height - kStep) {
            float v = src[(y+kStep)*width + x + kStep];
            mins = StoreMin3(v, mins);
        }
    }
    if (y >= kStep) {
        float v = src[(y-kStep)*width + x];
        mins = StoreMin3(v, mins);
    }
    if (y < height - kStep) {
        float v = src[(y+kStep)*width + x];
        mins = StoreMin3(v, mins);
    }
    dst[ux] = (0.45f * mins.x + 0.3f * mins.y + 0.25f * mins.z);
    //if (ux == 10000) printf("Erosion : %f from %f\n", dst[ux], src[ux]);
}
