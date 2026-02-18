

[shader("compute")]
[numthreads(16, 16, 1)]
void downsample(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* src, uniform float* dst, uniform uint64_t width, uniform uint64_t height){
    uint64_t x = threadIdx.x + blockIdx.x*16; // < width >> 1 +1
    uint64_t y = threadIdx.y + blockIdx.y*16; // < height >> 1 +1

    uint64_t newh = (height-1)/2 + 1;
    uint64_t neww = (width-1)/2 + 1;

    if (x >= neww || y >= newh) return;

    dst[y * neww + x] = src[min((int)(2*y), (int)(height-1)) * width + min((int)(2*x), (int)(width-1))];
    dst[y * neww + x] += src[min((int)(2*y + 1), (int)(height-1)) * width + min((int)(2*x), (int)(width-1))];
    dst[y * neww + x] += src[min((int)(2*y), (int)(height-1)) * width + min((int)(2*x+1), (int)(width-1))];
    dst[y * neww + x] += src[min((int)(2*y + 1), (int)(height-1)) * width + min((int)(2*x+1), (int)(width-1))];
    dst[y * neww + x] *= 0.25f;
}