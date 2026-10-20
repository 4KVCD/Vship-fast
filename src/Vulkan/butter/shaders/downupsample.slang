[shader("compute")]
[numthreads(16, 16, 1)]
void addsupersample2X(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* diffmap, uniform float* diffmapsmall, uniform uint64_t width, uniform uint64_t height, uniform float w){ //threads represents output pixels
    uint64_t x = threadIdx.x + blockIdx.x*16; // < width >> 1 +1
    uint64_t y = threadIdx.y + blockIdx.y*16; // < height >> 1 +1

    //int64_t newh = (height-1)/2 + 1;
    uint64_t neww = (width-1)/2 + 1;
    
    if (x >= width || y >= height) return;

    //if (y*width+x == 1841628) printf("small got %f and normal got %f\n", diffmapsmall[(y/2)*neww+x/2], diffmap[y*width+x]);

    diffmap[y*width+x] *= 1.0 - 0.3*w;
    diffmap[y*width+x] += w*diffmapsmall[(y/2)*neww+x/2];
}