import makeXYBSSIMU2;

[shader("compute")]
[numthreads(16, 16, 1)]
void downsample(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* src1, uniform float* src2, uniform float* src3, uniform float* dst1, uniform float* dst2, uniform float* dst3, uniform uint64_t width, uniform uint64_t height, uniform uint64_t info){
    //info is bit boolean (4 bits used) skipB skipY skipX result_XYB
    uint64_t x = threadIdx.x + blockIdx.x*16; // < width >> 1 +1
    uint64_t y = threadIdx.y + blockIdx.y*16; // < height >> 1 +1

    int64_t newh = (height-1)/2 + 1;
    int64_t neww = (width-1)/2 + 1;

    if (x >= neww || y >= newh) return;

    float3 srcData[4];
    for (int dy = 0; dy < 2; dy++){
        for (int dx = 0; dx < 2; dx++){
            srcData[dy*2+dx].x = src1[min((int)(2*y + dy), (int)(height-1)) * width + min((int)(2*x+dx), (int)(width-1))];
            srcData[dy*2+dx].y = src2[min((int)(2*y + dy), (int)(height-1)) * width + min((int)(2*x+dx), (int)(width-1))];
            srcData[dy*2+dx].z = src3[min((int)(2*y + dy), (int)(height-1)) * width + min((int)(2*x+dx), (int)(width-1))];
        }
    }

    float3 dst = {0.f, 0.f, 0.f};
    for (int i = 0; i < 4; i++){
        dst += srcData[i];
        //to later write srcData in XYB
        srcData[i] = rgb_to_positive_xyb_d(srcData[i]);
    }
    dst *= 0.25f;

    //done on last scale since the score won't be used for further downsampling
    if ((info & 1u) != 0u){
        dst = rgb_to_positive_xyb_d(dst);
    }

    dst1[y * neww + x] = dst.x;
    dst2[y * neww + x] = dst.y;
    dst3[y * neww + x] = dst.z;

    //write on source the XYB version directly with in registers data to reduce memory reads
    for (int dy = 0; dy < 2; dy++){
        for (int dx = 0; dx < 2; dx++){
            if (2*y+dy < height && 2*x+dx < width){
                if ((info & 2u) == 0) src1[(2*y+dy)*width+2*x+dx] = srcData[dy*2+dx].x;
                if ((info & 4u) == 0) src2[(2*y+dy)*width+2*x+dx] = srcData[dy*2+dx].y;
                if ((info & 8u) == 0) src3[(2*y+dy)*width+2*x+dx] = srcData[dy*2+dx].z;
            }
        }
    }
}
