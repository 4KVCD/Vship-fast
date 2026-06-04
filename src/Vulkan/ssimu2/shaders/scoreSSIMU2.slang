
static groupshared float tampon[32*32*3];

//tampon should be groupshared memory of size 32*32
void GaussianSmartSharedLoad(const int thx, const int thy, uint tamponOffset, float* src, int64_t x, int64_t y, int64_t width, int64_t height){
    const int64_t tampon_base_x = x - thx - 8;
    const int64_t tampon_base_y = y - thy - 8;

    //fill tampon
    tampon[tamponOffset+thy*32+thx] = (tampon_base_x + thx >= 0 && tampon_base_x + thx < width && tampon_base_y + thy >= 0 && tampon_base_y + thy < height) ? src[(tampon_base_y+thy)*width + tampon_base_x+thx] : 0.f;
    tampon[tamponOffset+(thy+16)*32+thx] = (tampon_base_x + thx >= 0 && tampon_base_x + thx < width && tampon_base_y + thy + 16 >= 0 && tampon_base_y + thy + 16 < height) ? src[(tampon_base_y+thy+16)*width + tampon_base_x+thx] : 0.f;
    tampon[tamponOffset+thy*32+thx+16] = (tampon_base_x + thx +16 >= 0 && tampon_base_x + thx +16 < width && tampon_base_y + thy >= 0 && tampon_base_y + thy < height) ? src[(tampon_base_y+thy)*width + tampon_base_x+thx+16] : 0.f;
    tampon[tamponOffset+(thy+16)*32+thx+16] = (tampon_base_x + thx +16 >= 0 && tampon_base_x + thx +16 < width && tampon_base_y + thy + 16 >= 0 && tampon_base_y + thy + 16 < height) ? src[(tampon_base_y+thy+16)*width + tampon_base_x+thx+16] : 0.f;
    GroupMemoryBarrierWithGroupSync();
}

//a whole block of 16x16 should into there, x and y corresponds to their real position in the src (or slighly outside)
//at the end, the central 16*16 part of tampon contains the blurred value for each thread
//tampon is of size 32*32
float GaussianSmart_Device(const int thx, const int thy, uint tamponOffset, int64_t x, int64_t y, int64_t width, int64_t height, float* gaussiankernel, float* gaussiankernel_integral){
    //horizontalBlur on tampon restraint into rectangle [8 - 24][0 - 32] -> 2 pass per thread

    //1st pass in [8 - 24][0 - 16]
    float tot;
    float out = 0.f;
    float out2 = 0.f;
    //border handling precompute
    int64_t beg = max((int64_t)0, x-8)-(x-8);
    int64_t end2 = min(width, x+9)-min(width, (x-8));
    tot = gaussiankernel_integral[end2] - gaussiankernel_integral[beg];
    for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
        out += tampon[tamponOffset+thy*32 + thx+i]*gaussiankernel[i];
        //2nd pass in [8 - 24][16 - 32]
        out2 += tampon[tamponOffset+(thy+16)*32 + thx+i]*gaussiankernel[i];
    }
    GroupMemoryBarrierWithGroupSync();
    tampon[tamponOffset+thy*32 + thx+8] = out/tot;
    tampon[tamponOffset+(thy+16)*32 + thx+8] = out2/tot;
    GroupMemoryBarrierWithGroupSync();

    //verticalBlur on tampon[tamponOffset] restraint into rectangle [8 - 24][8 - 24] -> 1 pass per thread
    out = 0.f;
    beg = max((int64_t)0, y-8)-(y-8);
    end2 = min(height, y+9)-min(height, (y-8));
    tot = gaussiankernel_integral[end2] - gaussiankernel_integral[beg];
    for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
        out += tampon[tamponOffset+(thy+i)*32 + thx+8]*gaussiankernel[i];
    }

    GroupMemoryBarrierWithGroupSync();
    return out/tot;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void planescale_map(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform float* im1, uniform float* im2, uniform int64_t width, uniform int64_t height, uniform float* gaussiankernel, uniform float* gaussiankernel_integral, uniform uint64_t bl_x, uniform uint64_t bl_y, uniform int skipMap){
    const int64_t x = (threadIdx.x + blockIdx.x*16);
    const int64_t y = (threadIdx.y + blockIdx.y*16);

    const int64_t threadnum = 256;
    const int64_t thid = threadIdx.y*16 + threadIdx.x;

    //fill image buffers
    GaussianSmartSharedLoad(threadIdx.x, threadIdx.y, 0, im1, x, y, width, height); 
    GaussianSmartSharedLoad(threadIdx.x, threadIdx.y, 1024, im2, x, y, width, height); 

    //our final values which may be unset depending on skip
    float im1p, im2p, m1, m2, s12, s11_s22;
    float ssim = 0, artifact = 0, detailloss = 0;

    im1p = tampon[(threadIdx.y+8)*32+threadIdx.x+8];
    im2p = tampon[1024+(threadIdx.y+8)*32+threadIdx.x+8];
    GroupMemoryBarrierWithGroupSync();

    if (skipMap%2 == 0){
        //product blur a*b
        tampon[2048+thid] = tampon[thid]*tampon[1024+thid];
        tampon[2048+thid+256] = tampon[thid+256]*tampon[1024+thid+256];
        tampon[2048+thid+512] = tampon[thid+512]*tampon[1024+thid+512];
        tampon[2048+thid+768] = tampon[thid+768]*tampon[1024+thid+768];
        GroupMemoryBarrierWithGroupSync();
        s12 = GaussianSmart_Device(threadIdx.x, threadIdx.y, 2048, x, y, width, height, gaussiankernel, gaussiankernel_integral);

        //sum squared blur (a+b)**2
        float tmp;
        tmp = tampon[thid]+tampon[1024+thid];
        tampon[2048+thid] = tmp*tmp;
        tmp = tampon[thid+256]+tampon[1024+thid+256];
        tampon[2048+thid+256] = tmp*tmp;
        tmp = tampon[thid+512]+tampon[1024+thid+512];
        tampon[2048+thid+512] = tmp*tmp;
        tmp = tampon[thid+768]+tampon[1024+thid+768];
        tampon[2048+thid+768] = tmp*tmp;
        GroupMemoryBarrierWithGroupSync();
        const float sumsquared = GaussianSmart_Device(threadIdx.x, threadIdx.y, 2048, x, y, width, height, gaussiankernel, gaussiankernel_integral);
        //we can deduce a**2 + b**2 now following linearity of gaussian blur
        s11_s22 = sumsquared - 2*s12;
    }

    //then we blur our im1buffer and im2buffer to get m1 and m2
    m1 = GaussianSmart_Device(threadIdx.x, threadIdx.y, 0, x, y, width, height, gaussiankernel, gaussiankernel_integral);
    m2 = GaussianSmart_Device(threadIdx.x, threadIdx.y, 1024, x, y, width, height, gaussiankernel, gaussiankernel_integral);

    //if (x == 0 && y == 0) printf("im1 %f im2 %f m1 %f m2 %f s12 %f s1122 %f\n", im1p, im2p, m1, m2, s12, s11_s22);
    //now we have im1p, im2p, m1, m2 and (s12, s11_s22 if ssim enabled)
    if (x < width && y < height){
        if (skipMap%2==0){
            const float m11 = m1*m1;
            const float m22 = m2*m2;
            const float m12 = m1*m2;
            const float m_diff = m1-m2;
            const float num_m = fma(m_diff, m_diff*-1.0f, 1.0f);
            const float num_s = fma(s12 - m12, 2.0f, 0.0009f);
            const float denom_s = s11_s22 - m11 - m22 + 0.0009f;
            ssim = max(1.0f - ((num_m * num_s)/denom_s), 0.0f);
        }
        //edge diff
        const float v1 = (abs(im2p - m2)+1.0f) / (abs(im1p - m1)+1.0f) - 1.0f;
        artifact = max(v1, 0.0f);
        detailloss = max(v1*-1.0f, 0.0f);
    }

    //accumulation of values across the block, we use tampon for it which has size 1024*3 float > 256*6

    tampon[thid] = ssim;
    tampon[thid+256] = artifact;
    tampon[thid+512] = detailloss;
    tampon[thid+768] = (ssim*ssim)*(ssim*ssim);
    tampon[thid+1024] = (artifact*artifact)*(artifact*artifact);
    tampon[thid+1280] = (detailloss*detailloss)*(detailloss*detailloss);
    GroupMemoryBarrierWithGroupSync();
    //now we need to do some pointer jumping to regroup every block sums;
    int next = threadnum;
    while (next > 1){
        next /= 2;
        if (thid < next){
            for (uint i = 0; i < 6; i++){
                tampon[thid+256*i] += tampon[thid+256*i+next];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (thid == 0){
        for (uint i = 0; i < 6; i++){
            dst[i*bl_x*bl_y + blockIdx.y*bl_x + blockIdx.x] = tampon[256*i]/(width*height);
        }
    }
}
