
//these parameters needs to be the same in the gaussianblur.hpp file

//for now only accepts squares Tiling (see GaussianSmart_device out and tot arrays)
//only allows powers of 2 for Thx and Thy due to pointer jumping (8/8 16/16 and 32/32)
static const int gaussianThx = 16;
static const int gaussianThy = 16;
//Optimization Island in gaussianSize lies in gaussianTh/2
static const int gaussianSize = 8; //if gaussianSize is below gaussianTh/2, you need to change tampon size

static const int gaussianTileX = 1;
static const int gaussianTileY = 1;

static const int tamponWidth = (gaussianThx*gaussianTileX + 2*gaussianSize);
static const int tamponHeight = (gaussianThy*gaussianTileY + 2*gaussianSize);

//constexpr max doesnt work so you may swap between tamponWidth*tamponHeight*3 and 6*gaussianThx*gaussianThy
static groupshared float tampon[tamponWidth*tamponHeight*3];

void GaussianSmartSharedLoad(const int thx, const int thy, uint tamponOffset, float* src, int64_t x, int64_t y, int64_t width, int64_t height){
    const int64_t tampon_base_x = x - (int64_t)thx - (int64_t)gaussianSize;
    const int64_t tampon_base_y = y - (int64_t)thy - (int64_t)gaussianSize;
    const int thid = thy*gaussianThx+thx;

    for (int i = thid; i < tamponWidth*tamponHeight; i+=gaussianThx*gaussianThy){
        const int tamponX = i%tamponWidth;
        const int tamponY = i/tamponWidth;

        tampon[tamponOffset+i] = (tampon_base_x + tamponX >= 0 && tampon_base_x + tamponX < width && tampon_base_y + tamponY >= 0 && tampon_base_y + tamponY < height) ? src[(tampon_base_y+tamponY)*width + tampon_base_x+tamponX] : 0.f;
    }
    GroupMemoryBarrierWithGroupSync();
}

//the following comment is only true if you uncomment putting the end result in the tampon, currently, the result is only put in output
//at the end, the central (tamponWidth-gaussianSize*2)*(tamponHeight-gaussianSize*2) part of tampon contains the blurred value for each thread
float[gaussianTileX*gaussianTileY] GaussianSmart_Device(const int thx, const int thy, uint tamponOffset, int64_t x, int64_t y, int64_t width, int64_t height, float* gaussiankernel, float* gaussiankernel_integral){
    float tot[gaussianTileX];
    float out[gaussianTileX*(gaussianTileY+1)];
    for (int i = 0; i < gaussianTileX; i++) for (int j = 0; j < gaussianTileY+1; j++) out[i+j*gaussianTileX] = 0.f;

    //horizontal blur on each X tile columns (each of size gaussianTileX * tamponHeight)

    for (int region_x = 0; region_x < gaussianTileX; region_x++){

        //border handling precompute
        const int beg = (int)(max((int64_t)0, x-(int64_t)gaussianSize+(int64_t)region_x*(int64_t)gaussianThx)-(x-(int64_t)gaussianSize+(int64_t)region_x*(int64_t)gaussianThx));
        const int end2 = (int)(min(width, x+(int64_t)gaussianSize+1+(int64_t)region_x*(int64_t)gaussianThx)-min(width, (x-(int64_t)gaussianSize+(int64_t)region_x*(int64_t)gaussianThx)));
        tot[region_x] = gaussiankernel_integral[end2]-gaussiankernel_integral[beg];

        //if (2*gaussianSize)%gaussianThy != 0, we will have idle threads at last step
        int tileRow = 0;

        for (int tamponY = thy; tamponY < tamponHeight; tamponY += gaussianThy){
            for (int i = 0; i < 2*gaussianSize+1; i++){ //starting gaussianSize to the left and going gaussianSize to the right
                out[region_x+tileRow*gaussianTileX] += tampon[tamponOffset+tamponY*tamponWidth + thx+i+region_x*gaussianThx]*gaussiankernel[i];
            }
            tileRow++;
        }
    }

    //rewrite result in tampon and normalize
    GroupMemoryBarrierWithGroupSync();

    for (int region_x = 0; region_x < gaussianTileX; region_x++){
        int tileRow = 0;

        for (int tamponY = thy; tamponY < tamponHeight; tamponY += gaussianThy){
            tampon[tamponOffset+(tamponY)*tamponWidth + thx+gaussianSize+region_x*gaussianThx] = out[region_x+tileRow*gaussianTileX]/tot[region_x];
            tileRow++;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    //verticalBlur on tampon restraint into final center rectangle.

    for (int region_y = 0; region_y < gaussianTileY; region_y++){

        //border handling precompute
        const int beg = (int)(max((int64_t)0, y-(int64_t)gaussianSize+(int64_t)region_y*(int64_t)gaussianThy)-(y-(int64_t)gaussianSize+(int64_t)region_y*(int64_t)gaussianThy));
        const int end2 = (int)(min(height, y+(int64_t)gaussianSize+1+(int64_t)region_y*(int64_t)gaussianThy)-min(height, (y-(int64_t)gaussianSize+(int64_t)region_y*(int64_t)gaussianThy)));
        tot[region_y] = gaussiankernel_integral[end2] - gaussiankernel_integral[beg];


        for (int region_x = 0; region_x < gaussianTileX; region_x++){
            out[region_y*gaussianTileX+region_x] = 0;
            for (int i = 0; i < 2*gaussianSize+1; i++){ //starting 8 to the left and going 8 to the right
                out[region_y*gaussianTileX+region_x] += tampon[tamponOffset+(thy+i+region_y*gaussianThy)*tamponWidth + thx+gaussianSize+region_x*gaussianThx]*gaussiankernel[i];
            }
        }
    }

    //put data to result, one data point per Tile
    float output[gaussianTileX*gaussianTileY];
    for (int region_y = 0; region_y < gaussianTileY; region_y++){

        for (int region_x = 0; region_x < gaussianTileX; region_x++){
            output[region_y*gaussianTileX+region_x] = out[region_y*gaussianTileX+region_x]/tot[region_y];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    return output;
    //put data back to tampon -> extra operations and sync
    /*

    for (int region_y = 0; region_y < gaussianTileY; region_y++){
        for (int region_x = 0; region_x < gaussianTileX; region_x++){
            tampon[tamponOffset+(thy+gaussianSize+region_y*gaussianThy)*tamponWidth + thx+gaussianSize+region_x*gaussianThx] = out[region_y*gaussianTileX+region_x]/tot[region_y];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    */
}

[shader("compute")]
[numthreads(gaussianThx, gaussianThy, 1)]
void planescale_map(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform float* im1, uniform float* im2, uniform int64_t width, uniform int64_t height, uniform float* gaussiankernel, uniform float* gaussiankernel_integral, uniform uint64_t bl_x, uniform uint64_t bl_y, uniform int skipMap){
    const int64_t x = (threadIdx.x + blockIdx.x*gaussianThx*gaussianTileX);
    const int64_t y = (threadIdx.y + blockIdx.y*gaussianThy*gaussianTileY);

    const int64_t threadnum = gaussianThx*gaussianThy;
    const int thid = threadIdx.y*gaussianThx + threadIdx.x;

    //fill image buffers
    GaussianSmartSharedLoad(threadIdx.x, threadIdx.y, 0, im1, x, y, width, height);
    GaussianSmartSharedLoad(threadIdx.x, threadIdx.y, tamponWidth*tamponHeight, im2, x, y, width, height);

    //our final values which may be unset depending on skip
    float im1p[gaussianTileX*gaussianTileY], im2p[gaussianTileX*gaussianTileY], m1[gaussianTileX*gaussianTileY], m2[gaussianTileX*gaussianTileY], s12[gaussianTileX*gaussianTileY], s11_s22[gaussianTileX*gaussianTileY];
    float ssim[gaussianTileX*gaussianTileY], artifact[gaussianTileX*gaussianTileY], detailloss[gaussianTileX*gaussianTileY];
    for (int i = 0; i < gaussianTileX*gaussianTileY; i++){
        ssim[i] = 0;
        artifact[i] = 0;
        detailloss[i] = 0;
    }

    for (int region_y = 0; region_y < gaussianTileY; region_y++){
        for (int region_x = 0; region_x < gaussianTileX; region_x++){
            im1p[region_y*gaussianTileX+region_x] = tampon[(threadIdx.y+gaussianSize+gaussianThy*region_y)*tamponWidth+threadIdx.x+gaussianSize+gaussianThx*region_x];
            im2p[region_y*gaussianTileX+region_x] = tampon[tamponWidth*tamponHeight+(threadIdx.y+gaussianSize+gaussianThy*region_y)*tamponWidth+threadIdx.x+gaussianSize+gaussianThx*region_x];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (skipMap%2 == 0){
        //product blur a*b
        for (int i = thid; i < tamponWidth*tamponHeight; i += gaussianThx*gaussianThy){
            tampon[tamponWidth*tamponHeight*2+i] = tampon[tamponWidth*tamponHeight+i]*tampon[i];
        }
        GroupMemoryBarrierWithGroupSync();
        s12 = GaussianSmart_Device(threadIdx.x, threadIdx.y, tamponWidth*tamponHeight*2, x, y, width, height, gaussiankernel, gaussiankernel_integral);

        //sum squared blur (a+b)**2
        float tmp;
        for (int i = thid; i < tamponWidth*tamponHeight; i += gaussianThx*gaussianThy){
            tmp = tampon[tamponWidth*tamponHeight+i]+tampon[i];
            tampon[tamponWidth*tamponHeight*2+i] = tmp*tmp;
        }
        GroupMemoryBarrierWithGroupSync();
        const float sumsquared[gaussianTileX*gaussianTileY] = GaussianSmart_Device(threadIdx.x, threadIdx.y, tamponWidth*tamponHeight*2, x, y, width, height, gaussiankernel, gaussiankernel_integral);
        //we can deduce a**2 + b**2 now following linearity of gaussian blur
        for (int i = 0; i < gaussianTileX*gaussianTileY; i++){
            s11_s22[i] = sumsquared[i] - 2*s12[i];
        }
    }

    //then we blur our im1buffer and im2buffer to get m1 and m2
    m1 = GaussianSmart_Device(threadIdx.x, threadIdx.y, 0, x, y, width, height, gaussiankernel, gaussiankernel_integral);
    m2 = GaussianSmart_Device(threadIdx.x, threadIdx.y, tamponWidth*tamponHeight, x, y, width, height, gaussiankernel, gaussiankernel_integral);

    //if (x == 0 && y == 0) printf("im1 %f im2 %f m1 %f m2 %f s12 %f s1122 %f\n", im1p, im2p, m1, m2, s12, s11_s22);
    //now we have im1p, im2p, m1, m2 and (s12, s11_s22 if ssim enabled)
    for (int i = 0; i < gaussianTileX*gaussianTileY; i++){
        const int region_x = i%gaussianTileX;
        const int region_y = i/gaussianTileX;
        if (x+gaussianThx*region_x < width && y+gaussianThy*region_y < height){
            if (skipMap%2==0){
                const float m11 = m1[i]*m1[i];
                const float m22 = m2[i]*m2[i];
                const float m12 = m1[i]*m2[i];
                const float m_diff = m1[i]-m2[i];
                const float num_m = fma(m_diff, m_diff*-1.0f, 1.0f);
                const float num_s = fma(s12[i] - m12, 2.0f, 0.0009f);
                const float denom_s = s11_s22[i] - m11 - m22 + 0.0009f;
                ssim[i] = max(1.0f - ((num_m * num_s)/denom_s), 0.0f);
            }
            //edge diff
            const float v1 = (abs(im2p[i] - m2[i])+1.0f) / (abs(im1p[i] - m1[i])+1.0f) - 1.0f;
            artifact[i] = max(v1, 0.0f);
            detailloss[i] = max(v1*-1.0f, 0.0f);
        }
    }

    //accumulation of values across the block, we use tampon for it which has size 1024*3 float > 256*6

    tampon[thid] = 0;
    tampon[thid+gaussianThx*gaussianThy] = 0;
    tampon[thid+gaussianThx*gaussianThy*2] = 0;
    tampon[thid+gaussianThx*gaussianThy*3] = 0;
    tampon[thid+gaussianThx*gaussianThy*4] = 0;
    tampon[thid+gaussianThx*gaussianThy*5] = 0;
    for (int i = 0; i < gaussianTileX*gaussianTileY; i++){
        tampon[thid] += ssim[i];
        tampon[thid+gaussianThx*gaussianThy] += artifact[i];
        tampon[thid+gaussianThx*gaussianThy*2] += detailloss[i];
        tampon[thid+gaussianThx*gaussianThy*3] += (ssim[i]*ssim[i])*(ssim[i]*ssim[i]);
        tampon[thid+gaussianThx*gaussianThy*4] += (artifact[i]*artifact[i])*(artifact[i]*artifact[i]);
        tampon[thid+gaussianThx*gaussianThy*5] += (detailloss[i]*detailloss[i])*(detailloss[i]*detailloss[i]);
    }
    GroupMemoryBarrierWithGroupSync();
    //now we need to do some pointer jumping to regroup every block sums;
    int next = threadnum;
    while (next > 1){
        next /= 2;
        if (thid < next){
            for (uint i = 0; i < 6; i++){
                tampon[thid+gaussianThx*gaussianThy*i] += tampon[thid+gaussianThx*gaussianThy*i+next];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (thid == 0){
        for (uint i = 0; i < 6; i++){
            dst[i*bl_x*bl_y + blockIdx.y*bl_x + blockIdx.x] = tampon[gaussianThx*gaussianThy*i]/(width*height);
        }
    }
}
