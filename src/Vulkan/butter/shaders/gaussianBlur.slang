import frequencyOpsModule;

// the passes TiledGaussianBlur can do after the blur (postOp), as BlurPostOp in gaussianblur.hpp
static const int POST_NONE = 0;
static const int POST_SUBTRACT_REMOVE_RANGE = 4;
static const int POST_SUBTRACT_AMPLIFY_RANGE = 5;

[shader("compute")]
[numthreads(16, 16, 1)]
void TiledGaussianBlur(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* src, uniform float* dst, uniform int64_t width, uniform int64_t height, uniform float* gaussiankernel, uniform float* gaussiankernel_integral, uniform float* second, uniform int postOp, uniform float w){
    int64_t originalBl_X = blockIdx.x;
    //let's determine which scale our block is in and adjust our input parameters accordingly

    const int64_t blockwidth = (width-1)/32+1;

    const int64_t x = threadIdx.x + 32*(originalBl_X%blockwidth);
    const int64_t y = threadIdx.y + 32*(originalBl_X/blockwidth);
    const int64_t thx = threadIdx.x;
    const int64_t thy = threadIdx.y;

    static groupshared float tampon[48*48]; //we import into tampon, compute onto tampon and then put into dst
    //tampon has 8 of border on each side with no thread
    const int tampon_base_x = (int)(x - thx - 8);
    const int tampon_base_y = (int)(y - thy - 8);

    //fill tampon: the 9 loads of a thread are all issued before its first store, so that their waits overlap
    float loaded[9];
    [ForceUnroll]
    for (int region = 0; region < 9; region++){
        const int region_x = 16*(region/3);
        const int region_y = 16*(region%3);
        loaded[region] = (tampon_base_x + thx +region_x >= 0 && tampon_base_x + thx +region_x < width && tampon_base_y + thy + region_y >= 0 && tampon_base_y + thy + region_y < height) ? src[(tampon_base_y+thy+region_y)*width + tampon_base_x+thx+region_x] : 0.f;
    }
    [ForceUnroll]
    for (int region = 0; region < 9; region++){
        tampon[(thy+16*(region%3))*48+thx+16*(region/3)] = loaded[region];
    }
    GroupMemoryBarrierWithGroupSync();

    //horizontalBlur on tampon restraint into rectangle [8 - 40][0 - 48] -> 6 pass per thread
    float tot[2];
    float out[2*3] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    
    for (int region_x = 0; region_x < 2; region_x++){

        //border handling precompute
        const int beg = (int)max((int64_t)0, x-8+region_x*16)-(int)(x-8+region_x*16);
        const int end2 = (int)min(width, x+9+region_x*16)-(int)min(width, (x-8+region_x*16));
        tot[region_x] = gaussiankernel_integral[end2]-gaussiankernel_integral[beg];

        
        for (int region_y = 0; region_y < 3; region_y++){
            for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
                out[region_x*3+region_y] += tampon[(thy+region_y*16)*48 + thx+i+region_x*16]*gaussiankernel[i];
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();
    
    for (int region_x = 0; region_x < 2; region_x++){
        
        for (int region_y = 0; region_y < 3; region_y++){
            tampon[(thy+region_y*16)*48 + thx+8+region_x*16] = out[region_x*3+region_y]/tot[region_x];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    //verticalBlur on tampon restraint into rectangle [8 - 40][8 - 40] -> 4 pass per thread
    
    for (int region_y = 0; region_y < 2; region_y++){
        
        //border handling precompute
        const int beg = (int)max((int64_t)0, y-8+region_y*16)-(int)(y-8+region_y*16);
        const int end2 = (int)min(height, y+9+region_y*16)-(int)min(height, (y-8+region_y*16));
        tot[region_y] = gaussiankernel_integral[end2] - gaussiankernel_integral[beg];

        
        for (int region_x = 0; region_x < 2; region_x++){
            out[region_y+3*region_x] = 0;
            for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
                out[region_y+3*region_x] += tampon[(thy+i+region_y*16)*48 + thx+8+region_x*16]*gaussiankernel[i];
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();
    
    for (int region_y = 0; region_y < 2; region_y++){
        
        for (int region_x = 0; region_x < 2; region_x++){
            tampon[(thy+8+region_y*16)*48 + thx+8+region_x*16] = out[region_y+3*region_x]/tot[region_y];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // postOp: also the pass that followed the blur, on the blurred value b read back from tampon (as it was read back
    // from dst before): second = src - b, and dst = RemoveRangeAroundZero(b, w) (subarray_removerangearound0) or
    // AmplifyRangeAroundZero(b, w) (subarray_amplifyrangearound0). second must not be src, whose tile borders other
    // workgroups still read.
    for (int region_x = 0; region_x < 2; region_x++){

        for (int region_y = 0; region_y < 2; region_y++){
            if (tampon_base_x + thx +8+region_x*16 >= 0 && tampon_base_x + thx +8+region_x*16 < width && tampon_base_y + thy +8+region_y*16 >= 0 && tampon_base_y + thy +8+region_y*16 < height){
                const int64_t pixel = (tampon_base_y+thy+8+region_y*16)*width + tampon_base_x+thx+8+region_x*16;
                const float b = tampon[(thy+8+region_y*16)*48+thx+8+region_x*16];
                if (postOp == POST_NONE){
                    dst[pixel] = b;
                } else {
                    second[pixel] = src[pixel] - b;
                    dst[pixel] = (postOp == POST_SUBTRACT_REMOVE_RANGE) ? RemoveRangeAroundZero_Device(b, w) : AmplifyRangeAroundZero_Device(b, w);
                }
            }
        }
    }
}
