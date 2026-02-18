[shader("compute")]
[numthreads(256, 1, 1)]
void horizontalBlur(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform float* src, uniform int64_t w, uniform int64_t h, uniform float* gaussiankernel, uniform float* gaussiankernel_integral, uniform int gaussiansize){
    int64_t x = threadIdx.x + blockIdx.x*256;
    int64_t size = w*h;
    if (x >= size) return;

    int64_t current_line = x/w;

    const int64_t begin = max(x-gaussiansize, current_line*w);
    const int64_t end = min(x+gaussiansize, (current_line+1)*w-1);
    float weight = gaussiankernel_integral[gaussiansize+end+1-x]-gaussiankernel_integral[gaussiansize+begin-x];

    float out = 0.0f;
    for (int64_t i = begin; i <= end; i++){
        out += src[i]*gaussiankernel[gaussiansize+i-x];
    }
    dst[x] = out/weight;
}

[shader("compute")]
[numthreads(32, 8, 1)]
void verticalBlur(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform float* src, uniform int64_t w, uniform int64_t h, uniform float* gaussiankernel, uniform float* gaussiankernel_integral, uniform int gaussiansize){
    int64_t x = threadIdx.x + blockIdx.x*32;
    int64_t y = threadIdx.y + blockIdx.y*8;
    if (x >= w) return;
    if (y >= h) return;

    const int64_t begin = max(y-gaussiansize, (int64_t)0);
    const int64_t end = min(y+gaussiansize, (int64_t)h-1);
    float weight = gaussiankernel_integral[gaussiansize+end+1-y]-gaussiankernel_integral[gaussiansize+begin-y];;
    float out = 0.0f;
    for (int64_t i = begin; i <= end; i++){
        out += src[i * w + x]*gaussiankernel[gaussiansize+i-y];
    }
    dst[y*w+x] = out/weight;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void TiledGaussianBlur(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* src, uniform float* dst, uniform int64_t width, uniform int64_t height, uniform float* gaussiankernel, uniform float* gaussiankernel_integral){
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

    //fill tampon
    
    for (int region_x = 0; region_x < 48; region_x += 16){
        
        for (int region_y = 0; region_y < 48; region_y += 16){
            tampon[(thy+region_y)*48+thx+region_x] = (tampon_base_x + thx +region_x >= 0 && tampon_base_x + thx +region_x < width && tampon_base_y + thy + region_y >= 0 && tampon_base_y + thy + region_y < height) ? src[(tampon_base_y+thy+region_y)*width + tampon_base_x+thx+region_x] : 0.f;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    //horizontalBlur on tampon restraint into rectangle [8 - 40][0 - 48] -> 6 pass per thread
    float tot[2];
    float out[2][3] = {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}};

    
    for (int region_x = 0; region_x < 2; region_x++){

        //border handling precompute
        const int beg = (int)max((int64_t)0, x-8+region_x*16)-(int)(x-8+region_x*16);
        const int end2 = (int)min(width, x+9+region_x*16)-(int)min(width, (x-8+region_x*16));
        tot[region_x] = gaussiankernel_integral[end2]-gaussiankernel_integral[beg];

        
        for (int region_y = 0; region_y < 3; region_y++){
            for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
                out[region_x][region_y] += tampon[(thy+region_y*16)*48 + thx+i+region_x*16]*gaussiankernel[i];
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();
    
    for (int region_x = 0; region_x < 2; region_x++){
        
        for (int region_y = 0; region_y < 3; region_y++){
            tampon[(thy+region_y*16)*48 + thx+8+region_x*16] = out[region_x][region_y]/tot[region_x];
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
            out[region_y][region_x] = 0;
            for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
                out[region_y][region_x] += tampon[(thy+i+region_y*16)*48 + thx+8+region_x*16]*gaussiankernel[i];
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();
    
    for (int region_y = 0; region_y < 2; region_y++){
        
        for (int region_x = 0; region_x < 2; region_x++){
            tampon[(thy+8+region_y*16)*48 + thx+8+region_x*16] = out[region_y][region_x]/tot[region_y];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    //if (x == 0 && y == 0) debugAddress[0] = float4(tampon[(8+0*16)*48+8+0*16], tampon[(8+1*16)*48+8+0*16], 1, 2);
    
    for (int region_x = 0; region_x < 2; region_x++){
        
        for (int region_y = 0; region_y < 2; region_y++){
            if (tampon_base_x + thx +8+region_x*16 >= 0 && tampon_base_x + thx +8+region_x*16 < width && tampon_base_y + thy +8+region_y*16 >= 0 && tampon_base_y + thy +8+region_y*16 < height) dst[(tampon_base_y+thy+8+region_y*16)*width + tampon_base_x+thx+8+region_x*16] = tampon[(thy+8+region_y*16)*48+thx+8+region_x*16];
        }
    }
}