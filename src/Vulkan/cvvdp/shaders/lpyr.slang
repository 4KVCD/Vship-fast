static const float kernel_a = 0.4f;
static const float gaussPyrKernel[5] = {0.25-kernel_a/2.,0.25, kernel_a, 0.25, 0.25-kernel_a/2.};

//separable, but not worth it for a kernel of size 5
[shader("compute")]
[numthreads(256, 1, 1)]
void gaussPyrReduce(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform float* src, uniform int64_t source_width, uniform int64_t source_height){
    const int64_t thid = threadIdx.x + blockIdx.x * 256;
    int64_t nw = (source_width+1)/2;
    int64_t nh = (source_height+1)/2;
    if (thid >= nw*nh) return;

    //in new space
    const int64_t x = thid%nw;
    const int64_t y = thid/nw;

    float nval = 0.f;

    for (int dx = -2; dx <= 2; dx++){
        const float kernel_x = gaussPyrKernel[dx+2];
        int64_t ref_ind_x = 2*x + dx;
        //symmetric padding
        if (ref_ind_x < 0) ref_ind_x = -ref_ind_x-1;
        if (ref_ind_x >= source_width) ref_ind_x = 2*source_width - ref_ind_x - 2;
        for (int dy = -2; dy <= 2; dy++){
            const float kernel_y = gaussPyrKernel[dy+2];
            int64_t ref_ind_y = 2*y + dy;
            //symmetric padding
            if (ref_ind_y < 0) ref_ind_y = -ref_ind_y-1;
            if (ref_ind_y >= source_height) ref_ind_y = 2*source_height - ref_ind_y - 2;

            nval += kernel_x*kernel_y*src[ref_ind_y*source_width+ref_ind_x];
            //if (y*nw+x == 0) printf("Gpyr from %f at %dx%d\n", src[ref_ind_y*source_width+ref_ind_x], ref_ind_x, ref_ind_y);
        }
    }

    dst[y*nw+x] = nval;
    //if (y*nw+x == 0) printf("Gpyr got %f at width %lld\n", nval, source_width);
}

//separable, but not worth it for a kernel of size 5
[shader("compute")]
[numthreads(256, 1, 1)]
void gaussPyrExpand(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform float* src, uniform int64_t new_width, uniform int64_t new_height, uniform uint conditions){
    const int64_t thid = threadIdx.x + blockIdx.x * 256;
    int64_t ow = (new_width+1)/2;
    int64_t oh = (new_height+1)/2;
    if (thid >= new_width*new_height) return;

    //in new space
    const int64_t x = thid%new_width;
    const int64_t y = thid/new_width;

    float nval = 0.f;

    int parity_x = (int)(x%2);
    int parity_y = (int)(y%2);

    for (int dx = -2+parity_x; dx <= 2; dx+=2){
        const float kernel_x = 2*gaussPyrKernel[dx+2];
        int64_t ref_ind_x = (x + dx)/2; //funny: x+dx is always even
        if (ref_ind_x < 0) ref_ind_x = -ref_ind_x-1;
        if (ref_ind_x >= ow) ref_ind_x = 2*ow - ref_ind_x -2;
        for (int dy = -2+parity_y; dy <= 2; dy+=2){
            const float kernel_y = 2*gaussPyrKernel[dy+2];
            int64_t ref_ind_y = (y+dy)/2; //(y+dy) is always even
            if (ref_ind_y < 0) ref_ind_y = -ref_ind_y-1;
            if (ref_ind_y >= oh) ref_ind_y = 2*oh - ref_ind_y -2;

            nval += kernel_x*kernel_y*src[ref_ind_y*ow+ref_ind_x];
        }
    }

    if ((conditions&4) == 4){ //clampMin
        nval = max(nval, 0.01f);
    }

    //if (thid == 0) printf("GlayerEx: %f at %lld\n", nval, new_width);

    if ((conditions&1) == 1) { //subdst
        dst[thid] -= nval;
        //if (thid == 0) printf("Layer got %f at width %lld from %f - %f\n", dst[0], new_width, dst[0]+nval, nval);
    } else if ((conditions&2) == 2) { //adddst
        dst[thid] += nval;
    } else {
        dst[thid] = nval;
        //if (thid == 13*1024 + 64 && new_width*new_height == 1184264) printf("We got %f at width %lld from %f\n", dst[thid] , new_width, src[y/2*ow+x/2]);
        //if (thid == 0) printf("Layer got %f at width %lld from %f\n", dst[0] , new_width, src[0]);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void baseBandPyrRefine(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform float* p, uniform float* Lbkg, uniform int64_t width, uniform uint conditions, uniform int multiplier){
    const int64_t thid = threadIdx.x + blockIdx.x * 256;
    if (thid >= width) return;

    //if (thid == 0) printf("Layer i&s %f at width %lld\n", p[thid], width);

    float val;
    if ((conditions&1) != 1){ //!isMean
        val = min(p[thid]/max(0.01f, Lbkg[thid]), 1000.f);
        //if (thid == 13*1024 + 64 && width == 1920*1080) printf("baseBandPyrRefine %f at width %d\n", p[thid], width);
    } else {
        //if (thid == 0) printf("value %f %f\n", Lbkg[0], p[thid]);
        //then our adress is the mean, a single float at 0
        val = min(p[thid]/max(0.01f, Lbkg[0]), 1000.f);
    }
    dst[thid] = val*multiplier;
    //if (thid == 0) printf("baseBandPyrRefine %f at width %lld\n", p[thid], width);
}
