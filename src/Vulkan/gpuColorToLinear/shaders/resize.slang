import vshipBicubicModule;

[shader("compute")]
[numthreads(256, 1, 1)]
void horizontalResizeTranspose(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform float* src, uniform int64_t source_width, uniform int64_t source_height, uniform int64_t resize_width){
    const int64_t thid = threadIdx.x + 256*blockIdx.x;
    if (thid >= resize_width*source_height) return; //x indexed on result

    //destination result
    const int64_t x = thid%resize_width;
    const int64_t y = thid/resize_width;

    //we need to find the closest corresponding left point on original image
    //-0.5 corresponds to our kernel using location "center" instead of "topleft"
    const float approx_source_x = (float)(x+0.5f)*(float)source_width/(float)resize_width-0.5f;
    const int64_t x_oriBase = (int64_t)(approx_source_x);

    //x_oriBase can be -1
    const float v1 = (x_oriBase >= 0) ?  src[y*source_width+x_oriBase] : src[y*source_width+x_oriBase+1];
    const float v0 = (x_oriBase-1 >= 0) ? src[y*source_width+x_oriBase-1] : v1;
    const float v2 = (x_oriBase+1 < source_width) ? src[y*source_width+x_oriBase+1] : v1;
    const float v3 = (x_oriBase+2 < source_width) ? src[y*source_width+x_oriBase+2] : v2;
    float4 interpolator = loadCubicHermitSpline(v1, (v2-v0)/2.f, v2, (v3-v1)/2.f);
    const float res = getCubicHermitSpline(interpolator, approx_source_x - (float)x_oriBase);

    //dst is transposed
    dst[x*source_height+y] = res;
}