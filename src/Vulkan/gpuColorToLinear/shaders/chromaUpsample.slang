import vshipBicubicModule;

//block x should range from 0 to smallwidth INCLUDED
//dst of size width*height while src is of size smallwidth*height
[shader("compute")]
[numthreads(16, 16, 1)]
void bicubicHorizontalCenterUpscaleX2(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform float* src, uniform uint64_t width, uniform uint64_t height){
    int64_t x = (int64_t)(threadIdx.x + 16*blockIdx.x);
    x--;
    const int64_t y = (int64_t)(threadIdx.y + 16*blockIdx.y);
    const int64_t smallwidth = (int64_t)((width-1)/2+1);
    float4 interpolator = getHorizontalInterpolator_device(src, x, y, smallwidth, height);
    //this interpolator is valid on interval [0, 1] representing [x, x+1]
    //we are Center so we are interested in values: 0.25 and 0.75
    if (y < height){
        if (x != -1 && 2*x+1 < width) dst[y*width + 2*x+1] = getCubicHermitSpline(interpolator, 0.25f);
        if (2*x+2 < width) dst[y*width + 2*x+2] = getCubicHermitSpline(interpolator, 0.75f);
    }
}

//block x should range from 0 to smallwidth-1 INCLUDED
//dst of size width*height while src is of size smallwidth*height
[shader("compute")]
[numthreads(16, 16, 1)]
void bicubicHorizontalLeftUpscaleX2(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform float* src, uniform uint64_t width, uniform uint64_t height){
    const int64_t x = (int64_t)(threadIdx.x + 16*blockIdx.x);
    const int64_t y = (int64_t)(threadIdx.y + 16*blockIdx.y);
    const int64_t smallwidth = (int64_t)((width-1)/2+1);
    float4 interpolator = getHorizontalInterpolator_device(src, x, y, smallwidth, height);
    //this interpolator is valid on interval [0, 1] representing [x, x+1]
    //we are Left so we are interested in values: 0 and 0.5 (0 is directly our value)
    if (y < height){
        if (x*2 < width) dst[y*width + 2*x] = src[y*smallwidth + x];
        if (x*2+1 < width) dst[y*width + 2*x+1] = getCubicHermitSpline(interpolator, 0.5f);
    }
}

//block x should range from 0 to smallwidth INCLUDED
//dst of size width*height while src is of size smallwidth*height
[shader("compute")]
[numthreads(16, 16, 1)]
void bicubicHorizontalCenterUpscaleX4(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform float* src, uniform uint64_t width, uniform uint64_t height){
    int64_t x = (int64_t)(threadIdx.x + 16*blockIdx.x);
    x--;
    const int64_t y = (int64_t)(threadIdx.y + 16*blockIdx.y);
    const int64_t smallwidth = (int64_t)((width-1)/4+1);
    float4 interpolator = getHorizontalInterpolator_device(src, x, y, smallwidth, height);
    //this interpolator is valid on interval [0, 1] representing [x, x+1]
    //we are Center so we are interested in values: 0.125, 0.375, 0.625 and 0.875
    if (y < height){
        if (x != -1 && 4*x+2 < width) dst[y*width + 4*x+2] = getCubicHermitSpline(interpolator, 0.125f);
        if (x != -1 && 4*x+3 < width) dst[y*width + 4*x+3] = getCubicHermitSpline(interpolator, 0.375f);
        if (4*x+4 < width) dst[y*width + 4*x+4] = getCubicHermitSpline(interpolator, 0.625f);
        if (4*x+5 < width) dst[y*width + 4*x+5] = getCubicHermitSpline(interpolator, 0.875f);
    }
}

//block x should range from 0 to smallwidth-1 INCLUDED
//dst of size width*height while src is of size smallwidth*height
[shader("compute")]
[numthreads(16, 16, 1)]
void bicubicHorizontalLeftUpscaleX4(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform float* src, uniform uint64_t width, uniform uint64_t height){
    const int64_t x = (int64_t)(threadIdx.x + 16*blockIdx.x);
    const int64_t y = (int64_t)(threadIdx.y + 16*blockIdx.y);
    const int64_t smallwidth = (int64_t)((width-1)/4+1);
    float4 interpolator = getHorizontalInterpolator_device(src, x, y, smallwidth, height);
    //this interpolator is valid on interval [0, 1] representing [x, x+1]
    //we are Left so we are interested in values: 0, 0.25, 0.5 and 0.75 (0 is directly our value)
    if (y < height){
        if (4*x < width) dst[y*width + 4*x] = src[y*smallwidth + x];
        if (4*x+1 < width) dst[y*width + 4*x+1] = getCubicHermitSpline(interpolator, 0.25f);
        if (4*x+2 < width) dst[y*width + 4*x+2] = getCubicHermitSpline(interpolator, 0.5f);
        if (4*x+3 < width)  dst[y*width + 4*x+3] = getCubicHermitSpline(interpolator, 0.75f);
    }
}

//block y should range from 0 to smallheight INCLUDED
//dst of size width*height while src is of size width*smallheight
[shader("compute")]
[numthreads(16, 16, 1)]
void bicubicVerticalCenterUpscaleX2(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform float* src, uniform uint64_t width, uniform uint64_t height){
    const int64_t x = (int64_t)(threadIdx.x + 16*blockIdx.x);
    int64_t y = (int64_t)(threadIdx.y + 16*blockIdx.y);
    y--;
    const int64_t smallheight = (height-1)/2+1;
    float4 interpolator = getVerticalInterpolator_device(src, x, y, width, smallheight);
    //this interpolator is valid on interval [0, 1] representing [y, y+1]
    //we are Center so we are interested in values: 0.25 and 0.75
    if (x < width){
        if (y != -1 && 2*y+1 < height) dst[(2*y +1)*width + x] = getCubicHermitSpline(interpolator, 0.25f);
        if (2*y+2 < height) dst[(2*y+2)*width + x] = getCubicHermitSpline(interpolator, 0.75f);
    }
}

//block y should range from 0 to smallheight-1 INCLUDED
//dst of size width*height while src is of size width*smallheight
[shader("compute")]
[numthreads(16, 16, 1)]
void bicubicVerticalTopUpscaleX2(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform float* src, uniform uint64_t width, uniform uint64_t height){
    const int64_t x = (int64_t)(threadIdx.x + 16*blockIdx.x);
    const int64_t y = (int64_t)(threadIdx.y + 16*blockIdx.y);
    const int64_t smallheight = (height-1)/2+1;
    float4 interpolator = getVerticalInterpolator_device(src, x, y, width, smallheight);
    //this interpolator is valid on interval [0, 1] representing [y, y+1]
    //we are Top so we are interested in values: 0 and 0.5
    if (x < width){
        if (2*y < height) dst[(2*y)*width + x] = src[y*width + x];
        if (2*y+1 < height) dst[(2*y+1)*width + x] = getCubicHermitSpline(interpolator, 0.5f);
    }
}

//block y should range from 0 to smallheight INCLUDED
//dst of size width*height while src is of size width*smallheight
[shader("compute")]
[numthreads(16, 16, 1)]
void bicubicVerticalCenterUpscaleX4(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform float* src, uniform uint64_t width, uniform uint64_t height){
    const int64_t x = (int64_t)(threadIdx.x + 16*blockIdx.x);
    int64_t y = (int64_t)(threadIdx.y + 16*blockIdx.y);
    y--;
    const int64_t smallheight = (height-1)/4+1;
    float4 interpolator = getVerticalInterpolator_device(src, x, y, width, smallheight);
    //this interpolator is valid on interval [0, 1] representing [y, y+1]
    //we are Center so we are interested in values: 0.25 and 0.75
    if (x < width){
        if (y != -1 && 4*y+2 < height) dst[(4*y +2)*width + x] = getCubicHermitSpline(interpolator, 0.125f);
        if (y != -1 && 4*y+3 < height) dst[(4*y +3)*width + x] = getCubicHermitSpline(interpolator, 0.375f);
        if (4*y+4 < height) dst[(4*y+4)*width + x] = getCubicHermitSpline(interpolator, 0.625f);
        if (4*y+5 < height) dst[(4*y+5)*width + x] = getCubicHermitSpline(interpolator, 0.875f);
    }
}

//block y should range from 0 to smallheight-1 INCLUDED
//dst of size width*height while src is of size width*smallheight
[shader("compute")]
[numthreads(16, 16, 1)]
void bicubicVerticalTopUpscaleX4(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform float* src, uniform uint64_t width, uniform uint64_t height){
    const int64_t x = (int64_t)(threadIdx.x + 16*blockIdx.x);
    const int64_t y = (int64_t)(threadIdx.y + 16*blockIdx.y);
    const int64_t smallheight = (height-1)/4+1;
    float4 interpolator = getVerticalInterpolator_device(src, x, y, width, smallheight);
    //this interpolator is valid on interval [0, 1] representing [y, y+1]
    //we are Top so we are interested in values: 0 and 0.5
    if (x < width){
        if (4*y < height) dst[(4*y)*width + x] = src[y*width + x];
        if (4*y+1 < height) dst[(4*y+1)*width + x] = getCubicHermitSpline(interpolator, 0.25f);
        if (4*y+2 < height) dst[(4*y+2)*width + x] = getCubicHermitSpline(interpolator, 0.5f);
        if (4*y+3 < height) dst[(4*y+3)*width + x] = getCubicHermitSpline(interpolator, 0.75f);
    }
}
