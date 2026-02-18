module vshipBicubicModule;

public float4 loadCubicHermitSpline(float p0, float m0, float p1, float m1){
    float4 res;
    res.x = 2*p0 + m0 - 2*p1 + m1;
    res.y = -3*p0 + 3*p1 - 2*m0 - m1;
    res.z = m0;
    res.w = p0;
    return res;
}

public float getCubicHermitSpline(float4 loaded, float t){
    float res = loaded.x;
    res *= t;
    res += loaded.y;
    res *= t;
    res += loaded.z;
    res *= t;
    res += loaded.w;
    return res;
}

public float4 getHorizontalInterpolator_device(float* src, int64_t x, int64_t y, int64_t width, int64_t height){ //width and height must be the one of source!!!!
    y = min(y, height-1);
    x = min(x, width-1); //can be -1

    float elm1 = (x <= 0) ? src[y*width] : src[y*width+x-1];
    float el0 = (x < 0) ? elm1 : src[y*width+x];
    float el1 = (x >= width-1) ? el0 : src[y*width+x+1];
    float el2 = (x >= width-2) ? el1 : src[y*width+x+2];

    return loadCubicHermitSpline(el0, (el1 - elm1)/2, el1, (el2 - el0)/2);
}

public float4 getVerticalInterpolator_device(float* src, int64_t x, int64_t y, int64_t width, int64_t height){ //width and height must be the one of source!!!!
    y = min(y, height-1); //can be equal to -1
    x = min(x, width-1);

    float elm1 = (y <= 0) ? src[x] : src[(y-1)*width+x];
    float el0 = (y < 0) ? elm1 : src[y*width+x];
    float el1 = (y >= height-1) ? el0 : src[(y+1)*width+x];
    float el2 = (y >= height-2) ? el1 : src[(y+2)*width+x];
    
    return loadCubicHermitSpline(el0, (el1 - elm1)/2, el1, (el2 - el0)/2);
}