import "../../gpuColorToLinear/shaders/vshipColorModule";
import "../../gpuColorToLinear/shaders/primaries";
import "../../gpuColorToLinear/shaders/inverseTransferModule";
import "../../gpuColorToLinear/shaders/YUVToLinRGB";

float3 xyz_to_LMS2006(float3 a){
    float3 out;

    out.x = fma(0.187596268556126, a.x, fma(0.585168649077728, a.y, -0.026384263306304*a.z));
    out.y = fma(-0.133397430663221, a.x, fma(0.405505777260049, a.y, 0.034502127690364*a.z));
    out.z = fma(0.000244379021663, a.x, fma(-0.000542995890619, a.y, 0.019406849066323*a.z));

    return out;
}

float3 LMS2006_to_DKLd65(float3 a){
    //a = x+y
    a.x = a.x + a.y;
    //b = x -2.311130179947035y = a -3.311130179947035y
    a.y = a.x - 3.311130179947035*a.y;
    //c = -x-y + 50.977571328718781z = -a + 50.977571328718781z
    a.z = 50.977571328718781*a.z - a.x;
    return a;
}

// pycvvdp's display model: the light the display emits for the linear value v, its reflections included
float displayEncodeBT709(float v, float Y_peak, float Y_black, float Y_refl, float exposure){
    return (Y_peak - Y_black) * (max(0.f, min(1.f, v*exposure))) + Y_black + Y_refl;
}

// the same for a PQ display, v being normalised to 100 cd/m2
float displayEncodePQ(float v, float Y_peak, float Y_black, float Y_refl, float exposure){
    return max(max(0.005f, Y_black), min(Y_peak, v*100.f*exposure)) + Y_black + Y_refl;
}

// One image's linear RGB planes in1..in3 to the DKL planes out1..out3 CVVDP works on: the display model
// (displayEncodePQ when pq, else displayEncodeBT709), the image's primaries to XYZ, then XYZ to DKL. One pass over
// memory for what were 5 kernels and a copy; in and out may be the same planes.
[shader("compute")]
[numthreads(256, 1, 1)]
void displayToDKL(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* in1, uniform float* in2, uniform float* in3, uniform float* out1, uniform float* out2, uniform float* out3, uniform int64_t size, uniform float Y_peak, uniform float Y_black, uniform float Y_refl, uniform float exposure, uniform int pq, uniform Vship_Primaries_t primaries){
    const int64_t x = threadIdx.x + blockIdx.x * 256;
    if (x >= size) return;

    float3 v = {in1[x], in2[x], in3[x]};
    [ForceUnroll]
    for (int i = 0; i < 3; i++){
        v[i] = (pq != 0) ? displayEncodePQ(v[i], Y_peak, Y_black, Y_refl, exposure) : displayEncodeBT709(v[i], Y_peak, Y_black, Y_refl, exposure);
    }
    v = primariesToXYZ(v, primaries);
    v = LMS2006_to_DKLd65(xyz_to_LMS2006(v));

    out1[x] = v.x; out2[x] = v.y; out3[x] = v.z;
}

// The converter's last step (YUVToLinRGBUpsampled: the chroma's vertical upsampling and YUV to linear RGB) and
// displayToDKL in one pass over memory: the luma plane and the horizontally upsampled chroma
// (Converter::placeChromaForPixelConversion) to the DKL planes out1..out3. The linear RGB goes through an OR with a
// runtime zero, as it went through memory before, so that the compiler cannot fuse the two steps' arithmetic.
// displayToDKL's code is repeated as it is written there: the driver compiled it to different roundings when it
// was a function shared by the two kernels. One thread per pixel, blockIdx.y the row.
[shader("compute")]
[numthreads(256, 1, 1)]
void yuvToDKL(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* luma, uniform float* chroma1, uniform float* chroma2, uniform float* out1, uniform float* out2, uniform float* out3, uniform uint64_t width64, uniform uint64_t height64, uniform int verticalMode, uniform Vship_YUVMatrix_t matrix, uniform Vship_TransferFunction_t transfer, uniform int libjxlArithmetic, uniform float Y_peak, uniform float Y_black, uniform float Y_refl, uniform float exposure, uniform int pq, uniform Vship_Primaries_t primaries, uniform uint zero){
    const int64_t width = (int64_t)width64;
    const int64_t height = (int64_t)height64;
    const int64_t x = threadIdx.x + 256*blockIdx.x;
    const int64_t r = blockIdx.y;
    if (x >= width) return;
    const int64_t i = r*width + x;

    float3 val = {luma[i], unfused(upsampledChroma(chroma1, x, r, width, height, verticalMode), zero), unfused(upsampledChroma(chroma2, x, r, width, height, verticalMode), zero)};
    val = inverseMatrixNCL(val, matrix);
    val = inverseTransfer(val, transfer, libjxlArithmetic != 0);
    val = inverseMatrixCL(val, matrix);

    float3 v = {unfused(val.x, zero), unfused(val.y, zero), unfused(val.z, zero)};
    [ForceUnroll]
    for (int c = 0; c < 3; c++){
        v[c] = (pq != 0) ? displayEncodePQ(v[c], Y_peak, Y_black, Y_refl, exposure) : displayEncodeBT709(v[c], Y_peak, Y_black, Y_refl, exposure);
    }
    v = primariesToXYZ(v, primaries);
    v = LMS2006_to_DKLd65(xyz_to_LMS2006(v));

    out1[i] = v.x; out2[i] = v.y; out3[i] = v.z;
}
