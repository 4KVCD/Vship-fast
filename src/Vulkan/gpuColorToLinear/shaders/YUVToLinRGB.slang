import vshipColorModule;
import inverseTransferModule;
import vshipBicubicModule;
import anyDepthToFloat;

float3 ncl_yuv_to_rgb_from_kr_kb(float3 val, const float Kr, const float Kb){
    const float Kg = 1.f-Kb-Kr;
    const float uscale = 0.5f/(1.f-Kb);
    const float vscale = 0.5f/(1.f-Kr);

    float3 res;
    res.x = (val.z + val.x * vscale)/(vscale*Kr + 0.5f);
    res.z = (val.y + val.x * uscale)/(uscale*Kb + 0.5f);
    res.y = (val.x - Kr*res.x - Kb*res.z)/Kg;

    return res;
}

float3 inverseMatrixNCL(float3 val, Vship_YUVMatrix_t matrix){
    switch (matrix){
        case Vship_YUVMatrix_t.Vship_MATRIX_BT2100_ICTCP:{
            //at this point, we have ICtCp
            //ICtCp => L'M'S'
            //L' = I + 0.0086090370379327566f* Ct + 0.11102962500302595655f* Cp
            val.x = fma(val.y, 0.0086090370379327566f, fma(val.z, 0.11102962500302595655f, val.x));
            //M' = I - 0.0086090370379327566f* Ct - 0.11102962500302595655* Cp
            //=> M' = L' - 0.01721807407586551* Ct - 0.22205925000605192* Cp
            val.y = fma(val.y, -0.01721807407586551f, fma(val.z, -0.22205925000605192f, val.x));
            //S' = I + 0.560031335710679118f* Ct - 0.32062717498731885184f* Cp
            //we have Ct = (L' - M' - 0.22205925000605192* Cp)/0.01721807407586551
            //and I = 0.5L' + 0.5M'
            //=> S' = 33.025782688766114 * L' - 32.025782688766114 * M' - 7.543278084714549 * Cp
            val.z = fma(val.x, 33.025782688766114f, fma(val.y, -32.025782688766114f, -val.z*7.543278084714549f));

            return val;
        }
        case Vship_YUVMatrix_t.Vship_MATRIX_BT2020_CL:{
            const float Kr = 0.2627f;
            const float Kb = 0.0593f;
            //const float Kg = 1.f - Kr - Kb;
            //we got YcCbcCrc right now
            const float uscale = 0.5f/(1.f-Kb);
            const float vscale = 0.5f/(1.f-Kr);

            float3 res;
            res.x = (val.z + val.x * vscale)/(vscale*Kr + 0.5f);
            res.z = (val.y + val.x * uscale)/(uscale*Kb + 0.5f);
            res.y = val.x;

            //then we apply the inverse of transfer function to go from Yc'B'R' -> YcBR
            return res;
        }
        case Vship_YUVMatrix_t.Vship_MATRIX_BT2020_NCL:{
            const float Kr = 0.2627f;
            const float Kb = 0.0593f;

            return ncl_yuv_to_rgb_from_kr_kb(val, Kr, Kb);
        }
        case Vship_YUVMatrix_t.Vship_MATRIX_BT709:{
            const float Kr = 0.2126f;
            const float Kb = 0.0722f;

            return ncl_yuv_to_rgb_from_kr_kb(val, Kr, Kb);
        }
        case Vship_YUVMatrix_t.Vship_MATRIX_ST170_M:
        case Vship_YUVMatrix_t.Vship_MATRIX_BT470_BG:{
            const float Kr = 0.299f;
            const float Kb = 0.114f;

            return ncl_yuv_to_rgb_from_kr_kb(val, Kr, Kb);
        }
        case Vship_YUVMatrix_t.Vship_MATRIX_YCGCO:{
            float3 out;
            out.x = val.x + val.y - val.z;
            out.y = val.x + val.z;
            out.z = val.x - val.y - val.z;
            return out;
        }
        case Vship_YUVMatrix_t.Vship_MATRIX_YCGCO_RE:
        case Vship_YUVMatrix_t.Vship_MATRIX_YCGCO_RO:{
            float3 out;
            out.x = val.x + 0.5f*(val.y - val.z);
            out.y = val.x + 0.5f*val.z;
            out.z = val.x - 0.5f*(val.y + val.z);
            return out;
        }
        case Vship_YUVMatrix_t.Vship_MATRIX_RGB:
        default:{ //unused
            return val;
        }
    }
}

float3 inverseMatrixCL(float3 val, Vship_YUVMatrix_t matrix){
    switch (matrix){
        case Vship_YUVMatrix_t.Vship_MATRIX_BT2100_ICTCP:{
            //At that point, we got LMS
            //R = 3.4366066943330784267*L - 2.5064521186562698975*M + 0.069845424323191470954*S
            val.x = fma(val.x, 3.4366066943330784267f, fma(val.y, -2.5064521186562698975f, val.z*0.069845424323191470954f));
            //G = -0.79132955559892875305*L + 1.9836004517922907339*M -0.19227089619336198096*S
            //or L = 0.290984709320676*R + 0.7293392411733872*M + 0.020323950494063145*S
            //=> G = -0.23026480071281397*R + 1.4064527541936944*M + -0.2083538389058436*S
            val.y = fma(val.x, -0.23026480071281397f, fma(val.y, 1.4064527541936944f, -val.z*0.2083538389058436f));
            //B = -0.025949899690592673413*L + -0.098913714711726441685*M + 1.1248636144023191151*S
            //or M = 0.16372025297417298*R + 0.7110085973512066*G + 0.14814137075318312*S
            //=> L = 0.4103923143895743*R + 0.5185664708598834*G + 0.12836926542557514*S
            //=> B = -0.02684381778741865*R + -0.08378524945770065*G + 1.106879231557486*S
            val.z = fma(val.x, -0.02684381778741865f, fma(val.y, -0.08378524945770065f, val.z*1.106879231557486));
            return val;
        }
        case Vship_YUVMatrix_t.Vship_MATRIX_BT2020_CL:{
            const float Kr = 0.2627f;
            const float Kb = 0.0593f;
            const float Kg = 1.f - Kr - Kb;

            val.y = (val.y - Kr*val.x - Kb*val.z)/Kg;

            return val;
        }
        case Vship_YUVMatrix_t.Vship_MATRIX_YCGCO:
        case Vship_YUVMatrix_t.Vship_MATRIX_YCGCO_RE:
        case Vship_YUVMatrix_t.Vship_MATRIX_YCGCO_RO:
        case Vship_YUVMatrix_t.Vship_MATRIX_BT2020_NCL:
        case Vship_YUVMatrix_t.Vship_MATRIX_BT709:
        case Vship_YUVMatrix_t.Vship_MATRIX_BT470_BG:
        case Vship_YUVMatrix_t.Vship_MATRIX_ST170_M:
        case Vship_YUVMatrix_t.Vship_MATRIX_RGB:
        default:{ //unused
            return val;
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void YUVToLinRGB(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* p0, uniform float* p1, uniform float* p2, uniform uint64_t width, uniform Vship_YUVMatrix_t matrix, uniform Vship_TransferFunction_t transfer, uniform int libjxlArithmetic){
    const uint64_t x = threadIdx.x + 256*blockIdx.x;
    if (x >= width) return;

    float3 val = {p0[x], p1[x], p2[x]};

    val = inverseMatrixNCL(val, matrix);
    val = inverseTransfer(val, transfer, libjxlArithmetic != 0);
    val = inverseMatrixCL(val, matrix);

    p0[x] = val.x;
    p1[x] = val.y;
    p2[x] = val.z;
}

// The vertical upsampling of the chroma that YUVToLinRGBUpsampled does for each pixel, as the
// bicubicVertical*Upscale* kernels of chromaUpsample.slang do it (chromaVerticalMode in chromaUpsample.hpp)
static const int VERTICAL_NONE = 0;      // the chroma has the frame's rows
static const int VERTICAL_CENTER_X2 = 1;
static const int VERTICAL_TOP_X2 = 2;
static const int VERTICAL_CENTER_X4 = 3;
static const int VERTICAL_TOP_X4 = 4;

// The chroma of row r of the frame at column x, src holding the chroma at full width
float upsampledChroma(float* src, int64_t x, int64_t r, int64_t width, int64_t height, int verticalMode){
    if (verticalMode == VERTICAL_CENTER_X2){
        //bicubicVerticalCenterUpscaleX2: rows 2y+1 and 2y+2 at 0.25 and 0.75 of [y, y+1]
        const int64_t smallheight = (height-1)/2+1;
        return getCubicHermitSpline(getVerticalInterpolator_device(src, x, (r-1) >> 1, width, smallheight), ((r & 1) == 1) ? 0.25f : 0.75f);
    }
    if (verticalMode == VERTICAL_TOP_X2){
        //bicubicVerticalTopUpscaleX2: row 2y is row y, row 2y+1 at 0.5 of [y, y+1]
        const int64_t smallheight = (height-1)/2+1;
        if ((r & 1) == 0) return src[(r >> 1)*width + x];
        return getCubicHermitSpline(getVerticalInterpolator_device(src, x, r >> 1, width, smallheight), 0.5f);
    }
    if (verticalMode == VERTICAL_CENTER_X4){
        //bicubicVerticalCenterUpscaleX4: rows 4y+2 to 4y+5 at 0.125, 0.375, 0.625 and 0.875 of [y, y+1]
        const int64_t smallheight = (height-1)/4+1;
        const int phase = (int)((r-2) & 3);
        const float t = (phase == 0) ? 0.125f : ((phase == 1) ? 0.375f : ((phase == 2) ? 0.625f : 0.875f));
        return getCubicHermitSpline(getVerticalInterpolator_device(src, x, (r-2) >> 2, width, smallheight), t);
    }
    if (verticalMode == VERTICAL_TOP_X4){
        //bicubicVerticalTopUpscaleX4: row 4y is row y, rows 4y+1 to 4y+3 at 0.25, 0.5 and 0.75 of [y, y+1]
        const int64_t smallheight = (height-1)/4+1;
        const int phase = (int)(r & 3);
        if (phase == 0) return src[(r >> 2)*width + x];
        const float t = (phase == 1) ? 0.25f : ((phase == 2) ? 0.5f : 0.75f);
        return getCubicHermitSpline(getVerticalInterpolator_device(src, x, r >> 2, width, smallheight), t);
    }
    return src[r*width + x];
}

// Returns x. Its bits go through an OR with a zero only known at run time, so that the compiler cannot fuse the
// operations that produced x with those that use it: the upsampled chroma used to go through memory.
float unfused(float x, uint zero){
    return asfloat(asuint(x) | zero);
}

// The luma of pixel i: from the uploaded integer plane rawLuma, converted as convertToFloatPlane would (the float
// went through memory, so it goes through unfused()), or from the float plane luma when rawLuma is null.
float lumaAt(float* luma, uint8_t* rawLuma, int rawLumaFormat, float rawLumaScale, int64_t i, int64_t width, uint zero){
    if (rawLuma == nullptr) return luma[i];
    return unfused(rawLumaSample(rawLuma, (uint64_t)i, (uint64_t)width, rawLumaFormat, rawLumaScale), zero);
}

// YUVToLinRGB with the chroma's vertical upsampling done for each pixel instead of in a pass of its own: the
// chroma comes from chroma1 and chroma2 (full width, verticalMode says which rows), the luma from p0 or rawLuma
// (lumaAt), and the linear RGB goes to p0, p1 and p2. chroma1 and chroma2 may be p1 and p2 when verticalMode is
// VERTICAL_NONE. One thread per pixel, blockIdx.y the row.
[shader("compute")]
[numthreads(256, 1, 1)]
void YUVToLinRGBUpsampled(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* p0, uniform float* p1, uniform float* p2, uniform float* chroma1, uniform float* chroma2, uniform uint64_t width64, uniform uint64_t height64, uniform int verticalMode, uniform Vship_YUVMatrix_t matrix, uniform Vship_TransferFunction_t transfer, uniform int libjxlArithmetic, uniform uint zero, uniform uint8_t* rawLuma, uniform int rawLumaFormat, uniform float rawLumaScale){
    const int64_t width = (int64_t)width64;
    const int64_t height = (int64_t)height64;
    const int64_t x = threadIdx.x + 256*blockIdx.x;
    const int64_t r = blockIdx.y;
    if (x >= width) return;
    const int64_t i = r*width + x;

    float3 val = {lumaAt(p0, rawLuma, rawLumaFormat, rawLumaScale, i, width, zero), unfused(upsampledChroma(chroma1, x, r, width, height, verticalMode), zero), unfused(upsampledChroma(chroma2, x, r, width, height, verticalMode), zero)};

    val = inverseMatrixNCL(val, matrix);
    val = inverseTransfer(val, transfer, libjxlArithmetic != 0);
    val = inverseMatrixCL(val, matrix);

    p0[i] = val.x;
    p1[i] = val.y;
    p2[i] = val.z;
}