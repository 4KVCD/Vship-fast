#pragma once

namespace cvvdp{

//combine both
__device__ inline void xyz_to_DKLd65(float3& a){
    float3 out;

    out.x = fmaf(0.05419884, a.x, fmaf(0.99067443, a.y, 0.00811786*a.z));
    out.y = fmaf(0.4958951, a.x, fmaf(-0.35200799, a.y, -0.10612317*a.z));
    out.z = fmaf(-0.04174099, a.x, fmaf(-1.01835504, a.y, 0.98119617*a.z));

    a = out;
}

//we receive the linear input from the converter but we encode to display depending on what it was before
template<Vship_TransferFunction_t transferFunction>
__device__ void inline displayEncode_Device(float& a, float Y_peak, float Y_black, float Y_refl, float exposure);

template<>
__device__ void inline displayEncode_Device<Vship_TRC_BT709>(float& a, float Y_peak, float Y_black, float Y_refl, float exposure){
    a = (Y_peak - Y_black) * (max(0.f, min(1.f, a*exposure))) + Y_black + Y_refl;
}

template<>
__device__ void inline displayEncode_Device<Vship_TRC_PQ>(float& a, float Y_peak, float Y_black, float Y_refl, float exposure){
    a *= 100.f; //convert the Y normalized to 100 to actual PQ luminosity values
    a = max(max(0.005f, Y_black), min(Y_peak, a*exposure)) + Y_black + Y_refl;
}

template<Vship_Primaries_t Source_Primary, Vship_TransferFunction_t transferFunction>
__global__ void XYZ_to_dkl_DisplayEncode_Kernel(float* p1, float* p2, float* p3, float Y_peak, float Y_black, float Y_refl, float exposure, int64_t width){
    const int64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= width) return;

    float3 src = {p1[x], p2[x], p3[x]};

    displayEncode_Device<transferFunction>(src.x, Y_peak, Y_black, Y_refl, exposure);
    displayEncode_Device<transferFunction>(src.y, Y_peak, Y_black, Y_refl, exposure);
    displayEncode_Device<transferFunction>(src.z, Y_peak, Y_black, Y_refl, exposure);
    src = VshipColorConvert::primariesToPrimaries_device<Source_Primary, Vship_PRIMARIES_INTERNAL>(src);
    xyz_to_DKLd65(src);

    //if (x == 0) printf("value at DKL / Linrgb %f %f %f %f %f %f\n", src.x, src.y, src.z, p1[x], p2[x], p3[x]);

    p1[x] = src.x; p2[x] = src.y; p3[x] = src.z;
}

template<Vship_TransferFunction_t transferFunction>
void inline XYZ_to_dkl_DisplayEncode2(float* src_d[3], Vship_Primaries_t Source_Primary, float Y_peak, float Y_black, float Y_refl, float exposure, int64_t width, hipStream_t stream){
    int th_x = 256;
    int64_t bl_x = (width+th_x-1)/th_x;
    switch (Source_Primary){
        case Vship_PRIMARIES_INTERNAL:
            XYZ_to_dkl_DisplayEncode_Kernel<Vship_PRIMARIES_INTERNAL, transferFunction><<<dim3(bl_x), dim3(th_x), 0, stream>>>(src_d[0], src_d[1], src_d[2], Y_peak, Y_black, Y_refl, exposure, width);
            break;
        case Vship_PRIMARIES_BT709:
            XYZ_to_dkl_DisplayEncode_Kernel<Vship_PRIMARIES_BT709, transferFunction><<<dim3(bl_x), dim3(th_x), 0, stream>>>(src_d[0], src_d[1], src_d[2], Y_peak, Y_black, Y_refl, exposure, width);
            break;
        case Vship_PRIMARIES_BT470_BG:
            XYZ_to_dkl_DisplayEncode_Kernel<Vship_PRIMARIES_BT470_BG, transferFunction><<<dim3(bl_x), dim3(th_x), 0, stream>>>(src_d[0], src_d[1], src_d[2], Y_peak, Y_black, Y_refl, exposure, width);
            break;
        case Vship_PRIMARIES_BT470_M:
            XYZ_to_dkl_DisplayEncode_Kernel<Vship_PRIMARIES_BT470_M, transferFunction><<<dim3(bl_x), dim3(th_x), 0, stream>>>(src_d[0], src_d[1], src_d[2], Y_peak, Y_black, Y_refl, exposure, width);
            break;
        case Vship_PRIMARIES_ST170_M:
            XYZ_to_dkl_DisplayEncode_Kernel<Vship_PRIMARIES_ST170_M, transferFunction><<<dim3(bl_x), dim3(th_x), 0, stream>>>(src_d[0], src_d[1], src_d[2], Y_peak, Y_black, Y_refl, exposure, width);
            break;
        case Vship_PRIMARIES_ST240_M:
            XYZ_to_dkl_DisplayEncode_Kernel<Vship_PRIMARIES_ST240_M, transferFunction><<<dim3(bl_x), dim3(th_x), 0, stream>>>(src_d[0], src_d[1], src_d[2], Y_peak, Y_black, Y_refl, exposure, width);
            break;
        case Vship_PRIMARIES_BT2020:
            XYZ_to_dkl_DisplayEncode_Kernel<Vship_PRIMARIES_BT2020, transferFunction><<<dim3(bl_x), dim3(th_x), 0, stream>>>(src_d[0], src_d[1], src_d[2], Y_peak, Y_black, Y_refl, exposure, width);
            break;
        case Vship_PRIMARIES_DisplayP3:
            XYZ_to_dkl_DisplayEncode_Kernel<Vship_PRIMARIES_DisplayP3, transferFunction><<<dim3(bl_x), dim3(th_x), 0, stream>>>(src_d[0], src_d[1], src_d[2], Y_peak, Y_black, Y_refl, exposure, width);
            break;
    }
    GPU_CHECK(hipGetLastError());
}

void inline XYZ_to_dkl_DisplayEncode(float* src_d[3], Vship_Primaries_t Source_Primary, Vship_TransferFunction_t transferFunction, float Y_peak, float Y_black, float Y_refl, float exposure, DisplayColorspace isHDR, int64_t width, hipStream_t stream){
    if (!isHDR){
        transferFunction = Vship_TRC_BT709;
        //we supppose that the input is sRGB no matter the actual input since the screen will display it as is
    }
    switch (transferFunction){
        case Vship_TRC_PQ:
            XYZ_to_dkl_DisplayEncode2<Vship_TRC_PQ>(src_d, Source_Primary, Y_peak, Y_black, Y_refl, exposure, width, stream);
            break;
        //treated same as BT709/sRGB
        case Vship_TRC_Linear:
        case Vship_TRC_HLG:
        default:
            XYZ_to_dkl_DisplayEncode2<Vship_TRC_BT709>(src_d, Source_Primary, Y_peak, Y_black, Y_refl, exposure, width, stream);
            break;
    }
}


}
