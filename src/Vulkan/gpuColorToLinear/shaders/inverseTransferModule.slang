module inverseTransferModule;

import vshipColorModule;

float gamma_to_linrgbfunc(float a, float gamma){
    if (a < 0){
        return -pow(-a, gamma);
    } else {
        return pow(a, gamma);
    }
}

//default here will take linear and hlg since they are treated before
float inverseTransferMono(float a, Vship_TransferFunction_t transfer){
    switch (transfer){
        case Vship_TransferFunction_t.Vship_TRC_sRGB:{
            if (a < 0){
                if (a < -0.04045f){
                    a = -pow(((-a+0.055)*(1.0/1.055)), 2.4f);
                } else {
                    a *= 1.0/12.92;
                }
            } else {
                if (a > 0.04045f){
                    a = pow(((a+0.055)*(1.0/1.055)), 2.4f);
                } else {
                    a *= 1.0/12.92;
                }
            }
            return a;
        }
        case Vship_TransferFunction_t.Vship_TRC_BT709:{
            return gamma_to_linrgbfunc(a, 2.4);
        }
        case Vship_TransferFunction_t.Vship_TRC_BT470_M:{
            return gamma_to_linrgbfunc(a, 2.2);
        }
        case Vship_TransferFunction_t.Vship_TRC_BT470_BG:{
            return gamma_to_linrgbfunc(a, 2.6);
        }
        case Vship_TransferFunction_t.Vship_TRC_BT601:{
            return gamma_to_linrgbfunc(a, 2.8);
        }
        case Vship_TransferFunction_t.Vship_TRC_ST240_M:{
            if (a >= 0){
                if (a < 0.0228){
                    return 4*a;
                } else {
                    return 1.1115f*pow(a, 0.45f) - 0.1115;
                }
            } else {
                if (a > -0.0228){
                    return 4*a;
                } else {
                    return -1.1115f*pow(-a, 0.45f) + 0.1115;
                }
            }
        }
        case Vship_TransferFunction_t.Vship_TRC_ST428:{
            return 52.37f*gamma_to_linrgbfunc(a, 2.6f)/48.f;
        }
        case Vship_TransferFunction_t.Vship_TRC_PQ:{
            const float Lmax = 100.f;
            const float n    = 0.15930175781250000f;
            const float m    = 78.843750000000000f;
            const float c1   = 0.83593750000000000f;
            const float c2   = 18.851562500000000f;
            const float c3   = 18.687500000000000f;

            a = max(0.f, a);
            a = pow(a, 1.f/m);
            return Lmax * pow(max(0.f, (a-c1))/(c2 - c3*a), 1.f/n);
        }
        default:{ //unused but for the compiler
            return a;
        }
    }
}

public float3 inverseTransfer(float3 val, Vship_TransferFunction_t transfer){
    switch(transfer){
        case Vship_TransferFunction_t.Vship_TRC_Linear:{
            return val;
        }
        case Vship_TransferFunction_t.Vship_TRC_HLG:{
            const float a = 0.17883277f;
            const float b = 1.f - 4.f * a;
            //const float c = 0.5f - a * logf(4*a); C++ doesnt accept constexpr logf
            const float c = 0.5f - a * -0.3350097945111627f;
            const float gamma = 1.2;

            //inverse OETF
            if (val.x <= 0.5f){
                val.x = val.x*val.x / 3.f;
            } else {
                val.x = (exp((val.x - c)/a)+b)/12.f;
            }
            if (val.y <= 0.5f){
                val.y = val.y*val.y / 3.f;
            } else {
                val.y = (exp((val.y - c)/a)+b)/12.f;
            }
            if (val.z <= 0.5f){
                val.z = val.z*val.z / 3.f;
            } else {
                val.z = (exp((val.z - c)/a)+b)/12.f;
            }

            //OOTF
            const float Ys = 0.2627f*val.x + 0.6780f*val.y + 0.0593*val.z;
            val = val * pow(Ys, gamma-1.f);
            return val;
        }
        default:{
            val.x = inverseTransferMono(val.x, transfer);
            val.y = inverseTransferMono(val.y, transfer);
            val.z = inverseTransferMono(val.z, transfer);
            return val;
        }
    }
}