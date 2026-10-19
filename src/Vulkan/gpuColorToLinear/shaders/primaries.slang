import vshipColorModule;

//https://fr.mathworks.com/help/images/ref/whitepoint.html
//https://en.wikipedia.org/wiki/NTSC
//http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
//http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
//https://en.wikipedia.org/wiki/PAL
//https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2407-2017-PDF-E.pdf
float3 primariesToXYZ(float3 a, Vship_Primaries_t primarySource){
    float3 res;
    switch (primarySource){
        case Vship_Primaries_t.Vship_PRIMARIES_ST170_M:
        case Vship_Primaries_t.Vship_PRIMARIES_ST240_M:{
            res.x = fma(a.x, 0.39349893f, fma(a.y, 0.3652766f, a.z*0.19162447f));
            res.y = fma(a.x, 0.2123645f, fma(a.y, 0.70109542f, a.z*0.08654008f));
            res.z = fma(a.x, 0.01873804f, fma(a.y, 0.1119396f, a.z*0.95812235f));
            return res;
        }
        case Vship_Primaries_t.Vship_PRIMARIES_BT470_BG:{
            res.x = fma(a.x, 0.4305f, fma(a.y, 0.3416f, a.z*0.1783f));
            res.y = fma(a.x, 0.2220f, fma(a.y, 0.7067f, a.z*0.07132f));
            res.z = fma(a.x, 0.0202f, fma(a.y, 0.1296f, a.z*0.9391f));
            return res;
        }
        case Vship_Primaries_t.Vship_PRIMARIES_BT470_M:{
            res.x = fma(a.x, 0.6068f, fma(a.y, 0.1735f, a.z*0.2003f));
            res.y = fma(a.x, 0.2989f, fma(a.y, 0.5866f, a.z*0.1145f));
            res.z = fma(a.x, 0.f, fma(a.y, 0.06609f, a.z*1.116f));
            return res;
        }
        case Vship_Primaries_t.Vship_PRIMARIES_BT709:{
            //sRGB's matrix with the precision pycvvdp uses (its color_spaces.json, from brucelindbloom.com)
            res.x = fma(a.x, 0.4124564f, fma(a.y, 0.3575761f, a.z*0.1804375f));
            res.y = fma(a.x, 0.2126729f, fma(a.y, 0.7151522f, a.z*0.072175f));
            res.z = fma(a.x, 0.0193339f, fma(a.y, 0.119192f, a.z*0.9503041f));
            return res;
        }
        case Vship_Primaries_t.Vship_PRIMARIES_BT2020:{
            res.x = fma(a.x, 0.6370f, fma(a.y, 0.1446f, a.z*0.1689f));
            res.y = fma(a.x, 0.2627f, fma(a.y, 0.6780f, a.z*0.0593f));
            res.z = fma(a.x, 0.f, fma(a.y, 0.0281f, a.z*1.0610f));
            return res;
        }
        case Vship_Primaries_t.Vship_PRIMARIES_DisplayP3:{
            res.x = fma(a.x, 0.48655f, fma(a.y, 0.26578f, a.z*0.198168f));
            res.y = fma(a.x, 0.22897f, fma(a.y, 0.69177f, a.z*0.0792672f));
            res.z = fma(a.x, 0.f, fma(a.y, 0.0451152f, a.z*1.04368f));
            return res;
        }
        case Vship_Primaries_t.Vship_PRIMARIES_INTERNAL:
        default:{
            return a;
        }
    }
}

float3 primariesFromXYZ(float3 a, Vship_Primaries_t primaryDst){
    float3 res;
    switch (primaryDst){
        case Vship_Primaries_t.Vship_PRIMARIES_ST170_M:
        case Vship_Primaries_t.Vship_PRIMARIES_ST240_M:{
            res.x = fma(a.x, 3.5061991f, fma(a.y, -1.7398879f, -a.z*0.54408866f));
            res.y = fma(a.x, -1.06899334f, fma(a.y, 1.97767857f, a.z*0.03516964f));
            res.z = fma(a.x, 0.05632201f, fma(a.y, -0.1970296f, a.z*1.05023986f));
            return res;
        }
        case Vship_Primaries_t.Vship_PRIMARIES_BT470_BG:{
            res.x = fma(a.x, 3.064f, fma(a.y, -1.3935f, -a.z*0.4758f));
            res.y = fma(a.x, -0.9692f, fma(a.y, 1.876f, a.z*0.04155f));
            res.z = fma(a.x, 0.06788f, fma(a.y, -0.2289f, a.z*1.069f));
            return res;
        }
        case Vship_Primaries_t.Vship_PRIMARIES_BT470_M:{
            res.x = fma(a.x, 1.910f, fma(a.y, -0.5324f, -a.z*0.2882f));
            res.y = fma(a.x, -0.9846f, fma(a.y, 1.999f, -a.z*0.02831f));
            res.z = fma(a.x, 0.05831f, fma(a.y, -0.1184f, a.z*0.8976f));
            return res;
        }
        case Vship_Primaries_t.Vship_PRIMARIES_BT709:{
            res.x = fma(a.x, 3.2410f, fma(a.y, -1.5374f, -a.z*0.4986f));
            res.y = fma(a.x, -0.9692f, fma(a.y, 1.8760f, a.z*0.0416f));
            res.z = fma(a.x, 0.0556f, fma(a.y, -0.2040f, a.z*1.0570f));
            return res;
        }
        case Vship_Primaries_t.Vship_PRIMARIES_BT2020:{
            res.x = fma(a.x, 1.71650251f, fma(a.y, -0.35558469f, -a.z*0.25337521f));
            res.y = fma(a.x, -0.66662561f, fma(a.y, 1.61644657f, a.z*0.01577548f));
            res.z = fma(a.x, 0.01765521f, fma(a.y, -0.0428107f, a.z*0.94208926f));
            return res;
        }
        case Vship_Primaries_t.Vship_PRIMARIES_DisplayP3:{
            res.x = fma(a.x, 2.493586f, fma(a.y, -0.93141683f, -a.z*0.40272514f));
            res.y = fma(a.x, -0.82945554f, fma(a.y, 1.76259303f, a.z*0.02362373f));
            res.z = fma(a.x, 0.03585475f, fma(a.y, -0.07619133f, a.z*0.95712251f));
            return res;
        }
        case Vship_Primaries_t.Vship_PRIMARIES_INTERNAL:
        default:{
            return a;
        }
    }
}

float3 primariesToPrimaries_impl(const float3 a, Vship_Primaries_t primarySource, Vship_Primaries_t primaryDst){
    //specific case, the most common one
    if (primarySource == Vship_Primaries_t.Vship_PRIMARIES_BT2020 && primaryDst == Vship_Primaries_t.Vship_PRIMARIES_BT709){
        float3 res;
        res.x = fma(a.x, 1.6605f, fma(a.y, -0.5876f, -a.z*0.0728f));
        res.y = fma(a.x, -0.1246f, fma(a.y, 1.1329f, -a.z*0.0083f));
        res.z = fma(a.x, -0.0182f, fma(a.y, -0.1006f, a.z*1.1187f));
        return res;
    }
    //put everything in XYZ
    const float3 xyzrepr = primariesToXYZ(a, primarySource);
    return primariesFromXYZ(xyzrepr, primaryDst);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void primariesToPrimaries(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* p0, uniform float* p1, uniform float* p2, uniform uint64_t width, uniform Vship_Primaries_t primarySource, uniform Vship_Primaries_t primaryDst){
    const uint64_t x = threadIdx.x + 256*blockIdx.x;
    if (x >= width) return;

    float3 val = {p0[x], p1[x], p2[x]};

    val = primariesToPrimaries_impl(val, primarySource, primaryDst);

    p0[x] = val.x;
    p1[x] = val.y;
    p2[x] = val.z;
}
