import vshipColorModule;

float fullRange(Vship_Sample_t sampleType, Vship_Range_t range, bool ColorFamchromaPlane, float a){
    int bitdepth = bitprecisionSample(sampleType);
    if (range == Vship_Range_t.Vship_RangeFull){
        float normalization = (1 << bitdepth) - 1; //float and half get 1
        //put in range [0, 1]
        a /= normalization;
        //if UV -> -0.5
        if (ColorFamchromaPlane) a -= 0.5f;
        return a;
    } else {
        //limited
        if (bitdepth >= 8){
            //float or half have bitdepth 1. Here we have all uint types
            //-> range [0, 256[
            a /= (1 << (bitdepth - 8));
        } else {
            //float or half have bitdepth 1.
            //-> range [0, 256]
            //these types should ideally not use limited range since the formula is incorrect.
            a *= 256;
        }
        //range [0, 256] BUT we are in limited
        if (ColorFamchromaPlane){
            //chroma YUV formula
            return (a-128.f)/224.f;
        } else {
            //luma formula
            return (a-16.f)/219.f;
        }
    }
}

float pickValue(uint8_t* source_plane, uint64_t x, uint64_t stride, uint64_t width, Vship_Sample_t sampleType, bool misalignementHandling){
    int byteSample = bytesizeSample(sampleType);
    
    const uint64_t line = x/width;
    const uint64_t column = x%width;
    uint8_t* baseAdress = source_plane+line*stride + byteSample*column;
    
    //no misalignement possible
    if (byteSample == 1){
        return *baseAdress;
    }

    if (misalignementHandling){
        //handle misalignement by byte copy
        uint8_t raw[4];
        for (int i = 0; i < byteSample; i++){
            raw[i] = baseAdress[i];
        }

        if (sampleType == Vship_Sample_t.Vship_SampleFLOAT){
            return reinterpret<float>(raw);
#ifdef FP16SUPPORT
        } else if (sampleType == Vship_Sample_t.Vship_SampleHALF){
            return *((__half*)raw);
#endif
        } else {
            //only remains byteSample != 1 and non float type => byteSample == 2 uints
            //bitmasking to avoid assuming that garbage data is 0
            int bitSample = bitprecisionSample(sampleType);
            return reinterpret<uint16_t>(raw) & ((1u << bitSample)-1u);
        }
    } else {
        if (sampleType == Vship_Sample_t.Vship_SampleFLOAT){
            return *((float*)baseAdress);
#ifdef FP16SUPPORT
        } else if constexpr(sampleType == Vship_Sample_t.Vship_SampleHALF){
            return *((__half*)baseAdress);
#endif
        } else {
            //only remains byteSample != 1 and non float type => byteSample == 2 uints
            //bitmasking to avoid assuming that garbage data is 0
            int bitSample = bitprecisionSample(sampleType);
            return *((uint16_t*)baseAdress) & ((1u << bitSample)-1u);
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void convertToFloatPlane(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* output_plane, uniform uint8_t* source_plane, uniform uint64_t stride, uniform uint64_t width, uniform uint64_t height, uniform Vship_Sample_t sampleType, uniform Vship_Range_t range, uniform int ColorFamchromaPlane, uniform int misalignementHandling){
    uint64_t x = threadIdx.x + 256*blockIdx.x;
    if (x >= width*height) return;

    float val = pickValue(source_plane, x, stride, width, sampleType, (bool)misalignementHandling);
    val = fullRange(sampleType, range, (bool)ColorFamchromaPlane, val);
    output_plane[x] = val;
}