import vshipColorModule;

//fullRangeScale: when not 0, full range samples are multiplied by it instead of divided by their maximum
float fullRange(Vship_Sample_t sampleType, Vship_Range_t range, bool ColorFamchromaPlane, float a, float fullRangeScale){
    uint32_t bitdepth = bitprecisionSample(sampleType);
    if (range == Vship_Range_t.Vship_RangeFull){
        float normalization = (1 << bitdepth) - 1; //float and half get 1
        //put in range [0, 1]
        if (fullRangeScale != 0.f) a *= fullRangeScale;
        else a /= normalization;
        if (ColorFamchromaPlane){
            a -= 0.5;
        }
        return a;
    } else {
        //limited
        if (bitdepth >= 8){
            //float or half have bitdepth 1. Here we have all uint types
            //-> range [0, 256[
            a /= (1u << (bitdepth - 8u));
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
    // the byte offset added to the address as an integer: SPIRV-Cross (MoltenVK) mistranslates byte-pointer arithmetic
    uint8_t* baseAdress = (uint8_t*)((uint64_t)source_plane + line*stride + byteSample*column);
    
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
            uint32_t bitSample = bitprecisionSample(sampleType);
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

// The float of sample x of a luma plane of the given width with rows packed one after the other, as
// convertToFloatPlane gives it. format packs the sample type, the range and misalignementHandling
// (Converter::PixelConversion::rawLumaFormat); fullRangeScale is convertToFloatPlane's.
float rawLumaSample(uint8_t* plane, uint64_t x, uint64_t width, int format, float fullRangeScale){
    const Vship_Sample_t sampleType = (Vship_Sample_t)(format & 0xff);
    const Vship_Range_t range = (Vship_Range_t)((format >> 8) & 0xff);
    const bool misalignementHandling = ((format >> 16) & 1) != 0;
    const uint64_t stride = width*(uint64_t)bytesizeSample(sampleType);
    return fullRange(sampleType, range, false, pickValue(plane, x, stride, width, sampleType, misalignementHandling), fullRangeScale);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void convertToFloatPlane(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* output_plane, uniform uint8_t* source_plane, uniform uint64_t stride, uniform uint64_t width, uniform uint64_t height, uniform Vship_Sample_t sampleType, uniform Vship_Range_t range, uniform int ColorFamchromaPlane, uniform int misalignementHandling, uniform float fullRangeScale){
    uint64_t x = threadIdx.x + 256*blockIdx.x;
    if (x >= width*height) return;

    float val = pickValue(source_plane, x, stride, width, sampleType, (bool)misalignementHandling);
    val = fullRange(sampleType, range, (bool)ColorFamchromaPlane, val, fullRangeScale);
    output_plane[x] = val;
}

// Packs rows rows of rowBytes bytes, srcPitch bytes apart in src, into dst, dstPitch bytes apart: the upload of a
// plane with padded lines into the upload's layout. A thread per byte, blockIdx.y the row.
[shader("compute")]
[numthreads(256, 1, 1)]
void repackRows(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform uint8_t* src, uniform uint8_t* dst, uniform uint64_t srcPitch, uniform uint64_t dstPitch, uniform uint64_t rowBytes){
    const uint64_t x = threadIdx.x + 256*(uint64_t)blockIdx.x;
    const uint64_t row = blockIdx.y;
    if (x >= rowBytes) return;
    dst[row*dstPitch + x] = src[row*srcPitch + x];
}
