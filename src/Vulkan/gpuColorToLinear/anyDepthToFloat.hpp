#pragma once

namespace VshipColorConvert{

int bytesizeSample(Vship_Sample_t sampleType){
    switch (sampleType){
        case Vship_SampleUINT8:
            return 1;
        case Vship_SampleUINT9:
        case Vship_SampleUINT10:
        case Vship_SampleUINT12:
        case Vship_SampleUINT14:
        case Vship_SampleUINT16:
        case Vship_SampleHALF:
            return 2;
        case Vship_SampleFLOAT:
            return 4;
    }
    return 0;
}

//bits of an integer sample type, 0 for half and float
int bitsSample(Vship_Sample_t sampleType){
    switch (sampleType){
        case Vship_SampleUINT8: return 8;
        case Vship_SampleUINT9: return 9;
        case Vship_SampleUINT10: return 10;
        case Vship_SampleUINT12: return 12;
        case Vship_SampleUINT14: return 14;
        case Vship_SampleUINT16: return 16;
        default: return 0;
    }
}

//libjxlArithmetic: full range integer samples are multiplied by 1.0f/(2^bits - 1) computed on the CPU, as libjxl
//does (lib/jxl/enc_external_image.cc), instead of divided on the GPU, which Vulkan allows to be 2.5 ulp off.
//convertToFloatPlane's scale of full range samples (0: divided by their maximum instead)
float fullRangeScaleSample(Vship_Sample_t sampleType, bool libjxlArithmetic){
    const int bits = bitsSample(sampleType);
    return (libjxlArithmetic && bits != 0) ? 1.0f / ((1ull << bits) - 1) : 0.f;
}

//convertToFloatPlane's misalignementHandling for a plane of this line size
bool misalignmentHandlingSample(uint64_t stride, Vship_Sample_t sampleType){
    return stride%bytesizeSample(sampleType) == 0;
}

//repackRows (shaders/anyDepthToFloat.slang): rows rows of rowBytes bytes from src, srcPitch apart, to dst, dstPitch
//apart
void placeRepackRows(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress dst, uint64_t dstPitch, VkDeviceAddress src, uint64_t srcPitch, uint64_t rowBytes, uint64_t rows){
    VkPipeline pipeline = deviceHelper->cachedPipeline["anyDepthToFloat.spv:repackRows"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, src, dst, srcPitch, dstPitch, rowBytes);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, (rowBytes-1)/256+1, rows, 1);
}

void placeConvertToFloatPlane(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress dst, VkDeviceAddress src, uint64_t stride, uint64_t width, uint64_t height, Vship_Sample_t sampleType, Vship_Range_t range, Vship_ColorFamily_t colorFam, bool chromaPlane, bool libjxlArithmetic = false){
    if (!fp16Support && sampleType == Vship_SampleHALF) throw VshipError(NotSupported, __FILE__, __LINE__, "fp16 not supported in this build");

    bool ColorFamchromaPlane = chromaPlane && colorFam == Vship_ColorYUV;
    bool misalignmentHandling = misalignmentHandlingSample(stride, sampleType);

    uint64_t wh = width*height;
    uint64_t th_x = 256;
    uint64_t bl_x = (wh-1)/th_x + 1;
    VkPipeline pipeline = deviceHelper->cachedPipeline["anyDepthToFloat.spv:convertToFloatPlane"];
    uint8_t kernelArguments[128];
    const float fullRangeScale = fullRangeScaleSample(sampleType, libjxlArithmetic);
    fillKernelArguments(0, kernelArguments, dst, src, stride, width, height, (int)sampleType, (int)range, (int)ColorFamchromaPlane, (int)misalignmentHandling, fullRangeScale);

    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {{dst, width*height*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}});
}

}