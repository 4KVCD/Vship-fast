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

void placeConvertToFloatPlane(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress dst, VkDeviceAddress src, uint64_t stride, uint64_t width, uint64_t height, Vship_Sample_t sampleType, Vship_Range_t range, Vship_ColorFamily_t colorFam, bool chromaPlane){
    if (!fp16Support && sampleType == Vship_SampleHALF) throw VshipError(NotSupported, __FILE__, __LINE__, "fp16 not supported in this build");

    bool ColorFamchromaPlane = chromaPlane && colorFam == Vship_ColorYUV;
    bool misalignmentHandling = stride%bytesizeSample(sampleType) == 0;

    uint64_t wh = width*height;
    uint64_t th_x = 256;
    uint64_t bl_x = (wh-1)/th_x + 1;
    VkPipeline pipeline = deviceHelper->cachedPipeline["anyDepthToFloat.spv:convertToFloatPlane"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, dst, src, stride, width, height, (int)sampleType, (int)range, (int)ColorFamchromaPlane, (int)misalignmentHandling);

    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {{dst, width*height*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}});
}

}