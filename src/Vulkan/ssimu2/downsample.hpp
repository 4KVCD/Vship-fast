#pragma once

namespace ssimu2{

//Floats of a plane holding the 6 scales of a width x height image one after the other.
int64_t getTotalScaleSize(int64_t width, int64_t height){
    int64_t result = 0;
    for (int scale = 0; scale < 6; scale++){
        result += width*height;
        width = (width-1)/2+1;
        height = (height-1)/2+1;
    }
    return result;
}

//The 6 scales of an image in positive XYB into planes (each holding the scales one after the other) in one pass from
//the converter's luma and chroma (yuvToPyramid in shaders/downsampleSSIMU2.slang): what the converter's last steps
//and the 5 downsampling passes did.
void placeYuvToPyramid(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, const VshipColorConvert::Converter::PixelConversion& conversion, VkDeviceAddress planes[3], int64_t width, int64_t height){
    const int tile = 32; //PYRAMID_TILE: a workgroup does 32x32 pixels of scale 0
    const uint32_t zero = 0;
    const int64_t planeSize = getTotalScaleSize(width, height);

    //32x8 threads per workgroup on NVIDIA, 32x16 elsewhere (yuvToPyramidBody)
    const bool nvidia = deviceHelper->vendorID == 0x10DE;
    VkPipeline pipeline = deviceHelper->cachedPipeline[nvidia ? "downsampleSSIMU2.spv:yuvToPyramid8" : "downsampleSSIMU2.spv:yuvToPyramid16"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, conversion.luma, conversion.chroma1, conversion.chroma2, planes[0], planes[1], planes[2], (uint64_t)width, (uint64_t)height, conversion.verticalMode, (int)conversion.matrix, (int)conversion.transfer, (int)conversion.libjxlArithmetic, (int)conversion.toPrimaries, (int)conversion.primariesSource, (int)conversion.primariesDestination, zero, conversion.rawLuma, conversion.rawLumaFormat, conversion.rawLumaScale);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, (width-1)/tile + 1, (height-1)/tile + 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {planes[0], planeSize*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {planes[1], planeSize*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {planes[2], planeSize*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {conversion.chroma1, width*conversion.chromaHeight*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {conversion.chroma2, width*conversion.chromaHeight*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {(conversion.rawLuma != 0) ? conversion.rawLuma : conversion.luma, conversion.lumaBytes, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

//Halves the linear RGB planes src into dst and converts src to positive XYB; dst too when lastScale.
void placeDownsample(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress src1, VkDeviceAddress src2, VkDeviceAddress src3, VkDeviceAddress dst1, VkDeviceAddress dst2, VkDeviceAddress dst3, uint64_t width, uint64_t height, bool lastScale){
    int64_t newh = (height-1)/2 + 1;
    int64_t neww = (width-1)/2 + 1;

    int64_t th_x = 16;
    int64_t th_y = 16;
    int64_t bl_x = (neww-1)/th_x + 1;
    int64_t bl_y = (newh-1)/th_y + 1;

    const uint32_t lastScaleInt = lastScale ? 1u : 0u;
    const uint32_t zero = 0;

    VkPipeline pipeline = deviceHelper->cachedPipeline["downsampleSSIMU2.spv:downsample"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, src1, src2, src3, dst1, dst2, dst3, width, height, lastScaleInt, zero);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, bl_y, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {src1, height*width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src2, height*width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src3, height*width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {dst1, neww*newh*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {dst2, neww*newh*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {dst3, neww*newh*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
    });
}

}
