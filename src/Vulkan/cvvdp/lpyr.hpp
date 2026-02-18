#pragma once

namespace cvvdp{

#include "../util/vulkanDeviceManager.hpp"

void placeGaussPyrReduce(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress dst, VkDeviceAddress src, int64_t source_width, int64_t source_height){
    int nw = (source_width+1)/2;
    int nh = (source_height+1)/2;
    
    int th_x = 256;
    int64_t bl_x = (nw*nh+th_x-1)/th_x;

    VkPipeline pipeline = deviceHelper->cachedPipeline["lpyrCVVDP.spv:gaussPyrReduce"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, dst, src, source_width, source_height);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {dst, nh*nw*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src, source_width*source_height*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

template<bool subdst, bool adddst = false, bool clampMin = false>
void placeGaussPyrExpand(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress dst, VkDeviceAddress src, int64_t new_width, int64_t new_height){
    int th_x = 256;
    int64_t bl_x = (new_width*new_height+th_x-1)/th_x;
    int64_t old_width = (new_width-1)/2+1;
    int64_t old_height = (new_height-1)/2+1;

    constexpr uint conditions = (clampMin ? 4 : 0)|(adddst ? 2 : 0)|(subdst ? 1 : 0);
    VkPipeline pipeline = deviceHelper->cachedPipeline["lpyrCVVDP.spv:gaussPyrExpand"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, dst, src, new_width, new_height, conditions);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {dst, new_width*new_height*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src, old_width*old_height*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

//gets the contrast from the layers
template<bool isMean, int multiplier>
void placeBaseBandPyrRefine(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress p, VkDeviceAddress Lbkg, int64_t width){
    int th_x = 256;
    int64_t bl_x = (width + th_x-1)/th_x;

    constexpr uint conditions = (isMean ? 1 : 0);
    VkPipeline pipeline = deviceHelper->cachedPipeline["lpyrCVVDP.spv:baseBandPyrRefine"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, p, Lbkg, width, conditions, multiplier);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {p, width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {Lbkg, (isMean ? 1 : width)*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

std::vector<float> get_frequencies(const int64_t width, const int64_t height, const float ppd){
    const float min_freq = 0.2;
    const int maxLevel_forRes = std::log2(std::min(width, height))-1;
    const int maxLevel_forPPD = std::ceil(-std::log2(2*min_freq/0.3228/ppd))+3;
    const int maxLevel_hard = 14;
    const int levels = std::min(maxLevel_forPPD, std::min(maxLevel_forRes, maxLevel_hard));

    std::vector<float> band_frequencies(levels);
    band_frequencies[0] = 0.5*ppd;
    for (int i = 0; i < levels-1; i++){
        band_frequencies[i+1] = 0.3228*0.5*ppd/((float)(1 << i));
    }
    return band_frequencies;
}

//in float
int64_t LpyrMemoryNeedPerPlane(int64_t width, int64_t height, float ppd){
    std::vector<float> bands = get_frequencies(width, height, ppd);
    int64_t res = 0;
    int64_t w = width;
    int64_t h = height;
    for (uint level = 0; level < bands.size(); level++){
        res += h*w;

        w = (w+1)/2;
        h = (h+1)/2;
    }
    return res;
}

class LpyrManager{
    VshipDeviceHelper* deviceHelper;
    std::vector<std::pair<int64_t, int64_t>> resolutions; //for each band
    std::vector<VkDeviceAddress> adresses;
    int64_t planeOffset;
    std::vector<float> band_frequencies;
public: 
    //plane contains 5 planes each twice the size
    //the last plane will contain L_bkg while the first planes should contain the 4 temporal channels for the first half
    LpyrManager(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress plane, const int64_t width, const int64_t height, const float ppd, int64_t bandOffset){
        this->deviceHelper = deviceHelper;
        band_frequencies = get_frequencies(width, height, ppd);
        const int levels = band_frequencies.size();

        planeOffset = bandOffset;

        resolutions.resize(levels);
        adresses.resize(levels);
        int64_t w = width;
        int64_t h = height;
        VkDeviceAddress p = plane;
        for (int i = 0; i < levels; i++){
            resolutions[i].first = w;
            resolutions[i].second = h;
            adresses[i] = p;

            if (i != levels-1){
                //first is Y_sustained, it governs L_BKG
                placeGaussPyrReduce(deviceHelper, cmdBuff, p+sizeof(float)*w*h, p, w, h);
                placeGaussPyrExpand<false, false, true>(deviceHelper, cmdBuff, p+sizeof(float)*4*planeOffset, p+sizeof(float)*w*h, w, h);
                placeSubArray(deviceHelper, cmdBuff, p, p+sizeof(float)*4*planeOffset, p, w*h);
                if (i == 0){
                    placeBaseBandPyrRefine<false, 1>(deviceHelper, cmdBuff, p, p+sizeof(float)*4*planeOffset, w*h);
                } else {
                    placeBaseBandPyrRefine<false, 2>(deviceHelper, cmdBuff, p, p+sizeof(float)*4*planeOffset, w*h);
                }
                //then other channels
                for (int channel = 1; channel < 4; channel++){
                    //we first create the next step of the pyramid
                    placeGaussPyrReduce(deviceHelper, cmdBuff, p+sizeof(float)*(w*h+channel*planeOffset), p+sizeof(float)*channel*planeOffset, w, h);
                    //then we substract its upscaled version from the original to create the "layer"
                    placeGaussPyrExpand<true, false, false>(deviceHelper, cmdBuff, p+sizeof(float)*channel*planeOffset, p+sizeof(float)*(w*h+channel*planeOffset), w, h);
                    //then we transform this layer into a contrast by using the L_BKG computed before the loop
                    if (i == 0){
                        placeBaseBandPyrRefine<false, 1>(deviceHelper, cmdBuff, p+sizeof(float)*channel*planeOffset, p+sizeof(float)*4*planeOffset, w*h);
                    } else {
                        placeBaseBandPyrRefine<false, 2>(deviceHelper, cmdBuff, p+sizeof(float)*channel*planeOffset, p+sizeof(float)*4*planeOffset, w*h);
                    }
                }
            } else {
                //here Lbkg is different, it is the mean.
                //now we need to take the mean. 
                placeComputeMean<1>(deviceHelper, cmdBuff, p, p+sizeof(float)*4*planeOffset, w*h, true);
                VkDeviceAddress meanp = p+sizeof(float)*4*planeOffset;
                for (int channel = 0; channel < 4; channel++){
                    placeBaseBandPyrRefine<true, 1>(deviceHelper, cmdBuff, p+sizeof(float)*channel*planeOffset, meanp, w*h);
                }
            }

            p += sizeof(float)*w*h;
            w = (w+1)/2;
            h = (h+1)/2;
        }
    }
    int getSize(){
        return resolutions.size();
    }
    std::pair<int64_t, int64_t> getResolution(int i){
        return resolutions[i];
    }
    float getFrequency(int i){
        return band_frequencies[i];
    }
    VkDeviceAddress getLbkg(int band){
        return adresses[band]+sizeof(float)*4*planeOffset;
    }
    VkDeviceAddress getContrast(int channel, int band){
        return adresses[band]+sizeof(float)*channel*planeOffset;
    }
};

}