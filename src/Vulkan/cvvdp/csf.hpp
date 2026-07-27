#pragma once

#include "csf_lut_weber.hpp"

namespace cvvdp{

//takes an ordered array of size 32 and a value and search the biggest index of element smaller
//(except if x is smaller, then it returns 0)
int indexSearch(const float xp[32], float x){
    for (int i = 1; i < 32; i++){
        if (xp[i] > x){
            return i-1;
        }
    }
    return 31;
}

#include "../util/vulkanDeviceManager.hpp"

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

class CSF_Handler{
    VshipDeviceHelper* deviceHelper;
    float* cpumem;
public:
    VkDeviceAddress mem_d; //one allocation, we free at destroy
    void init(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, const int64_t width, const int64_t height, const float ppd){
        this->deviceHelper = deviceHelper;
        std::vector<float> band_frequencies = get_frequencies(width, height, ppd);
        const int levels = band_frequencies.size();
        mem_d = deviceHelper->device.mallocDevice(sizeof(float)*(1+4*levels)*32);
        cpumem = (float*)deviceHelper->device.mallocHost(sizeof(float)*(1+4*levels)*32);
        if (mem_d == 0){
            throw VshipError(OutOfVRAM, __FILE__, __LINE__);
        }

        float LUT_logrho[32];
        for (int i = 0; i < 32; i++){
            LUT_logrho[i] = std::log10(CSF_LUT::rho[i]);
            cpumem[i] = std::log10(CSF_LUT::L_bkg[i]);
        }

        //we create the logS_r_LUT on CPU since they are relatively small and not worth sending;
        for (int band = 0; band < levels; band++){
            float logrho = std::log10(band_frequencies[band]);
            const int x0 = std::min(30, indexSearch(LUT_logrho, logrho));
            const float slope = (logrho - LUT_logrho[x0])/(LUT_logrho[x0+1] - LUT_logrho[x0]);
            for (int channel = 0; channel < 4; channel++){
                float* res = cpumem+32*(1+band*4+channel);
                for (int i = 0; i < 32; i++){
                    const float y0i = CSF_LUT::D2LUT[channel][i][x0];
                    const float y1i = CSF_LUT::D2LUT[channel][i][x0+1];
                    res[i] = y0i + (y1i - y0i)*slope;
                }
            }
        }

        deviceHelper->device.memCpyHtoD(cmdBuff, mem_d, cpumem, sizeof(float)*(1+4*levels)*32);
        deviceHelper->device.placeMemoryBarriers(cmdBuff, {{mem_d, sizeof(float)*(1+4*levels)*32, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}});
    }
    void destroy(){
        deviceHelper->device.freeHost(cpumem);
        deviceHelper->device.freeDevice(mem_d);
    }
};

}
