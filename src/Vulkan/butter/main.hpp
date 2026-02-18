#pragma once

#include <string>

#include "../util/preprocessor.hpp"
#include "../../util/VshipExceptions.hpp"
#include "../util/gpuhelper.hpp"
#include "../util/basicOperations.hpp"
#include "../../util/concurrency.hpp"
#include "../util/Planed.hpp"
#include "../gpuColorToLinear/vshipColor.hpp"

#include "gaussianblur.hpp" 
#include "downupsample.hpp"
#include "colors.hpp" //OpsinDynamicsImage
#include "separatefrequencies.hpp"
#include "maltaDiff.hpp"
#include "simplerdiff.hpp" //L2 +asym diff + same noise diff
#include "maskPsycho.hpp"
#include "combineMasks.hpp"
#include "diffnorms.hpp" //takes diffmap and returns norm2, norm3 and norminf

namespace butter{

//returns the address to which the diffmap is planned to be
VkDeviceAddress placeGetdiffmap(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress src1_d[3], VkDeviceAddress src2_d[3], VkDeviceAddress mem_d, uint64_t width, uint64_t height, float intensity_multiplier, GaussianHandle& gaussianHandle){
    //Psycho Image planes

    VkDeviceAddress temp[3] = {mem_d, mem_d+sizeof(float)*1*width*height, mem_d+sizeof(float)*2*width*height};

    VkDeviceAddress lf1[3] = {mem_d+sizeof(float)*3*width*height, mem_d+sizeof(float)*4*width*height, mem_d+sizeof(float)*5*width*height};
    VkDeviceAddress mf1[3] = {mem_d+sizeof(float)*6*width*height, mem_d+sizeof(float)*7*width*height, mem_d+sizeof(float)*8*width*height};
    VkDeviceAddress hf1[3] = {mem_d+sizeof(float)*9*width*height, mem_d+sizeof(float)*10*width*height};
    VkDeviceAddress uhf1[3] = {mem_d+sizeof(float)*11*width*height, mem_d+sizeof(float)*12*width*height};

    VkDeviceAddress lf2[3] = {mem_d+sizeof(float)*13*width*height, mem_d+sizeof(float)*14*width*height, mem_d+sizeof(float)*15*width*height};
    VkDeviceAddress mf2[3] = {mem_d+sizeof(float)*16*width*height, mem_d+sizeof(float)*17*width*height, mem_d+sizeof(float)*18*width*height};
    VkDeviceAddress hf2[3] = {mem_d+sizeof(float)*19*width*height, mem_d+sizeof(float)*20*width*height};
    VkDeviceAddress uhf2[3] = {mem_d+sizeof(float)*21*width*height, mem_d+sizeof(float)*22*width*height};

    //to XYB
    placeOpsinDynamicsImage(deviceHelper, cmdBuff, src1_d, temp, width, height, gaussianHandle, intensity_multiplier);
    placeOpsinDynamicsImage(deviceHelper, cmdBuff, src2_d, temp, width, height, gaussianHandle, intensity_multiplier);

    placeSeparateFrequencies(deviceHelper, cmdBuff, src1_d, temp, lf1, mf1, hf1, uhf1, width, height, gaussianHandle);
    placeSeparateFrequencies(deviceHelper, cmdBuff, src2_d, temp, lf2, mf2, hf2, uhf2, width, height, gaussianHandle);

    //no more needs for src1_d and src2_d so we reuse them as masks for butter
    VkDeviceAddress block_diff_dc[3] = {src1_d[0], src1_d[1], src1_d[2]};
    VkDeviceAddress block_diff_ac[3] = {src2_d[0], src2_d[1], src2_d[2]};

    //set the accumulators to 0
    for (int c = 0; c < 3; c++){
        placeMemset(deviceHelper, cmdBuff, block_diff_ac[c], width*height, 0);
        placeMemset(deviceHelper, cmdBuff, block_diff_dc[c], width*height, 0);
    }

    const float hf_asymmetry_ = 0.8f;

    const float wUhfMalta = 1.10039032555f;
    const float norm1Uhf = 71.7800275169f;
    placeMaltaDiffMap(deviceHelper, cmdBuff, uhf1[1], uhf2[1], block_diff_ac[1], width, height, wUhfMalta * hf_asymmetry_, wUhfMalta / hf_asymmetry_, norm1Uhf);

    const float wUhfMaltaX = 173.5f;
    const float norm1UhfX = 5.0f;
    placeMaltaDiffMap(deviceHelper, cmdBuff, uhf1[0], uhf2[0], block_diff_ac[0], width, height, wUhfMaltaX * hf_asymmetry_, wUhfMaltaX / hf_asymmetry_, norm1UhfX);

    const float wHfMalta = 18.7237414387f;
    const float norm1Hf = 4498534.45232f;
    placeMaltaDiffMapLF(deviceHelper, cmdBuff, hf1[1], hf2[1], block_diff_ac[1], width, height, wHfMalta * std::sqrt(hf_asymmetry_), wHfMalta / std::sqrt(hf_asymmetry_), norm1Hf);

    const float wHfMaltaX = 6923.99476109f;
    const float norm1HfX = 8051.15833247f;
    placeMaltaDiffMapLF(deviceHelper, cmdBuff, hf1[0], hf2[0], block_diff_ac[0], width, height, wHfMaltaX * std::sqrt(hf_asymmetry_), wHfMaltaX / std::sqrt(hf_asymmetry_), norm1HfX);

    const float wMfMalta = 37.0819870399f;
    const float norm1Mf = 130262059.556f;
    placeMaltaDiffMapLF(deviceHelper, cmdBuff, mf1[1], mf2[1], block_diff_ac[1], width, height, wMfMalta, wMfMalta, norm1Mf);

    const float wMfMaltaX = 8246.75321353f;
    const float norm1MfX = 1009002.70582f;
    placeMaltaDiffMapLF(deviceHelper, cmdBuff, mf1[0], mf2[0], block_diff_ac[0], width, height, wMfMaltaX, wMfMaltaX, norm1MfX);

    const float wmul[9] = {
      400.0f,         1.50815703118f,  0.0f,
      2150.0f,        10.6195433239f,  16.2176043152f,
      29.2353797994f, 0.844626970982f, 0.703646627719f,
    };

    for (int c = 0; c < 3; c++){
        if (c < 2){
            placeL2AsymDiff(deviceHelper, cmdBuff, hf1[c], hf2[c], block_diff_ac[c], width*height, wmul[c] * hf_asymmetry_, wmul[c] / hf_asymmetry_);
        }
        placeL2diff(deviceHelper, cmdBuff, mf1[c], mf2[c], block_diff_ac[c], width*height, wmul[3 + c]);
        placeL2diff(deviceHelper, cmdBuff, lf1[c], lf2[c], block_diff_dc[c], width*height, wmul[6 + c]);
    }

    //from now on, lf and mf are not used so we will reuse the memory
    VkDeviceAddress mask = temp[1];
    VkDeviceAddress temp3[3] = {lf2[0], lf2[1], lf2[2]};
    VkDeviceAddress temp4[3] = {mf2[0], mf2[1], mf2[2]};

    placeMaskPsychoImage(deviceHelper, cmdBuff, hf1, uhf1, hf2, uhf2, temp3[0], temp4[0], mask, block_diff_ac, width, height, gaussianHandle);
    //at this point hf and uhf cannot be used anymore (they have been invalidated by the function)

    VkDeviceAddress diffmap_ = temp[0];
    placeComputeDiffmap(deviceHelper, cmdBuff, mask, block_diff_dc[0], block_diff_dc[1], block_diff_dc[2], block_diff_ac[0], block_diff_ac[1], block_diff_ac[2], diffmap_, width*height);

    //hipEventRecord(event_d, stream); //place an event in the stream at the end of all our operations
    //hipEventSynchronize(event_d); //when the event is complete, we know our gpu result is ready!
    //GPU_CHECK(hipGetLastError());

    //printf("End result: %f, %f and %f\n", norm2, norm3, norminf);
    
    return diffmap_;
}

//expects linear planar RGB as input, mem_d must contain 25 planes (srcs are rewritten so they are unusable after execution)
VkDeviceAddress placeGetmultiscalediffmap(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress src1_d[3], VkDeviceAddress src2_d[3], VkDeviceAddress mem_d, uint64_t width, uint64_t height, float intensity_multiplier, GaussianHandle& gaussianHandle){
    //computing downscaled before we overwrite src in getdiffmap (it s better for memory)
    uint64_t nwidth = (width-1)/2+1;
    uint64_t nheight = (height-1)/2+1;
    VkDeviceAddress nmem_d = mem_d; //allow usage up to mem_d+2*width*height;
    VkDeviceAddress nsrc1_d[3] = {nmem_d, nmem_d+sizeof(float)*nwidth*nheight, nmem_d+sizeof(float)*2*nwidth*nheight};
    VkDeviceAddress nsrc2_d[3] = {nmem_d+sizeof(float)*3*nwidth*nheight, nmem_d+sizeof(float)*4*nwidth*nheight, nmem_d+sizeof(float)*5*nwidth*nheight};

    //using 6 smaller planes is equivalent to 1.5 standard planes, so it fits within the 2 planes given here!)
    for (int i = 0; i < 3; i++){
        placeDownsample(deviceHelper, cmdBuff, src1_d[i], nsrc1_d[i], width, height);
        placeDownsample(deviceHelper, cmdBuff, src2_d[i], nsrc2_d[i], width, height);
    }

    VkDeviceAddress diffmap = placeGetdiffmap(deviceHelper, cmdBuff, src1_d, src2_d, mem_d+sizeof(float)*2*width*height, width, height, intensity_multiplier, gaussianHandle);
    //diffmap is stored at mem_d+8*width*height so we can build after that the second smaller scale
    //smaller scale now
    nmem_d = mem_d+sizeof(float)*6*width*height;
    VkDeviceAddress diffmapsmall = placeGetdiffmap(deviceHelper, cmdBuff, nsrc1_d, nsrc2_d, nmem_d, nwidth, nheight, intensity_multiplier, gaussianHandle);

    placeAddsupersample2X(deviceHelper, cmdBuff, diffmap, diffmapsmall, width, height, 0.5f);

    return diffmap;
}

class ButterComputingImplementation{
    GaussianHandle gaussianhandle;
    VshipColorConvert::Converter converter1;
    VshipColorConvert::Converter converter2;
    int Qnorm;
    float intensity_multiplier;
    uint64_t width;
    uint64_t height;

    VshipDeviceHelper* deviceHelper;
    VkCommandPoolManager* commandPool; //contains 3 buffers -> preprocess, runtime frame To Gpu, runtime ssimu2 process
    VkDeviceAddress mem_d;
    VkDeviceAddress converterTempMemory;
    FenceManager* fence;
    std::vector<VkSemaphore> semaphores;
    float* outputScoresCPU = NULL;
    float* outputCPU = NULL;
    uint64_t outputCPUSize = 0;
    VkDeviceAddress distmapAddress;

    //realloc outputCPU if needed, returns it anyway
    float* lazyPinnedRealloc(uint64_t size){
        if (size <= outputCPUSize) return outputCPU;
        if (outputCPUSize != 0){
            deviceHelper->device.freeHost((void*)outputCPU);
        }
        outputCPU = (float*)deviceHelper->device.mallocHost(size);
        outputCPUSize = size;
        return outputCPU;
    }
public:
    //Qnorm replace the old norm2 and allows getting really any norm wanted
    void init(Vship_Colorspace_t source_colorspace, Vship_Colorspace_t source_colorspace2, int gpu_id, int Qnorm, float intensity_multiplier){
        this->deviceHelper = globalVulkanInstance.get(gpu_id);
        this->Qnorm = Qnorm;
        this->intensity_multiplier = intensity_multiplier;
        commandPool = deviceHelper->device.createCmdPool(4, 0);
        auto& cmdBuffs = commandPool->primaryCommandBuffers;
        this->width = VshipColorConvert::getColorspaceWidth(source_colorspace);
        this->height = VshipColorConvert::getColorspaceHeight(source_colorspace);

        //memory allocations
        size_t tempSizeConverter1 = converter1.getTempMemoryNeeded();
        size_t tempSizeConverter2 = converter2.getTempMemoryNeeded();
        mem_d = deviceHelper->device.mallocDevice(sizeof(float)*width*height*(6) + std::max(sizeof(float)*width*height*(25), (uint64_t)(tempSizeConverter1+tempSizeConverter2)));

        //initial color planes
        VkDeviceAddress src1_d[3] = {mem_d, mem_d+sizeof(float)*width*height, mem_d+sizeof(float)*2*width*height};
        VkDeviceAddress src2_d[3] = {mem_d+sizeof(float)*3*width*height, mem_d+sizeof(float)*4*width*height, mem_d+sizeof(float)*5*width*height};
        
        outputScoresCPU = (float*)deviceHelper->device.mallocHost(3*sizeof(float));

        fence = deviceHelper->device.createFence();
        semaphores.resize(2);
        semaphores[0] = deviceHelper->device.createSemaphores();
        semaphores[1] = deviceHelper->device.createSemaphores();

        //here we define the final destination of our color converted frames
        converter1.init(source_colorspace, VshipColorConvert::linRGBBT709, deviceHelper, src1_d);
        converter2.init(source_colorspace2, VshipColorConvert::linRGBBT709, deviceHelper, src2_d);

        converterTempMemory = mem_d+sizeof(float)*6*width*height; //at that point in the treatment, this memory can be used by the converter

        //assert they have the same width/height
        if (converter2.getWidth() != width || converter2.getHeight() != height){
            throw VshipError(DifferingInputType, __FILE__, __LINE__);            
        }

        //preprocess
        beginCmdBuffer(cmdBuffs[0]);
        gaussianhandle.init(deviceHelper, cmdBuffs[0]);
        endCmdBuffer(cmdBuffs[0]);
        //execute
        deviceHelper->device.executeBuffer(cmdBuffs[0], fence);

        //while it's running, (even though it should take less time on gpu than queuesubmit oof) we do other stuff
        //record buffer2 (the heart of ssimu2)
        beginCmdBuffer(cmdBuffs[2]);
        //color convert
        converter1.placeConvert(cmdBuffs[2], converterTempMemory);
        converter2.placeConvert(cmdBuffs[2], converterTempMemory+tempSizeConverter1);
        //result should be in src1_d src2_d
        distmapAddress = placeGetmultiscalediffmap(deviceHelper, cmdBuffs[2], src1_d, src2_d, converterTempMemory, width, height, intensity_multiplier, gaussianhandle);
        placeDiffmapscore(deviceHelper, cmdBuffs[2], distmapAddress, mem_d+sizeof(float)*9*width*height, mem_d+sizeof(float)*10*width*height, deviceHelper->device.getHostMemoryDeviceAddress(outputScoresCPU), width*height, Qnorm);
        endCmdBuffer(cmdBuffs[2]);

        fence->synchronize();
    }
    void destroy(){
        gaussianhandle.destroy();
        converter1.destroy();
        converter2.destroy();
        delete commandPool;
        delete fence;
        deviceHelper->device.destroySemaphores(semaphores[0]);
        deviceHelper->device.destroySemaphores(semaphores[1]);
        deviceHelper->device.freeDevice(mem_d);
        if (outputCPU != NULL) deviceHelper->device.freeHost((void*)outputCPU);
        deviceHelper->device.freeHost((void*)outputScoresCPU);
    }
    //if dstp is NULL, distmap won't be retrieved
    std::tuple<double, double, double> run(const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
        //we need to record buffer 1 and 3, where 1 is putting frame on GPU and 3 is retrieving distmap if wanted
        auto& cmdBuffs = commandPool->primaryCommandBuffers;

        //create bufferToGPU
        resetCmdBuffer(cmdBuffs[1]);
        beginCmdBuffer(cmdBuffs[1]);
        size_t tempSizeConverter1 = converter1.getTempMemoryNeeded();
        converter1.placeToGPUBuffer(cmdBuffs[1], srcp1, lineSize, converterTempMemory);
        converter2.placeToGPUBuffer(cmdBuffs[1], srcp2, lineSize2, converterTempMemory+tempSizeConverter1);
        endCmdBuffer(cmdBuffs[1]);
        
        bool isHostAllocated;
        float* intermediate;
        if (dstp != NULL){
            isHostAllocated = deviceHelper->device.isHostAllocated(dstp);
            intermediate = isHostAllocated ? (float*)dstp : lazyPinnedRealloc((width*height)*sizeof(float));
            resetCmdBuffer(cmdBuffs[3]);
            beginCmdBuffer(cmdBuffs[3]);
            placeStrideAdder(deviceHelper, cmdBuffs[3], distmapAddress, mem_d+sizeof(float)*6*width*height, dststride, width, height);
            deviceHelper->device.memCpyDtoH(cmdBuffs[3], intermediate, mem_d+sizeof(float)*6*width*height, dststride * height);
            endCmdBuffer(cmdBuffs[3]);
        }

        //now we can execute cmdBuffs1, 2 and eventually 3 but synchronized by semaphores
        deviceHelper->device.executeBuffers(cmdBuffs.data()+1, semaphores.data(), (dstp != NULL) ? 3 : 2, fence);
        //then we have nothing else to do that wait
        fence->synchronize();
        //output should be in outputCPU

        if (dstp != NULL && intermediate != (float*)dstp){
            memcpy((float*)dstp, intermediate, dststride*height);
        }

        posttreatment(outputScoresCPU, Qnorm);
        return std::make_tuple((double)outputScoresCPU[0], (double)outputScoresCPU[1], (double)outputScoresCPU[2]);
    }
};

}
