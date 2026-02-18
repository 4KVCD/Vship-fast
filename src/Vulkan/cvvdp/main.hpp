#pragma once

#include "../util/preprocessor.hpp"
#include "../../util/VshipExceptions.hpp"
#include "../util/gpuhelper.hpp"
#include "../util/basicOperations.hpp"
#include "../../util/concurrency.hpp"
#include "../gpuColorToLinear/vshipColor.hpp"

#include "parameters.hpp"
#include "display_models.hpp"
#include "colors.hpp"
#include "resize.hpp"
#include "temporalFilter.hpp"
#include "pooling.hpp"
#include "lpyr.hpp"
#include "csf.hpp"
#include "gaussianBlur.hpp"
#include "maskingModel.hpp"
#include "distmap_specifics.hpp"

namespace cvvdp{

VkDeviceAddress placeCVVDPFinalStage(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, float* outputScoresCPU, bool outputDistmap, VkDeviceAddress mem_d, int64_t source_width, int64_t source_height, int64_t resize_width, int64_t resize_height, CSF_Handler& csfhandle, GaussianHandle& gaussianhandle, DisplayModel* model){
    const int64_t bandOffset = LpyrMemoryNeedPerPlane(resize_width, resize_height, model->get_screen_ppd());
    VkDeviceAddress Y_sustained1 = mem_d;
    VkDeviceAddress RG_sustained1 = mem_d + sizeof(float)*bandOffset;
    VkDeviceAddress YV_sustained1 = mem_d + sizeof(float)*2*bandOffset;
    VkDeviceAddress Y_transient1 = mem_d + sizeof(float)*3*bandOffset;

    VkDeviceAddress Y_sustained2 = mem_d + sizeof(float)*5*bandOffset;
    VkDeviceAddress RG_sustained2 = mem_d + sizeof(float)*6*bandOffset;
    VkDeviceAddress YV_sustained2 = mem_d + sizeof(float)*7*bandOffset;
    VkDeviceAddress Y_transient2 = mem_d + sizeof(float)*8*bandOffset;
    
    VkDeviceAddress tempResize1 = mem_d + sizeof(float)*10*bandOffset; //only allocated if needed
    VkDeviceAddress tempResize2 = mem_d + sizeof(float)*10*bandOffset+resize_width*source_height*sizeof(float);

    if (source_width < resize_width){
        //this means that the planes contained in Y_sustained... are not the right size!
        //we need to resize now
        placeResizePlane(deviceHelper, cmdBuff, Y_sustained1, tempResize1, Y_sustained1, source_width, source_height, resize_width, resize_height);
        placeResizePlane(deviceHelper, cmdBuff, RG_sustained1, tempResize1, RG_sustained1, source_width, source_height, resize_width, resize_height);
        placeResizePlane(deviceHelper, cmdBuff, YV_sustained1, tempResize1, YV_sustained1, source_width, source_height, resize_width, resize_height);
        placeResizePlane(deviceHelper, cmdBuff, Y_transient1, tempResize1, Y_transient1, source_width, source_height, resize_width, resize_height);
        placeResizePlane(deviceHelper, cmdBuff, Y_sustained2, tempResize2, Y_sustained2, source_width, source_height, resize_width, resize_height);
        placeResizePlane(deviceHelper, cmdBuff, RG_sustained2, tempResize2, RG_sustained2, source_width, source_height, resize_width, resize_height);
        placeResizePlane(deviceHelper, cmdBuff, YV_sustained2, tempResize2, YV_sustained2, source_width, source_height, resize_width, resize_height);
        placeResizePlane(deviceHelper, cmdBuff, Y_transient2, tempResize2, Y_transient2, source_width, source_height, resize_width, resize_height);
    }

    const float ppd = model->get_screen_ppd();
    LpyrManager LPyr1(deviceHelper, cmdBuff, mem_d, resize_width, resize_height, ppd, bandOffset);
    LpyrManager LPyr2(deviceHelper, cmdBuff, mem_d + sizeof(float)*5*bandOffset, resize_width, resize_height, ppd, bandOffset);
    
    int64_t w = resize_width;
    int64_t h = resize_height;
    const int levels = LPyr1.getSize();
    for (int band = 0; band < levels; band++){
        if (band != levels-1){
            for (int channel = 0; channel < 4; channel++){
                placePreGaussianPreCompute(deviceHelper, cmdBuff, LPyr1.getLbkg(band), LPyr1.getContrast(channel, band), LPyr2.getContrast(channel, band), w, h, channel, band, csfhandle);
            }
            //it writes on its source
            placeComputeD(deviceHelper, cmdBuff, LPyr1.getContrast(0, band), LPyr1.getContrast(1, band), LPyr1.getContrast(2, band), LPyr1.getContrast(3, band), LPyr2.getContrast(0, band), LPyr2.getContrast(1, band),LPyr2.getContrast(2, band), LPyr2.getContrast(3, band), w, h, gaussianhandle);
        } else {
            //baseband
            for (int channel = 0; channel < 4; channel++){
                placeComputeD_baseband(deviceHelper, cmdBuff, LPyr1.getLbkg(band), LPyr1.getContrast(channel, band), LPyr2.getContrast(channel, band), w, h, channel, band, csfhandle);
            }
        }
        w = (w+1)/2;
        h = (h+1)/2;
    }

    std::vector<float> scores(levels*4); //per_band and per channel
    int tempOffset = 0;
    VkDeviceAddress temp = LPyr1.getLbkg(0); //we take all the space! so we use band0
    for (int band = 0; band < levels; band++){
        for (int channel = 0; channel < 4; channel++){
            const auto [w, h] = LPyr1.getResolution(band);
            const int64_t size = w*h;
            placeComputeMean<2>(deviceHelper, cmdBuff, LPyr1.getContrast(channel, band), temp+sizeof(float)*tempOffset, size, true);
            tempOffset++; //we stored the norm at temp[tempOffset]
        }
    }

    deviceHelper->device.memCpyDtoH(cmdBuff, outputScoresCPU, temp, sizeof(float)*levels*4);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {{temp, sizeof(float)*levels*4, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}});

    if (outputDistmap){
        //we want a distmap
        placeGetDistMap(deviceHelper, cmdBuff, LPyr1);
        //result is in LPyr1.getLbkg(0), we can now discard other planes
        //LPyr1.getContrast(0, 0) can serve as a big buffer memory for the strided version
    }

    return LPyr1.getLbkg(0);
}

float cpuPostTreatment(float* scores, uint levels){
    //we have our levels*4 scores
    double finalValue = 0;
    for (int band = 0; band < levels; band++){
        for (int channel = 0; channel < 4; channel++){
            const float val = scores[band*4+channel];
            finalValue += std::pow(val, 4);
        }
    }
    finalValue = std::pow(finalValue, 1./4.);

    return finalValue;
}

//currently at 32 planes + 3*fps/2 planes consumed
class CVVDPComputingImplementation{
    DisplayModel* model = NULL;
    float fps = 0;
    //std::vector<float> all_scores; //for past frames
    int numFrame = 0;
    double score_squareSum = 0; //to avoid recomputing it all the time
    TemporalRing temporalRing1; //source
    TemporalRing temporalRing2; //encoded
    CSF_Handler csf_handler;
    GaussianHandle gaussianhandle;
    VshipColorConvert::Converter converter1;
    VshipColorConvert::Converter converter2;
    Vship_Colorspace_t ref_colorspace;
    Vship_Colorspace_t dis_colorspace;
    int64_t source_width = 0;
    int64_t source_height = 0;

    VshipDeviceHelper* deviceHelper;
    VkCommandPoolManager* commandPool; //contains 4 buffers -> preprocess, runtime frame To Gpu, runtime metric process, runtime distmap
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
    int64_t resize_width = 0;
    int64_t resize_height = 0;
    void init(Vship_Colorspace_t source_colorspace, Vship_Colorspace_t source_colorspace2, int gpu_id, float fps, bool resizeToDisplay, std::string model_key, std::string model_config_json){
        deviceHelper = globalVulkanInstance.get(gpu_id);
        model = new DisplayModel(model_key, model_config_json);
        this->fps = fps;
        ref_colorspace = source_colorspace;
        dis_colorspace = source_colorspace2;
        commandPool = deviceHelper->device.createCmdPool(4, 0);
        auto& cmdBuffs = commandPool->primaryCommandBuffers;
        this->source_width = VshipColorConvert::getColorspaceWidth(source_colorspace);
        this->source_height = VshipColorConvert::getColorspaceHeight(source_colorspace);

        if (resizeToDisplay){
            //we resize to match width since display_width/width <= display_height/height
            if (source_width*model->resolution[1] >= source_height*model->resolution[0]){
                resize_width = model->resolution[0];
                resize_height = (source_height*model->resolution[0])/source_width;
            } else {
                //here we resize to match source_height
                resize_height = model->resolution[1];
                resize_width = (source_width*model->resolution[1])/source_height;
            }
        } else {
            resize_width = source_width;
            resize_height = source_height;
        }

        int64_t minwidth = std::min(resize_width, source_width);
        int64_t minheight = std::min(resize_height, source_height);

        //memory allocations
        size_t tempSizeConverter1 = converter1.getTempMemoryNeeded();
        size_t tempSizeConverter2 = converter2.getTempMemoryNeeded();
        const int64_t bandOffset = LpyrMemoryNeedPerPlane(resize_width, resize_height, model->get_screen_ppd());
        size_t preLoadBufferSize = 2*(source_width*source_height*3*sizeof(float)+source_height*resize_width*sizeof(float));
        size_t finalStageBufferSize = 2*sizeof(float)*bandOffset*5;
        if (source_width < resize_width){
            //resize at the end
            finalStageBufferSize += 2*(resize_width*minheight*sizeof(float));
        }
        
        mem_d = deviceHelper->device.mallocDevice(std::max(preLoadBufferSize, finalStageBufferSize));

        //initial color planes
        VkDeviceAddress src1_d[3] = {mem_d, mem_d+sizeof(float)*source_width*source_height, mem_d+sizeof(float)*2*source_width*source_height};
        VkDeviceAddress src2_d[3] = {mem_d+sizeof(float)*3*source_width*source_height, mem_d+sizeof(float)*4*source_width*source_height, mem_d+sizeof(float)*5*source_width*source_height};
        //VkDeviceAddress resizeTemp[2] = {mem_d+sizeof(float)*6*width*height, mem_d+sizeof(float)*6*width*height+source_height*resize_width*sizeof(float)};
        
        outputScoresCPU = (float*)deviceHelper->device.mallocHost(sizeof(float)*14*4);

        fence = deviceHelper->device.createFence();
        semaphores.resize(2);
        semaphores[0] = deviceHelper->device.createSemaphores();
        semaphores[1] = deviceHelper->device.createSemaphores();

        //here we define the final destination of our color converted frames
        converter1.init(source_colorspace, VshipColorConvert::linRGB, deviceHelper, src1_d);
        converter2.init(source_colorspace2, VshipColorConvert::linRGB, deviceHelper, src2_d);

        converterTempMemory = mem_d+sizeof(float)*6*source_width*source_height; //at that point in the treatment, this memory can be used by the converter

        //assert they have the same width/height
        if (converter2.getWidth() != source_width || converter2.getHeight() != source_height){
            throw VshipError(DifferingInputType, __FILE__, __LINE__);            
        }

        //preprocess
        beginCmdBuffer(cmdBuffs[0]);
        //temporalRing are big VRAM consumers, we ll make themm store the smallest version of the video
        temporalRing1.init(deviceHelper, cmdBuffs[0], fps, minwidth, minheight);
        temporalRing2.init(deviceHelper, cmdBuffs[0], fps, minwidth, minheight);
        csf_handler.init(deviceHelper, cmdBuffs[0], resize_width, resize_height, model->get_screen_ppd());
        gaussianhandle.init(deviceHelper, cmdBuffs[0]);
        endCmdBuffer(cmdBuffs[0]);
        //execute
        deviceHelper->device.executeBuffer(cmdBuffs[0], fence);

        //while it's running, (even though it should take less time on gpu than queuesubmit oof) we do other stuff
        //record buffer2 (the heart of ssimu2)
        beginCmdBuffer(cmdBuffs[2]);
        //preprocess phase goes up to having temporally filtered planes filled up
        distmapAddress = placeCVVDPFinalStage(deviceHelper, cmdBuffs[2], outputScoresCPU, true, mem_d, source_width, source_height, resize_width, resize_height, csf_handler, gaussianhandle, model);
        endCmdBuffer(cmdBuffs[2]);

        fence->synchronize();

        //std::cout << "base/resize : " << width << "x" << source_height << "/" << resize_width << "x" << resize_height << std::endl;
        score_squareSum = 0;
        numFrame = 0;
    }
    void destroy(){
        temporalRing1.destroy();
        temporalRing2.destroy();
        csf_handler.destroy();
        gaussianhandle.destroy();
        converter1.destroy();
        converter2.destroy();
        delete model;
        delete commandPool;
        delete fence;
        deviceHelper->device.destroySemaphores(semaphores[0]);
        deviceHelper->device.destroySemaphores(semaphores[1]);
        deviceHelper->device.freeDevice(mem_d);
        if (outputCPU != NULL) deviceHelper->device.freeHost((void*)outputCPU);
        deviceHelper->device.freeHost((void*)outputScoresCPU);
    }
    void placeLoadImageToRing(VkCommandBuffer cmdBuff, const uint8_t *srcp1[3], const uint8_t *srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
        VkDeviceAddress src1_d[3] = {mem_d, mem_d+sizeof(float)*source_width*source_height, mem_d+sizeof(float)*2*source_width*source_height};
        VkDeviceAddress src2_d[3] = {mem_d+sizeof(float)*3*source_width*source_height, mem_d+sizeof(float)*4*source_width*source_height, mem_d+sizeof(float)*5*source_width*source_height};
        VkDeviceAddress resizeTemp[2] = {mem_d+sizeof(float)*6*source_width*source_height, mem_d+sizeof(float)*6*source_width*source_height+source_height*resize_width*sizeof(float)};

        //free up a plane by increasing history to 1, we can now edit the frame 0 which is blank
        temporalRing1.rotate();
        temporalRing2.rotate();
        //take color planes from the ring
        VkDeviceAddress source_ptr = temporalRing1.getFramePointer(0);
        VkDeviceAddress encoded_ptr = temporalRing2.getFramePointer(0);

        //final destination
        int64_t minwidth = std::min(source_width, resize_width);
        int64_t minheight = std::min(source_height, resize_height);

        VkDeviceAddress filterSrc1_d[3] = {source_ptr, source_ptr+sizeof(float)*minwidth*minheight, source_ptr+sizeof(float)*2*minwidth*minheight};
        VkDeviceAddress filterSrc2_d[3] = {encoded_ptr, encoded_ptr+sizeof(float)*minwidth*minheight, encoded_ptr+sizeof(float)*2*minwidth*minheight};

        VkDeviceAddress base_plane1[3];
        VkDeviceAddress base_plane2[3];
        //we work on the temporalRing buffer if source fits (=> smaller than resize)
        if (source_width > resize_width){
            for (int i = 0; i < 3; i++){
                base_plane1[i] = src1_d[i];
                base_plane2[i] = src2_d[i];
            }
        } else {
            for (int i = 0; i < 3; i++){
                base_plane1[i] = filterSrc1_d[i];
                base_plane2[i] = filterSrc2_d[i];
            }
        }

        size_t tempSizeConverter1 = converter1.getTempMemoryNeeded();
        //we perform the whole conversion directly
        converter1.placeToGPUBuffer(cmdBuff, srcp1, lineSize, converterTempMemory);
        converter2.placeToGPUBuffer(cmdBuff, srcp2, lineSize2, converterTempMemory+tempSizeConverter1);
        converter1.placeConvert(cmdBuff, converterTempMemory);
        converter2.placeConvert(cmdBuff, converterTempMemory+tempSizeConverter1);

        //resize as early as possible if source bigger than resize
        if (source_width > resize_width){
            for (int i = 0; i < 3; i++){
                placeResizePlane(deviceHelper, cmdBuff, filterSrc1_d[i], resizeTemp[0], base_plane1[i], source_width, source_height, resize_width, resize_height);
                placeResizePlane(deviceHelper, cmdBuff, filterSrc2_d[i], resizeTemp[1], base_plane2[i], source_width, source_height, resize_width, resize_height);
            }
        } else {
            //copy src_d into filterSrc
            for (int i = 0; i < 3; i++){
                deviceHelper->device.memCpyDtoD(cmdBuff, filterSrc1_d[i], src1_d[i], sizeof(float)*source_width*source_height);
                deviceHelper->device.memCpyDtoD(cmdBuff, filterSrc2_d[i], src2_d[i], sizeof(float)*source_width*source_height);
                deviceHelper->device.placeMemoryBarriers(cmdBuff, {
                    {filterSrc1_d[i], sizeof(float)*source_width*source_height, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
                    {filterSrc2_d[i], sizeof(float)*source_width*source_height, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
                    {src1_d[i], sizeof(float)*source_width*source_height, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
                    {src2_d[i], sizeof(float)*source_width*source_height, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
                });
            }
        }

        //now we work on src_d which has size minimum of source and resize

        const float Y_peak = model->max_luminance;
        const float Y_black = model->getBlackLevel();
        const float Y_refl = model->getReflLevel();
        const float exposure = model->exposure;
        const DisplayColorspace isHDR = model->colorspace;

        //std::cout << "Y_peak, Y_black, Y_refl, exposure : " << Y_peak << " " <<  Y_black << " " << Y_refl << " " << exposure << std::endl;

        //we put the frame's planes on GPU
        //do we write directly in final after stride eliminaation?
        for (int i = 0; i < 3; i++){
            placeDisplayEncode(deviceHelper, cmdBuff, filterSrc1_d[i], minwidth*minheight, Y_peak, Y_black, Y_refl, exposure, ref_colorspace.transferFunction, isHDR);
            placeDisplayEncode(deviceHelper, cmdBuff, filterSrc2_d[i], minwidth*minheight, Y_peak, Y_black, Y_refl, exposure, dis_colorspace.transferFunction, isHDR);
        }

        //go to XYZ
        VshipColorConvert::placePrimariesToPrimaries(deviceHelper, cmdBuff, filterSrc1_d[0], filterSrc1_d[1], filterSrc1_d[2], minwidth*minheight, ref_colorspace.primaries, Vship_PRIMARIES_INTERNAL);
        VshipColorConvert::placePrimariesToPrimaries(deviceHelper, cmdBuff, filterSrc2_d[0], filterSrc2_d[1], filterSrc2_d[2], minwidth*minheight, dis_colorspace.primaries, Vship_PRIMARIES_INTERNAL);

        //colorspace conversion at this point
        placeXYZ_to_dkl(deviceHelper, cmdBuff, filterSrc1_d, minwidth*minheight);
        placeXYZ_to_dkl(deviceHelper, cmdBuff, filterSrc2_d, minwidth*minheight);
        //and we are done, the frame is loaded in the ring with the right colorspace, next step is temporal filtering

        const int64_t bandOffset = LpyrMemoryNeedPerPlane(resize_width, resize_height, model->get_screen_ppd());
        VkDeviceAddress Y_sustained1 = mem_d;
        VkDeviceAddress RG_sustained1 = mem_d + sizeof(float)*bandOffset;
        VkDeviceAddress YV_sustained1 = mem_d + sizeof(float)*2*bandOffset;
        VkDeviceAddress Y_transient1 = mem_d + sizeof(float)*3*bandOffset;

        VkDeviceAddress Y_sustained2 = mem_d + sizeof(float)*5*bandOffset;
        VkDeviceAddress RG_sustained2 = mem_d + sizeof(float)*6*bandOffset;
        VkDeviceAddress YV_sustained2 = mem_d + sizeof(float)*7*bandOffset;
        VkDeviceAddress Y_transient2 = mem_d + sizeof(float)*8*bandOffset;

        //let's get the temporal channel out of the temporal ring!

        placeComputeTemporalChannels(deviceHelper, cmdBuff, temporalRing1, Y_sustained1, RG_sustained1, YV_sustained1, Y_transient1);
        placeComputeTemporalChannels(deviceHelper, cmdBuff, temporalRing2, Y_sustained2, RG_sustained2, YV_sustained2, Y_transient2);
    }
    void loadImageToRing(const uint8_t *srcp1[3], const uint8_t *srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
        auto& cmdBuffs = commandPool->primaryCommandBuffers;

        //create bufferToGPU
        resetCmdBuffer(cmdBuffs[1]);
        beginCmdBuffer(cmdBuffs[1]);
        placeLoadImageToRing(cmdBuffs[1], srcp1, srcp2, lineSize, lineSize2);
        endCmdBuffer(cmdBuffs[1]);

        //now we can execute cmdBuffs1
        deviceHelper->device.executeBuffers(cmdBuffs.data()+1, semaphores.data(), 1, fence);
        //then we have nothing else to do that wait
        fence->synchronize();
    }
    double run(const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
        //we need to record buffer 1 and 3, where 1 is putting frame on GPU and 3 is retrieving distmap if wanted
        auto& cmdBuffs = commandPool->primaryCommandBuffers;
        float ppd = model->get_screen_ppd();
        const int64_t bandOffset = LpyrMemoryNeedPerPlane(resize_width, resize_height, ppd);
        uint levels = get_frequencies(resize_width, resize_height, ppd).size();

        //create bufferToGPU
        resetCmdBuffer(cmdBuffs[1]);
        beginCmdBuffer(cmdBuffs[1]);
        placeLoadImageToRing(cmdBuffs[1], srcp1, srcp2, lineSize, lineSize2);
        endCmdBuffer(cmdBuffs[1]);
        
        bool isHostAllocated;
        float* intermediate;
        if (dstp != NULL){
            isHostAllocated = deviceHelper->device.isHostAllocated(dstp);
            intermediate = isHostAllocated ? (float*)dstp : lazyPinnedRealloc((resize_width*resize_height)*sizeof(float));
            resetCmdBuffer(cmdBuffs[3]);
            beginCmdBuffer(cmdBuffs[3]);
            placeStrideAdder(deviceHelper, cmdBuffs[3], distmapAddress, mem_d + sizeof(float)*4*bandOffset, dststride, resize_width, resize_height);
            deviceHelper->device.memCpyDtoH(cmdBuffs[3], intermediate, mem_d + sizeof(float)*4*bandOffset, dststride * resize_height);
            endCmdBuffer(cmdBuffs[3]);
        }

        //now we can execute cmdBuffs1, 2 and eventually 3 but synchronized by semaphores
        deviceHelper->device.executeBuffers(cmdBuffs.data()+1, semaphores.data(), (dstp != NULL) ? 3 : 2, fence);
        //then we have nothing else to do that wait
        fence->synchronize();
        //output should be in outputCPU

        if (dstp != NULL && intermediate != (float*)dstp){
            memcpy((float*)dstp, intermediate, dststride*resize_height);
        }
        
        const double current_score = cpuPostTreatment(outputScoresCPU, levels);
        score_squareSum += std::pow(current_score, beta_t);
        double resQ;
        numFrame++;
        if (numFrame == 1){
            resQ = current_score * image_int;
        } else {
            resQ = std::pow(score_squareSum/(double)numFrame, 1./beta_t);
        }
        return toJOD(resQ);
    }
    void flushOnlyScore(){
        numFrame = 0;
        score_squareSum = 0.;
    }
    //empties the history.
    void flushTemporalRing(){
        numFrame = 0;
        score_squareSum = 0.;
        temporalRing1.reset();
        temporalRing2.reset();
    }
};

}