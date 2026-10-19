#pragma once

#include "../util/basicOperations.hpp"

//define enums for colorspaces
#include "../../VshipColor.h"

#include "resize.hpp"
#include "anyDepthToFloat.hpp" //manages range and sample type
#include "chromaUpsample.hpp" //manages chroma location and chroma subsampling
#include "YUVToLinRGB.hpp" //manages transfer function and YUV Matrix
#include "primaries.hpp" //manages primaries
#include "../util/Planed.hpp"

namespace VshipColorConvert{

enum ConverterDestination_t{
    XYZ,
    linRGBBT709,
    linRGB, //no primaries conversion
};

int64_t getColorspaceWidth(Vship_Colorspace_t source_colorspace){
    if (source_colorspace.target_width == -1) source_colorspace.target_width = source_colorspace.width;
    return source_colorspace.target_width - source_colorspace.crop.left - source_colorspace.crop.right;
}
int64_t getColorspaceHeight(Vship_Colorspace_t source_colorspace){
    if (source_colorspace.target_height == -1) source_colorspace.target_height = source_colorspace.height;
    return source_colorspace.target_height - source_colorspace.crop.top - source_colorspace.crop.bottom;
}

size_t getTempMemoryNeeded(Vship_Colorspace_t source_colorspace){
    uint64_t maxWidth = std::max(source_colorspace.target_width, source_colorspace.width);
    uint64_t maxHeight= std::max(source_colorspace.target_height, source_colorspace.height);
    uint maxplaneBuffer = 2;
    return sizeof(float)*maxWidth*maxHeight*maxplaneBuffer;
}

//you will call placeToGPU to fill a new fresh buffer each frame to send to gpu because we can't prerecord that one
//first call init to initialize it, then call placeConvert to add kernels into the buffer
class Converter{
    VshipDeviceHelper* deviceHelper;
    Vship_Colorspace_t source_colorspace;
    ConverterDestination_t destination;
    bool libjxlArithmetic = false;
    uint64_t width;
    uint64_t height;

    uint8_t* lazyAllocatedPinnedMemory = NULL;
    size_t lazyPinnedSize = 0;
    //this is used if stride is too big, we need to reallocate memory on the fly
    VkDeviceAddress lazyAllocatedReceiverMemory = 0;
    size_t lazyReceiverSize = 0;
    VkDeviceAddress outPlanes[3]; //plane on which to put the data in the end will be given upfront
    VkDeviceAddress intermediatePlanes[3]; //can be either newly allocated or copy of outPlanes

    bool isCropped;
    bool isResized;

    uint8_t* getLazyPinned(size_t size){
        if (size > lazyPinnedSize){
            if (lazyAllocatedPinnedMemory != NULL) deviceHelper->device.freeHost(lazyAllocatedPinnedMemory);
            lazyAllocatedPinnedMemory = (uint8_t*)deviceHelper->device.mallocHost(size);
            lazyPinnedSize = size;
        }
        return lazyAllocatedPinnedMemory;
    }
    VkDeviceAddress getLazyDevice(size_t size){
        if (size > lazyReceiverSize){
            if (lazyAllocatedReceiverMemory != 0) deviceHelper->device.freeDevice(lazyAllocatedReceiverMemory);
            lazyAllocatedReceiverMemory = deviceHelper->device.mallocDevice(size);
            lazyReceiverSize = size;
        }
        return lazyAllocatedReceiverMemory;
    }
public:
    Converter() = default;
    //width and height are of the input (before crop)
    //libjxlArithmetic: convert integer samples to float and undo the sRGB curve with libjxl's arithmetic, for the
    //metrics whose reference is libjxl (SSIMULACRA2 and Butteraugli are sensitive to an ulp of difference there)
    void init(Vship_Colorspace_t colorspace, ConverterDestination_t destination, VshipDeviceHelper* deviceHelper, VkDeviceAddress outPlanes[3], bool libjxlArithmetic = false){
        this->libjxlArithmetic = libjxlArithmetic;
        for (uint i = 0; i < 3; i++){
            this->outPlanes[i] = outPlanes[i];
        }
        this->deviceHelper = deviceHelper;
        this->width = colorspace.width;
        this->height = colorspace.height;
        this->source_colorspace = colorspace;
        this->destination = destination;  

        isCropped = true;
        if (colorspace.crop.top == 0 && colorspace.crop.bottom == 0 && colorspace.crop.left == 0 && colorspace.crop.right == 0) isCropped = false;

        if (colorspace.target_width == -1) source_colorspace.target_width = colorspace.width;
        if (colorspace.target_height == -1) source_colorspace.target_height = colorspace.height;
        isResized = true;
        if (source_colorspace.target_width == colorspace.width && source_colorspace.target_height == colorspace.height) isResized = false;
        //int64_t maxWidth = std::max(source_colorspace.target_width, source_colorspace.width);
        //int64_t maxHeight= std::max(source_colorspace.target_height, source_colorspace.height);
        
        if (!isResized && !isCropped){
            for (uint i = 0; i < 3; i++){
                intermediatePlanes[i] = outPlanes[i];
            }
        } else {
            //we need to allocate it
            intermediatePlanes[0] = deviceHelper->device.mallocDevice(sizeof(float)*width*height*3);
            intermediatePlanes[1] = intermediatePlanes[0]+sizeof(float)*width*height;
            intermediatePlanes[2] = intermediatePlanes[1]+sizeof(float)*width*height;
        }

        if (colorspace.YUVMatrix == Vship_MATRIX_RGB) {
            colorspace.colorFamily = Vship_ColorRGB;
        } else {
            colorspace.colorFamily = Vship_ColorYUV;
        }
    }
    void destroy(){
        if (isResized || isCropped){
            deviceHelper->device.freeDevice(intermediatePlanes[0]);
        }
        if (lazyAllocatedPinnedMemory != NULL){
            deviceHelper->device.freeHost(lazyAllocatedPinnedMemory);
            lazyAllocatedPinnedMemory = NULL;
        }
        if (lazyAllocatedReceiverMemory != 0){
            deviceHelper->device.freeDevice(lazyAllocatedReceiverMemory);
            lazyAllocatedReceiverMemory = 0;
        }
    }
    int64_t getWidth(){
        return source_colorspace.target_width - source_colorspace.crop.left - source_colorspace.crop.right;
    }
    int64_t getHeight(){
        return source_colorspace.target_height - source_colorspace.crop.top - source_colorspace.crop.bottom;
    }
    //not taking into account out
    size_t getTempMemoryNeeded(){
        uint64_t maxWidth = std::max(source_colorspace.target_width, source_colorspace.width);
        uint64_t maxHeight= std::max(source_colorspace.target_height, source_colorspace.height);
        uint maxplaneBuffer = 2;
        return sizeof(float)*maxWidth*maxHeight*maxplaneBuffer;
    }
    //performs a memcpy from inp in the case it's not host but doesnt otherwise
    //tempDevice will be used up to getTempMemoryNeeded in case stride is small enough to make it fit. Otherwise, lazy alloc memory
    void placeToGPUBuffer(VkCommandBuffer cmdbuff, const uint8_t* inp[3], const int64_t lineSize[3], VkDeviceAddress tempDevice){
        const uint64_t plane_widths[3] = {width, ((width-1) >> source_colorspace.subsampling.subw)+1, ((width-1) >> source_colorspace.subsampling.subw)+1};
        const uint64_t plane_heights[3] = {height, ((height-1) >> source_colorspace.subsampling.subh)+1, ((height-1) >> source_colorspace.subsampling.subh)+1};
        const uint byteSize = bytesizeSample(source_colorspace.sample);
        size_t maxByteSizeToCopy = 0;
        size_t lazyPinnedRequiredSize = 0;
        bool isHostAllocated = true;
        for (uint i = 0; i < 3; i++){
            const size_t bytesToCopy = lineSize[i]*plane_heights[i] - (lineSize[i] - byteSize*plane_widths[i]);
            maxByteSizeToCopy = std::max(bytesToCopy, maxByteSizeToCopy);
            lazyPinnedRequiredSize += bytesToCopy;
            if (!deviceHelper->device.isHostAllocated(inp[i])){
                isHostAllocated = false;
            }
        }
        VkDeviceAddress src_d; //receiver
        if (maxByteSizeToCopy <= getTempMemoryNeeded()){
            src_d = tempDevice;
        } else {
            src_d = getLazyDevice(maxByteSizeToCopy);
        }
        const uint8_t* src[3] = {inp[0], inp[1], inp[2]};
        uint8_t* lazyPinned = isHostAllocated ? NULL : getLazyPinned(lazyPinnedRequiredSize);
        size_t offsetPinned = 0;
        if (!isHostAllocated){
            for (uint i = 0; i < 3; i++){
                const size_t bytesToCopy = lineSize[i]*plane_heights[i] - (lineSize[i] - byteSize*plane_widths[i]);
                src[i] = lazyPinned+offsetPinned;
                memcpy(lazyPinned+offsetPinned, inp[i], bytesToCopy);
                offsetPinned += bytesToCopy;
            }
        }
        //src contains our data, always host allocated
        for (uint i = 0; i < 3; i++){
            const size_t bytesToCopy = lineSize[i]*plane_heights[i] - (lineSize[i] - byteSize*plane_widths[i]);
            deviceHelper->device.memCpyHtoD(cmdbuff, src_d, src[i], bytesToCopy);
            deviceHelper->device.placeMemoryBarriers(cmdbuff, {{src_d, bytesToCopy, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}});
            placeConvertToFloatPlane(deviceHelper, cmdbuff, intermediatePlanes[i], src_d, lineSize[i], plane_widths[i], plane_heights[i], source_colorspace.sample, source_colorspace.range, source_colorspace.colorFamily, (bool)(i != 0), libjxlArithmetic);
            deviceHelper->device.placeMemoryBarriers(cmdbuff, {
                {src_d, bytesToCopy, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
                {intermediatePlanes[i], sizeof(float)*plane_widths[i]*plane_heights[i], VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
            });
        }
    }
    bool workConvertInPlace(){
        return !isResized && !isCropped;
    }
    void placeConvert(VkCommandBuffer cmdbuff, VkDeviceAddress temp){
        uint64_t maxWidth = std::max(source_colorspace.target_width, source_colorspace.width);
        uint64_t maxHeight= std::max(source_colorspace.target_height, source_colorspace.height);
        if (placeChromaUpsample(deviceHelper, cmdbuff, temp, intermediatePlanes, width, height, source_colorspace.chromaLocation, source_colorspace.subsampling) != 0) throw VshipError(NotSupported, __FILE__, __LINE__, "the given chroma subsampling is not supported");
        placeYUVToLinRGB(deviceHelper, cmdbuff, intermediatePlanes[0], intermediatePlanes[1], intermediatePlanes[2], width*height, source_colorspace.YUVMatrix, source_colorspace.transferFunction, libjxlArithmetic);
        //then we manage primaries, it depends on the destination
        switch(destination){
            case linRGB:
                break; //we are done already
            case XYZ:
                if (source_colorspace.primaries != Vship_PRIMARIES_INTERNAL){
                    placePrimariesToPrimaries(deviceHelper, cmdbuff, intermediatePlanes[0], intermediatePlanes[1], intermediatePlanes[2], width*height, source_colorspace.primaries, Vship_PRIMARIES_INTERNAL);
                }
                break;
            case linRGBBT709:
                if (source_colorspace.primaries != Vship_PRIMARIES_BT709){
                    placePrimariesToPrimaries(deviceHelper, cmdbuff, intermediatePlanes[0], intermediatePlanes[1], intermediatePlanes[2], width*height, source_colorspace.primaries, Vship_PRIMARIES_BT709);
                }
                break;
        }

        //now we crop and put in out
        for (int i = 0; i < 3; i++){
            //resize and crop (we are still in linear space)
            VkDeviceAddress base = intermediatePlanes[i];
            VkDeviceAddress interm = temp;
            VkDeviceAddress interm2 = temp + maxWidth*maxHeight;
            VkDeviceAddress fin = outPlanes[i];
            if (isCropped && isResized){
                //std::cout << source_colorspace.width << "x" << source_colorspace.height << " / " << source_colorspace.target_width << "x" << source_colorspace.target_height << std::endl;
                //std::cout << source_colorspace.crop.top << " " << source_colorspace.crop.bottom << " " << source_colorspace.crop.left << " " << source_colorspace.crop.right << std::endl;
                placeResizePlane(deviceHelper, cmdbuff, interm, interm2, base, source_colorspace.width, source_colorspace.height, source_colorspace.target_width, source_colorspace.target_height);
                placeStrideEliminator(deviceHelper, cmdbuff, fin, interm+sizeof(float)*(source_colorspace.target_width*source_colorspace.crop.top+source_colorspace.crop.left), source_colorspace.target_width*sizeof(float), getWidth(), getHeight());
            } else if (isCropped){
                //std::cout << "Pointer offset: " << source_colorspace.target_width*source_colorspace.crop.top+source_colorspace.crop.left << ", stride: " << source_colorspace.target_width*sizeof(float) << ", new width: " << getWidth() << ", new height: " << getHeight() << std::endl;
                placeStrideEliminator(deviceHelper, cmdbuff, fin, base+sizeof(float)*(source_colorspace.target_width*source_colorspace.crop.top+source_colorspace.crop.left), source_colorspace.target_width*sizeof(float), getWidth(), getHeight());
            } else if (isResized){
                placeResizePlane(deviceHelper, cmdbuff, fin, interm, base, source_colorspace.width, source_colorspace.height, source_colorspace.target_width, source_colorspace.target_height);
            }
        }
    }
};

}
