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
    VkDeviceAddress uploadedPlanes[3]; //the input planes as copied by placeUpload, and their line sizes
    int64_t uploadedLineSize[3];
    size_t uploadedSize = 0;
    //The upload's layout, the same for every frame so that kernels recorded once can read it: each plane's rows
    //packed one after the other from uploadOffset[i] of the upload buffer, 16-byte aligned for the conversion
    //kernel's reads.
    int64_t uploadLineSize[3];
    size_t uploadOffset[3];
    size_t uploadBytes = 0;
    //Input with padded lines is copied to a staging area as it is and packed into the upload's layout by placeToFloat:
    //each padded plane's staging address and line size, 0 for the planes copied straight into the layout.
    VkDeviceAddress stagedPlanes[3] = {0, 0, 0};
    int64_t stagedLineSize[3];

    uint64_t planeWidth(uint i){
        return (i == 0) ? width : ((width-1) >> source_colorspace.subsampling.subw)+1;
    }
    uint64_t planeHeight(uint i){
        return (i == 0) ? height : ((height-1) >> source_colorspace.subsampling.subh)+1;
    }
    void initUploadLayout(){
        const size_t alignment = 16;
        const uint byteSize = bytesizeSample(source_colorspace.sample);
        uploadBytes = 0;
        for (uint i = 0; i < 3; i++){
            uploadLineSize[i] = byteSize*planeWidth(i);
            uploadOffset[i] = uploadBytes;
            uploadBytes += (uploadLineSize[i]*planeHeight(i) + alignment - 1)/alignment*alignment;
        }
    }
    //Where placeChromaForPixelConversion puts the upsampled chroma in its temp: after the upload when the luma is read
    //from it (readsRawLuma).
    size_t chromaTempOffset(){
        return readsRawLuma() ? uploadBytes : 0;
    }

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
    VkDeviceAddress lazyStaging = 0;
    size_t lazyStagingSize = 0;
    VkDeviceAddress getLazyStaging(size_t size){
        if (size > lazyStagingSize){
            if (lazyStaging != 0) deviceHelper->device.freeDevice(lazyStaging);
            lazyStaging = deviceHelper->device.mallocDevice(size);
            lazyStagingSize = size;
        }
        return lazyStaging;
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
        initUploadLayout();
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
        if (lazyStaging != 0){
            deviceHelper->device.freeDevice(lazyStaging);
            lazyStaging = 0;
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
    //Puts the 3 input planes on the GPU in the upload's layout (uploadOffset, uploadLineSize): the CPU copies them to
    //pinned memory unless they already are, and uploadCmd gets the copy commands, so it can be a command buffer of
    //the device's upload queue. placeToFloat converts them afterwards. The copies go to
    //tempDevice when they fit in getTempMemoryNeeded(), else to a lazily allocated buffer.
    void placeUpload(VkCommandBuffer uploadCmd, const uint8_t* inp[3], const int64_t lineSize[3], VkDeviceAddress tempDevice){
        //each plane as the caller lays it out (its last line without padding), copied to pinned memory in one piece
        //unless it is there already, then in one piece to the GPU: into the upload's layout when its lines are not
        //padded, else into the staging area after it (placeToFloat packs the rows)
        const size_t alignment = 16;
        size_t inputBytes[3];
        size_t pinnedBytes = 0;
        size_t stagingBytes = 0;
        bool isHostAllocated = true;
        for (uint i = 0; i < 3; i++){
            inputBytes[i] = lineSize[i]*(planeHeight(i) - 1) + uploadLineSize[i];
            pinnedBytes += (inputBytes[i] + alignment - 1)/alignment*alignment;
            if (lineSize[i] != uploadLineSize[i]) stagingBytes += (inputBytes[i] + alignment - 1)/alignment*alignment;
            if (!deviceHelper->device.isHostAllocated(inp[i])){
                isHostAllocated = false;
            }
        }
        const VkDeviceAddress uploadBuffer = (uploadBytes <= getTempMemoryNeeded()) ? tempDevice : getLazyDevice(uploadBytes);
        //the staging area: in temp after what the conversion keeps there, else in a buffer of its own
        const size_t stagingOffset = uploadBytes + (readsRawLuma() ? chromaTempBytes() : 0);
        VkDeviceAddress staging = 0;
        if (stagingBytes != 0){
            staging = (uploadBuffer == tempDevice && stagingOffset + stagingBytes <= getTempMemoryNeeded()) ? tempDevice + stagingOffset : getLazyStaging(stagingBytes);
        }

        const uint8_t* src[3] = {inp[0], inp[1], inp[2]};
        if (!isHostAllocated){
            uint8_t* lazyPinned = getLazyPinned(pinnedBytes);
            size_t offsetPinned = 0;
            for (uint i = 0; i < 3; i++){
                src[i] = lazyPinned+offsetPinned;
                memcpy(lazyPinned+offsetPinned, inp[i], inputBytes[i]);
                offsetPinned += (inputBytes[i] + alignment - 1)/alignment*alignment;
            }
        }
        //src contains our data, always host allocated
        size_t offsetStaging = 0;
        for (uint i = 0; i < 3; i++){
            uploadedPlanes[i] = uploadBuffer + uploadOffset[i];
            uploadedLineSize[i] = uploadLineSize[i];
            if (lineSize[i] == uploadLineSize[i]){
                stagedPlanes[i] = 0;
                deviceHelper->device.memCpyHtoD(uploadCmd, uploadedPlanes[i], src[i], inputBytes[i]);
            } else {
                stagedPlanes[i] = staging + offsetStaging;
                stagedLineSize[i] = lineSize[i];
                deviceHelper->device.memCpyHtoD(uploadCmd, stagedPlanes[i], src[i], inputBytes[i]);
                offsetStaging += (inputBytes[i] + alignment - 1)/alignment*alignment;
            }
        }
        uploadedSize = uploadBytes;
    }
    //Packs the planes placeUpload staged into the upload's layout, then converts the planes to float, for the
    //conversion kernels; not the luma when the pixel conversion reads it from the upload itself (readsRawLuma).
    //cmdbuff must run after the upload has completed, and on the device's queue: it is recorded for each frame.
    void placeToFloat(VkCommandBuffer cmdbuff){
        //the planes with padded lines into the upload's layout
        std::vector<MemoryBarrierPipeline> packed;
        for (uint i = 0; i < 3; i++){
            if (stagedPlanes[i] == 0) continue;
            placeRepackRows(deviceHelper, cmdbuff, uploadedPlanes[i], uploadLineSize[i], stagedPlanes[i], stagedLineSize[i], uploadLineSize[i], planeHeight(i));
            packed.push_back({uploadedPlanes[i], (size_t)(uploadLineSize[i]*planeHeight(i)), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT});
            packed.push_back({stagedPlanes[i], (size_t)(stagedLineSize[i]*(planeHeight(i) - 1) + uploadLineSize[i]), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT});
        }
        if (!packed.empty()) deviceHelper->device.placeMemoryBarriers(cmdbuff, packed);
        const uint firstPlane = readsRawLuma() ? 1 : 0;
        for (uint i = firstPlane; i < 3; i++){
            placeConvertToFloatPlane(deviceHelper, cmdbuff, intermediatePlanes[i], uploadedPlanes[i], uploadedLineSize[i], planeWidth(i), planeHeight(i), source_colorspace.sample, source_colorspace.range, source_colorspace.colorFamily, (bool)(i != 0), libjxlArithmetic);
        }
        std::vector<MemoryBarrierPipeline> barriers = {{uploadedPlanes[0], uploadedSize, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}};
        for (uint i = firstPlane; i < 3; i++){
            barriers.push_back({intermediatePlanes[i], sizeof(float)*planeWidth(i)*planeHeight(i), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT});
        }
        deviceHelper->device.placeMemoryBarriers(cmdbuff, barriers);
    }
    //placeUpload and placeToFloat in the same command buffer
    void placeToGPUBuffer(VkCommandBuffer cmdbuff, const uint8_t* inp[3], const int64_t lineSize[3], VkDeviceAddress tempDevice){
        placeUpload(cmdbuff, inp, lineSize, tempDevice);
        deviceHelper->device.placeMemoryBarriers(cmdbuff, {{uploadedPlanes[0], uploadedSize, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}});
        placeToFloat(cmdbuff);
    }
    bool workConvertInPlace(){
        return !isResized && !isCropped;
    }
    //The conversion of the luma plane (luma) and the chroma (chroma1, chroma2: full width, chromaHeight rows,
    //verticalMode from chromaVerticalMode) to linear RGB that YUVToLinRGBUpsampled does for each pixel
    struct PixelConversion{
        //the luma: a float plane, or (rawLuma not 0) the uploaded integer plane, which the kernel converts as
        //convertToFloatPlane would (rawLumaSample in shaders/anyDepthToFloat.slang); lumaBytes is the plane's size
        VkDeviceAddress luma;
        VkDeviceAddress rawLuma;
        int rawLumaFormat;
        float rawLumaScale;
        size_t lumaBytes;
        VkDeviceAddress chroma1;
        VkDeviceAddress chroma2;
        int64_t chromaHeight;
        int verticalMode;
        Vship_YUVMatrix_t matrix;
        Vship_TransferFunction_t transfer;
        bool libjxlArithmetic;
        //then, when toPrimaries, primariesToPrimaries from primariesSource to primariesDestination, as placeConvert
        //does for the converter's destination
        bool toPrimaries;
        Vship_Primaries_t primariesSource;
        Vship_Primaries_t primariesDestination;
    };
    //The primaries placeConvert converts the linear RGB to for the destination; false when it keeps the source's.
    bool destinationPrimaries(Vship_Primaries_t& primaries){
        switch(destination){
            case XYZ:
                primaries = Vship_PRIMARIES_INTERNAL;
                break;
            case linRGBBT709:
                primaries = Vship_PRIMARIES_BT709;
                break;
            case linRGB:
                return false;
        }
        return source_colorspace.primaries != primaries;
    }
    //Whether placeChromaForPixelConversion takes the input: it is not linear RGB already, and YUVToLinRGBUpsampled
    //covers its chroma layout.
    bool coversPixelConversion(){
        const int verticalMode = chromaVerticalMode(source_colorspace.chromaLocation, source_colorspace.subsampling.subh);
        const bool toLinRGB = !(source_colorspace.transferFunction == Vship_TRC_Linear && source_colorspace.YUVMatrix == Vship_MATRIX_RGB);
        return toLinRGB && verticalMode >= 0;
    }
    //Whether the pixel conversion reads the luma from the upload (PixelConversion::rawLuma), so that placeToFloat
    //skips it: integer samples, an upload in the conversion's temp, and room after it for the upsampled chroma.
    bool readsRawLuma(){
        if (!coversPixelConversion() || bitsSample(source_colorspace.sample) == 0) return false;
        return uploadBytes + chromaTempBytes() <= getTempMemoryNeeded();
    }
    //The bytes of the horizontally upsampled chroma placeChromaForPixelConversion writes into its temp
    size_t chromaTempBytes(){
        const Vship_ChromaSubsample_t subsampling = source_colorspace.subsampling;
        const bool subsampled = (subsampling.subw != 0 || subsampling.subh != 0);
        return subsampled ? 2*sizeof(float)*width*planeHeight(1) : 0;
    }
    //Prepares the chroma (its horizontal upsampling into temp) for a kernel that converts each pixel to linear RGB as
    //YUVToLinRGBUpsampled does, and fills conversion. Returns false, placing nothing, when the input is linear RGB
    //already (nothing to convert) or its chroma layout is not one YUVToLinRGBUpsampled covers.
    bool placeChromaForPixelConversion(VkCommandBuffer cmdbuff, VkDeviceAddress temp, PixelConversion& conversion){
        if (!coversPixelConversion()) return false;
        const Vship_ChromaSubsample_t subsampling = source_colorspace.subsampling;
        const int verticalMode = chromaVerticalMode(source_colorspace.chromaLocation, subsampling.subh);
        const VkDeviceAddress chromaTemp = temp + chromaTempOffset();
        if (placeChromaUpsampleHorizontal(deviceHelper, cmdbuff, chromaTemp, intermediatePlanes, width, height, source_colorspace.chromaLocation, subsampling) != 0) throw VshipError(NotSupported, __FILE__, __LINE__, "the given chroma subsampling is not supported");
        const bool subsampled = (subsampling.subw != 0 || subsampling.subh != 0);
        if (readsRawLuma()){
            //temp is the upload buffer placeUpload was given
            conversion.luma = 0;
            conversion.rawLuma = temp + uploadOffset[0];
            conversion.rawLumaFormat = (int)source_colorspace.sample | ((int)source_colorspace.range << 8) | (misalignmentHandlingSample(uploadLineSize[0], source_colorspace.sample) ? 1 << 16 : 0);
            conversion.rawLumaScale = fullRangeScaleSample(source_colorspace.sample, libjxlArithmetic);
            conversion.lumaBytes = uploadLineSize[0]*height;
        } else {
            conversion.luma = intermediatePlanes[0];
            conversion.rawLuma = 0;
            conversion.rawLumaFormat = 0;
            conversion.rawLumaScale = 0.f;
            conversion.lumaBytes = sizeof(float)*width*height;
        }
        conversion.chromaHeight = ((height-1) >> subsampling.subh)+1;
        conversion.chroma1 = subsampled ? chromaTemp : intermediatePlanes[1];
        conversion.chroma2 = subsampled ? chromaTemp+sizeof(float)*width*conversion.chromaHeight : intermediatePlanes[2];
        conversion.verticalMode = verticalMode;
        conversion.matrix = source_colorspace.YUVMatrix;
        conversion.transfer = source_colorspace.transferFunction;
        conversion.libjxlArithmetic = libjxlArithmetic;
        conversion.primariesSource = source_colorspace.primaries;
        conversion.toPrimaries = destinationPrimaries(conversion.primariesDestination);
        return true;
    }
    //Whether the converter's output is the conversion of each input pixel, without crop or resize, so that a kernel
    //can take over from placeChromaForPixelConversion
    bool outputIsPixelConversion(){
        return !isResized && !isCropped;
    }
    void placeConvert(VkCommandBuffer cmdbuff, VkDeviceAddress temp){
        uint64_t maxWidth = std::max(source_colorspace.target_width, source_colorspace.width);
        uint64_t maxHeight= std::max(source_colorspace.target_height, source_colorspace.height);
        PixelConversion conversion;
        if (placeChromaForPixelConversion(cmdbuff, temp, conversion)){
            //the chroma's vertical upsampling within the conversion to linear RGB
            placeYUVToLinRGBUpsampled(deviceHelper, cmdbuff, intermediatePlanes[0], intermediatePlanes[1], intermediatePlanes[2], conversion.chroma1, conversion.chroma2, width, height, conversion.chromaHeight, conversion.verticalMode, conversion.matrix, conversion.transfer, conversion.libjxlArithmetic, conversion.rawLuma, conversion.rawLumaFormat, conversion.rawLumaScale, conversion.lumaBytes);
        } else {
            const Vship_ChromaSubsample_t subsampling = source_colorspace.subsampling;
            if (placeChromaUpsample(deviceHelper, cmdbuff, temp, intermediatePlanes, width, height, source_colorspace.chromaLocation, subsampling) != 0) throw VshipError(NotSupported, __FILE__, __LINE__, "the given chroma subsampling is not supported");
            placeYUVToLinRGB(deviceHelper, cmdbuff, intermediatePlanes[0], intermediatePlanes[1], intermediatePlanes[2], width*height, source_colorspace.YUVMatrix, source_colorspace.transferFunction, libjxlArithmetic);
        }
        //then the primaries of the destination
        Vship_Primaries_t primaries;
        if (destinationPrimaries(primaries)){
            placePrimariesToPrimaries(deviceHelper, cmdbuff, intermediatePlanes[0], intermediatePlanes[1], intermediatePlanes[2], width*height, source_colorspace.primaries, primaries);
        }

        //now we crop and put in out
        for (int i = 0; i < 3; i++){
            //resize and crop (we are still in linear space)
            VkDeviceAddress base = intermediatePlanes[i];
            VkDeviceAddress interm = temp;
            VkDeviceAddress interm2 = temp + sizeof(float)*maxWidth*maxHeight;
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
