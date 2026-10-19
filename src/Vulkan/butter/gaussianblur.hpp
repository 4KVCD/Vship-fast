namespace butter{

void loadGaussianKernel(float* gaussiankernel, float* gaussiankernel_integral, int gaussiansize, double sigma){
    double acc = 0;
    gaussiankernel_integral[0] = 0;
    for (int i = 0; i <= 2*gaussiansize; i++){
        double val = std::exp(-(gaussiansize-i)*(gaussiansize-i)/(2*sigma*sigma))/(std::sqrt(TAU*sigma*sigma));
        gaussiankernel[i] = val;
        acc += val;
        gaussiankernel_integral[i+1] = acc;
    }
}

//alows to precompute and reuse easily the same kernel array in gpu
class GaussianHandle{
    VshipDeviceHelper* deviceHelper;
    VkDeviceAddress gaussiankernel_d;
    float* staging_buffer;
    int indices[6];
public:
    //cmdBuff is preprocessing buffer
    void init(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff){
        this->deviceHelper = deviceHelper;
        gaussiankernel_d = deviceHelper->device.mallocDevice(sizeof(float)*1024);
        staging_buffer = (float*)deviceHelper->device.mallocHost(sizeof(float)*1024);

        indices[0] = 0;
        const double sigmas[5] = {1.2, 1.56416327805, 2.7, 3.22489901262, 7.15593339443};
        const int windowsizes[5] = {3, 4, 8, 8, 18};

        for (int i = 0; i < 5; i++){
            const double sigma = sigmas[i];
            const int windowsize = windowsizes[i];
            indices[i+1] = indices[i]+4*windowsize+3;
            loadGaussianKernel(staging_buffer+indices[i], staging_buffer+indices[i]+2*windowsize+1, windowsize, sigma);
        }
        //now fill the preprocessing buffer to upload staging buffer to gpu memory
        deviceHelper->device.memCpyHtoD(cmdBuff, gaussiankernel_d, staging_buffer, (1024*sizeof(float)));
    }
    VkDeviceAddress get(int i){
        return gaussiankernel_d+sizeof(float)*indices[i];
    }
    VkDeviceAddress getIntegral(int i){
        return gaussiankernel_d+sizeof(float)*(indices[i]+2*getWindow(i)+1);
    }
    int getWindow(int i){
        return ((indices[i+1]-indices[i])-3)/4;
    }
    void destroy(){
        deviceHelper->device.freeDevice(gaussiankernel_d);
        deviceHelper->device.freeHost(staging_buffer);
    }
};

/*void placeBlur(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress mem_d, VkDeviceAddress temp, uint64_t width, uint64_t height, GaussianHandle& gaussianHandle, int i){
    const int gaussiansize = gaussianHandle.getWindow(i);
    VkDeviceAddress gaussianKernel = gaussianHandle.get(i);
    VkDeviceAddress gaussianKernel_integral = gaussianHandle.getIntegral(i);

    uint64_t wh = width*height;
    uint64_t th_x = 256;
    uint64_t bl_x = (wh-1)/th_x + 1;

    uint64_t verticalth_x = 32;
    uint64_t verticalth_y = 8;
    uint64_t verticalbl_x = (width-1)/verticalth_x+1;
    uint64_t verticalbl_y = (height-1)/verticalth_y+1;

    VkPipeline horizontalPipeline = deviceHelper->cachedPipeline["gaussianBlurButter.spv:horizontalBlur"];
    VkPipeline verticalPipeline = deviceHelper->cachedPipeline["gaussianBlurButter.spv:verticalBlur"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, temp, mem_d, width, height, gaussianKernel, gaussianKernel_integral, gaussiansize);
    deviceHelper->device.placeKernel(cmdBuff, horizontalPipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {mem_d, width*height*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {temp, width*height*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
    });
    fillKernelArguments(0, kernelArguments, mem_d, temp, width, height, gaussianKernel, gaussianKernel_integral, gaussiansize);
    deviceHelper->device.placeKernel(cmdBuff, verticalPipeline, kernelArguments, verticalbl_x, verticalbl_y, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {temp, width*height*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {mem_d, width*height*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
    });
}*/

//The passes placeBlur can do after the blur, writing dst and second (postOp in separableBlur.slang; windows 3, 4 and
//18 only, and second must not be mem_d)
enum BlurPostOp {
    BLUR_POST_NONE = 0,
    BLUR_POST_RESIDUAL = 1,         //second = mem_d - blur
    BLUR_POST_REMOVE_RANGES = 2,    //subarray_removerangearound0 (1.5) then removerangearound0 (0.04) on second
    BLUR_POST_SEPARATE_HF_UHF = 3,  //separateHf_Uhf of the blur and mem_d
    //window 8 only (TiledGaussianBlur), with the weight w:
    BLUR_POST_SUBTRACT_REMOVE_RANGE = 4,  //second = mem_d - blur, then dst = RemoveRangeAroundZero(blur, w)
    BLUR_POST_SUBTRACT_AMPLIFY_RANGE = 5  //second = mem_d - blur, then dst = AmplifyRangeAroundZero(blur, w)
};

void placeBlur(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress dst, VkDeviceAddress mem_d, VkDeviceAddress temp, uint64_t width, uint64_t height, GaussianHandle& gaussianHandle, int i, VkDeviceAddress second = 0, BlurPostOp postOp = BLUR_POST_NONE, float w = 0.0f){
    const int gaussiansize = gaussianHandle.getWindow(i);
    VkDeviceAddress gaussianKernel = gaussianHandle.get(i);
    VkDeviceAddress gaussianKernel_integral = gaussianHandle.getIntegral(i);

    //each kernel does only its own passes after the blur
    const bool tiledPostOp = (postOp == BLUR_POST_SUBTRACT_REMOVE_RANGE || postOp == BLUR_POST_SUBTRACT_AMPLIFY_RANGE);
    if (postOp != BLUR_POST_NONE && tiledPostOp != (gaussiansize == 8)) throw VshipError(NotSupported, __FILE__, __LINE__, "this pass after the blur is not available for this window");

    VkPipeline pipeline;
    uint8_t kernelArguments[128];
    uint64_t bl_x, bl_y;
    if (gaussiansize == 8){
        //TiledGaussianBlur, 32x32 outputs per workgroup. The fused separableBlur kernel cannot replace it: its
        //compiled code mixes fused and separate multiply-adds, and each compiler mixes them its own way.
        pipeline = deviceHelper->cachedPipeline["gaussianBlurButter.spv:TiledGaussianBlur"];
        fillKernelArguments(0, kernelArguments, mem_d, dst, width, height, gaussianKernel, gaussianKernel_integral, second, (int)postOp, w);
        bl_x = ((width-1)/32+1)*((height-1)/32+1);
        bl_y = 1;
    } else {
        //the windows 3, 4 and 18: both passes in one kernel (separableBlur.slang), 32x64 outputs per workgroup.
        //0.0f is the 0 that each sum starts from.
        const std::string window = std::to_string(gaussiansize);
        pipeline = deviceHelper->cachedPipeline["separableBlur" + window + "Butter.spv:separableBlur" + window];
        fillKernelArguments(0, kernelArguments, mem_d, dst, width, height, gaussianKernel, gaussianKernel_integral, 0.0f, second, (int)postOp);
        bl_x = (width-1)/32+1;
        bl_y = (height-1)/64+1;
    }
    //temp is not needed: neither kernel goes through memory between its two passes
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, bl_y, 1);
    std::vector<MemoryBarrierPipeline> barriers = {
        {mem_d, width*height*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {dst, width*height*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
    };
    if (postOp != BLUR_POST_NONE) barriers.push_back({second, width*height*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT});
    deviceHelper->device.placeMemoryBarriers(cmdBuff, barriers);
}
/*void placeBlurDstNoTemp(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress dst, VkDeviceAddress mem_d, uint64_t width, uint64_t height, GaussianHandle& gaussianHandle, int i){
    const int gaussiansize = gaussianHandle.getWindow(i);
    VkDeviceAddress gaussianKernel = gaussianHandle.get(i);
    VkDeviceAddress gaussianKernel_integral = gaussianHandle.getIntegral(i);

    assert(gaussiansize == 8);
    //special gaussian blur! It doesnt even use temp
    int64_t th_x = 16;
    int64_t th_y = 16;
    int64_t bl_x = (width-1)/(2*th_x)+1;
    int64_t bl_y = (height-1)/(2*th_y)+1;
    
    VkPipeline tiledBlurPipeline = deviceHelper->cachedPipeline["gaussianBlurButter.spv:TiledGaussianBlur"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, mem_d, dst, width, height, gaussianKernel, gaussianKernel_integral);
    deviceHelper->device.placeKernel(cmdBuff, tiledBlurPipeline, kernelArguments, bl_x*bl_y, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {mem_d, width*height*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {dst, width*height*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
    });
}*/

//all gaussian sigmas:
/*
1.2
1.56416327805
2.7
3.22489901262
7.15593339443
*/

}
