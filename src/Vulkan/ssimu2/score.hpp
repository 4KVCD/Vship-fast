namespace ssimu2{

//if the weight is below or equal to weight_pruning, we skip and put to 0
const float weight_pruning = 0.01f;
const float weights[108] = {
    0.0f,
    0.0007376606707406586f,
    0.0f,
    0.0f,
    0.0007793481682867309f,
    0.0f,
    0.0f,
    0.0004371155730107379f,
    0.0f,
    1.1041726426657346f,
    0.00066284834129271f,
    0.00015231632783718752f,
    0.0f,
    0.0016406437456599754f,
    0.0f,
    1.8422455520539298f,
    11.441172603757666f,
    0.0f,
    0.0007989109436015163f,
    0.000176816438078653f,
    0.0f,
    1.8787594979546387f,
    10.94906990605142f,
    0.0f,
    0.0007289346991508072f,
    0.9677937080626833f,
    0.0f,
    0.00014003424285435884f,
    0.9981766977854967f,
    0.00031949755934435053f,
    0.0004550992113792063f,
    0.0f,
    0.0f,
    0.0013648766163243398f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    7.466890328078848f,
    0.0f,
    17.445833984131262f,
    0.0006235601634041466f,
    0.0f,
    0.0f,
    6.683678146179332f,
    0.00037724407979611296f,
    1.027889937768264f,
    225.20515300849274f,
    0.0f,
    0.0f,
    19.213238186143016f,
    0.0011401524586618361f,
    0.001237755635509985f,
    176.39317598450694f,
    0.0f,
    0.0f,
    24.43300999870476f,
    0.28520802612117757f,
    0.0004485436923833408f,
    0.0f,
    0.0f,
    0.0f,
    34.77906344483772f,
    44.835625328877896f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0008680556573291698f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0005313191874358747f,
    0.0f,
    0.00016533814161379112f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0004179171803251336f,
    0.0017290828234722833f,
    0.0f,
    0.0020827005846636437f,
    0.0f,
    0.0f,
    8.826982764996862f,
    23.19243343998926f,
    0.0f,
    95.1080498811086f,
    0.9863978034400682f,
    0.9834382792465353f,
    0.0012286405048278493f,
    171.2667255897307f,
    0.9807858872435379f,
    0.0f,
    0.0f,
    0.0f,
    0.0005130064588990679f,
    0.0f,
    0.00010854057858411537f,
};

struct SkipMap{
    bool ssim;
    bool artifact;
    bool detailloss;
};

SkipMap getSkipMap(int plane, int scale){
    SkipMap res;
    //we want both norms to be below pruning in order to avoid an error map computation
    res.ssim = weights[plane*6*2*3 + scale*2*3 + 0*3 + 0] <= weight_pruning;
    res.ssim &= weights[plane*6*2*3 + scale*2*3 + 1*3 + 0] <= weight_pruning;
    res.artifact = weights[plane*6*2*3 + scale*2*3 + 0*3 + 1] <= weight_pruning;
    res.artifact &= weights[plane*6*2*3 + scale*2*3 + 1*3 + 1] <= weight_pruning;
    res.detailloss = weights[plane*6*2*3 + scale*2*3 + 0*3 + 2] <= weight_pruning;
    res.detailloss &= weights[plane*6*2*3 + scale*2*3 + 1*3 + 2] <= weight_pruning;
    return res;
}

int64_t tempAllocsizeScore(int64_t width, int64_t height){
    const int th_x = gaussianThx;
    const int th_y = gaussianThy;
    const int bl_x = (width-1)/(th_x*gaussianTileX)+1;
    const int bl_y = (height-1)/(th_y*gaussianTileY)+1;
    const int bl2_x = (width-1)/(th_x*gaussianTileX)/2+1;
    const int bl2_y = (height-1)/(th_y*gaussianTileY)/2+1;
    return 108 + 6*bl_x*bl_y + 6*bl2_x*bl2_y + 6*((bl_x*bl_y-1)/1024+1) + 6*((bl2_x*bl2_y-1)/1024+1);
}

//reduces the 6 next to each other arrays aat once
void placeSumReduce(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress destination, VkDeviceAddress source, uint64_t sourceNB){
    uint64_t thx = 1024;
    uint64_t blx = (sourceNB-1)/thx+1;

    VkPipeline pipeline = deviceHelper->cachedPipeline["sumreduceSSIMU2.spv:sumreduce"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, destination, source, sourceNB, blx);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, blx, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {destination, sizeof(float)*blx*6, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {source, sizeof(float)*sourceNB*6, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

void placePlanescale_map_noaccumulate(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress dst, VkDeviceAddress im1, VkDeviceAddress im2, int64_t width, int64_t height, GaussianHandle& gaussianhandle, SkipMap skipMap){
    if (skipMap.ssim && skipMap.artifact && skipMap.detailloss) return;
    const int th_x = gaussianThx;
    const int th_y = gaussianThy;
    const uint64_t bl_x = (width-1)/(th_x*gaussianTileX)+1;
    const uint64_t bl_y = (height-1)/(th_y*gaussianTileY)+1;

    VkPipeline pipeline = deviceHelper->cachedPipeline["scoreSSIMU2.spv:planescale_map"];
    
    int skipMapInteger = ((int)skipMap.detailloss << 2) | ((int)skipMap.artifact << 1) | ((int)skipMap.ssim);

    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, dst, im1, im2, width, height, gaussianhandle.gaussiankernel_d, gaussianhandle.gaussiankernel_integral_d, bl_x, bl_y, skipMapInteger);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, bl_y, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {dst, sizeof(float)*bl_x*bl_y*6, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {im1, sizeof(float)*width*height, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {im2, sizeof(float)*width*height, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

void placePlanescale_map(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress outbuffer, VkDeviceAddress im1, VkDeviceAddress im2, VkDeviceAddress temp, int64_t width, int64_t height, GaussianHandle& gaussianhandle, SkipMap skipMap){
    //outbuffer is a device buffer of size 6 float which should get {map0norm1, map1norm1, map2norm1, map0norm4, map1norm4, map2norm4} at the end (still async to stream)
    //skipped values can remain unset
    if (skipMap.ssim && skipMap.artifact && skipMap.detailloss) return;

    int64_t firstbufferSize = ((width-1)/(gaussianThx*gaussianTileX)+1)*((height-1)/(gaussianThy*gaussianTileY)+1);
    VkDeviceAddress tmps[2] = {temp, temp+6*sizeof(float)*firstbufferSize}; //of size firstbufferSize*6 and size 6*((firstbufferSize-1) / 1024+1)
    placePlanescale_map_noaccumulate(deviceHelper, cmdBuff, (firstbufferSize == 1) ? outbuffer:tmps[0], im1, im2, width, height, gaussianhandle, skipMap);

    int oscillate = 0; //current valid buffer
    while (firstbufferSize != 1){
        placeSumReduce(deviceHelper, cmdBuff, (firstbufferSize <= 1024) ? outbuffer:tmps[oscillate^1], tmps[oscillate], firstbufferSize);

        oscillate ^= 1;
        firstbufferSize = (firstbufferSize-1)/1024+1;
    }
}

//res of size 108
void placeAllscore_map(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, float* res, VkDeviceAddress im1[3], VkDeviceAddress im2[3], VkDeviceAddress temp, int64_t basewidth, int64_t baseheight, GaussianHandle& gaussianhandle){
    //measure_vec[plane*6*2*3 + scale*2*3 + n*3 + i]
    
    VkDeviceAddress outbuffer_d = temp;

    VkDeviceAddress temp1 = temp+sizeof(float)*108;
    const int th_x = gaussianThx;
    const int th_y = gaussianThy;
    const int bl_x = (basewidth-1)/(th_x*gaussianTileX)+1;
    const int bl_y = (baseheight-1)/(th_y*gaussianTileY)+1;
    VkDeviceAddress temp2 = temp1+sizeof(float)*(6*bl_x*bl_y + 6*((bl_x*bl_y-1)/1024+1));

    int64_t w = basewidth;
    int64_t h = baseheight;
    int64_t index = 0;
    //stream1 will exclusively manage scale 0 while stream2 will manage smaller scales
    for (int scale = 0; scale < 6; scale++){
        VkDeviceAddress tempScale = (scale == 0) ? temp1 : temp2;
        for (int plane = 0; plane < 3; plane++){
            SkipMap skipMap = getSkipMap(plane, scale);
            placePlanescale_map(deviceHelper, cmdBuff, outbuffer_d+sizeof(float)*(plane*6*2*3 + scale*2*3), im1[plane]+sizeof(float)*index, im2[plane]+sizeof(float)*index, tempScale, w, h, gaussianhandle, skipMap);
        }
        index += w*h;
        w = (w+1)/2;
        h = (h+1)/2;
    }

    deviceHelper->device.memCpyDtoH(cmdBuff, res, outbuffer_d, sizeof(float)*108);
}

float final_score(float* scores){

    for (int scale = 0; scale < 6; scale++){
        for (int plane = 0; plane < 3; plane++){
            SkipMap skipMap = getSkipMap(plane, scale);
            if (!skipMap.ssim){
                //complete norm4
                scores[plane*6*2*3 + scale*2*3 + 1*3 + 0] = std::sqrt(std::sqrt(scores[plane*6*2*3 + scale*2*3 + 1*3 + 0]));
            } else {
                //force 0 since we don't memset prior to computation it can be random
                scores[plane*6*2*3 + scale*2*3 + 1*3 + 0] = 0;
                scores[plane*6*2*3 + scale*2*3 + 0*3 + 0] = 0;
            }
            if (!skipMap.artifact){
                //complete norm4
                scores[plane*6*2*3 + scale*2*3 + 1*3 + 1] = std::sqrt(std::sqrt(scores[plane*6*2*3 + scale*2*3 + 1*3 + 1]));    
            } else {
                //force 0 since we don't memset prior to computation it can be random
                scores[plane*6*2*3 + scale*2*3 + 1*3 + 1] = 0;
                scores[plane*6*2*3 + scale*2*3 + 0*3 + 1] = 0;
            }
            if (!skipMap.detailloss){
                //complete norm4
                scores[plane*6*2*3 + scale*2*3 + 1*3 + 2] = std::sqrt(std::sqrt(scores[plane*6*2*3 + scale*2*3 + 1*3 + 2]));
            } else {
                //force 0 since we don't memset prior to computation it can be random
                scores[plane*6*2*3 + scale*2*3 + 1*3 + 2] = 0;
                scores[plane*6*2*3 + scale*2*3 + 0*3 + 2] = 0;
            }
        }
    }

    //score has to be of size 108
    double ssim = 0.0f;
    for (int i = 0; i < 108; i++){
        ssim = fmaf(weights[i], scores[i], ssim);
    }
    ssim *= 0.9562382616834844;
    ssim = (6.248496625763138e-5 * ssim * ssim) * ssim +
        2.326765642916932 * ssim -
        0.020884521182843837 * ssim * ssim;
    
    if (ssim > 0.0) {
        ssim = std::pow(ssim, 0.6276336467831387) * -10.0 + 100.0;
    } else {
        ssim = 100.0f;
    }

    return ssim;
}

}
