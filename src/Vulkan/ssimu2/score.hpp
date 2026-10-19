namespace ssimu2{

//libjxl's weights (tools/ssimulacra2.cc), in its order: for each plane, scale and norm (1, 4), the ssim, ringing
//(artifact) and blurring (detail lost) maps
const double weights[108] = {
    0.0,
    0.0007376606707406586,
    0.0,
    0.0,
    0.0007793481682867309,
    0.0,
    0.0,
    0.0004371155730107379,
    0.0,
    1.1041726426657346,
    0.00066284834129271,
    0.00015231632783718752,
    0.0,
    0.0016406437456599754,
    0.0,
    1.8422455520539298,
    11.441172603757666,
    0.0,
    0.0007989109436015163,
    0.000176816438078653,
    0.0,
    1.8787594979546387,
    10.94906990605142,
    0.0,
    0.0007289346991508072,
    0.9677937080626833,
    0.0,
    0.00014003424285435884,
    0.9981766977854967,
    0.00031949755934435053,
    0.0004550992113792063,
    0.0,
    0.0,
    0.0013648766163243398,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    7.466890328078848,
    0.0,
    17.445833984131262,
    0.0006235601634041466,
    0.0,
    0.0,
    6.683678146179332,
    0.00037724407979611296,
    1.027889937768264,
    225.20515300849274,
    0.0,
    0.0,
    19.213238186143016,
    0.0011401524586618361,
    0.001237755635509985,
    176.39317598450694,
    0.0,
    0.0,
    24.43300999870476,
    0.28520802612117757,
    0.0004485436923833408,
    0.0,
    0.0,
    0.0,
    34.77906344483772,
    44.835625328877896,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0008680556573291698,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0005313191874358747,
    0.0,
    0.00016533814161379112,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0004179171803251336,
    0.0017290828234722833,
    0.0,
    0.0020827005846636437,
    0.0,
    0.0,
    8.826982764996862,
    23.19243343998926,
    0.0,
    95.1080498811086,
    0.9863978034400682,
    0.9834382792465353,
    0.0012286405048278493,
    171.2667255897307,
    0.9807858872435379,
    0.0,
    0.0,
    0.0,
    0.0005130064588990679,
    0.0,
    0.00010854057858411537,
};

constexpr int scoreThreads = 64; //THREADS in shaders/scoreSSIMU2.slang: the columns of one workgroup
constexpr int scoreRowGroup = 16; //ROW_GROUP in shaders/scoreSSIMU2.slang: the rows of one workgroup of blurRows
constexpr int sumsPerWorkgroup = 6;

//Like libjxl, a scale is only scored when the scale above it (or the image, for the first) is at least 8x8.
int scoredScales(int64_t width, int64_t height){
    int scales = 0;
    for (int scale = 0; scale < 6; scale++){
        if (width < 8 || height < 8) break;
        scales++;
        width = (width-1)/2+1;
        height = (height-1)/2+1;
    }
    return scales;
}

//Floats of the sums the score kernels write, for all planes and scales.
int64_t sumsSize(int64_t width, int64_t height){
    int64_t result = 0;
    for (int scale = 0; scale < 6; scale++){
        result += 3*sumsPerWorkgroup*((width-1)/scoreThreads+1);
        width = (width-1)/2+1;
        height = (height-1)/2+1;
    }
    return result;
}

//Floats of the 5 horizontally blurred signals of the planes the score kernels work on at once: one plane at the full
//scale, or the 3 planes of all the smaller scales.
int64_t scoreRowsSize(int64_t width, int64_t height, int scales){
    int64_t smallerScales = 0;
    int64_t w = width;
    int64_t h = height;
    for (int scale = 1; scale < scales; scale++){
        w = (w-1)/2+1;
        h = (h-1)/2+1;
        smallerScales += 3*5*w*h;
    }
    return std::max(5*width*height, smallerScales);
}

//Working memory of the score kernels.
int64_t tempAllocsizeScore(int64_t width, int64_t height){
    return scoreRowsSize(width, height, scoredScales(width, height));
}

//Blurs the 5 signals of planes firstPlane .. firstPlane+planes-1 at scales firstScale .. firstScale+scales-1 and writes
//their sums of the error maps to sums (Job in shaders/scoreSSIMU2.slang). im1 and im2 are the first planes, planeStride
//floats apart.
void placeScoreJobs(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress sums, VkDeviceAddress im1, VkDeviceAddress im2, int64_t planeStride, VkDeviceAddress rows, int64_t rowsSize, int64_t baseWidth, int64_t baseHeight, int firstScale, int scales, int firstPlane, int planes, GaussianHandle& gaussianhandle){
    const uint32_t zero = 0;
    uint8_t kernelArguments[128];
    int64_t width = baseWidth;
    int64_t height = baseHeight;
    for (int scale = 0; scale < firstScale; scale++){
        width = (width-1)/2+1;
        height = (height-1)/2+1;
    }

    fillKernelArguments(0, kernelArguments, im1, im2, rows, gaussianhandle.coefficients_d, planeStride, baseWidth, baseHeight, firstScale, firstPlane, planes, zero);
    deviceHelper->device.placeKernel(cmdBuff, deviceHelper->cachedPipeline["scoreSSIMU2.spv:blurRows"], kernelArguments, (height-1)/scoreRowGroup+1, planes, scales);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {rows, sizeof(float)*rowsSize, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
    });

    fillKernelArguments(0, kernelArguments, im1, im2, rows, gaussianhandle.coefficients_d, sums, planeStride, baseWidth, baseHeight, firstScale, firstPlane, planes, zero);
    deviceHelper->device.placeKernel(cmdBuff, deviceHelper->cachedPipeline["scoreSSIMU2.spv:blurColumnsAndScore"], kernelArguments, (width-1)/scoreThreads+1, planes, scales);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {rows, sizeof(float)*rowsSize, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

//Scores every plane of every scored scale; sums_d gets their sums (sumsSize floats), copied to sums on the CPU.
//The planes of im1 and im2 must be equally spaced.
void placeAllscore_map(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, float* sums, VkDeviceAddress sums_d, VkDeviceAddress im1[3], VkDeviceAddress im2[3], VkDeviceAddress temp, int64_t basewidth, int64_t baseheight, GaussianHandle& gaussianhandle){
    const int scales = scoredScales(basewidth, baseheight);
    const int64_t planeStride = (int64_t)(im1[1]-im1[0])/sizeof(float);
    const int64_t rowsSize = scoreRowsSize(basewidth, baseheight, scales);
    for (int plane = 0; plane < 3; plane++){
        placeScoreJobs(deviceHelper, cmdBuff, sums_d, im1[0], im2[0], planeStride, temp, rowsSize, basewidth, baseheight, 0, 1, plane, 1, gaussianhandle);
    }
    if (scales > 1){
        placeScoreJobs(deviceHelper, cmdBuff, sums_d, im1[0], im2[0], planeStride, temp, rowsSize, basewidth, baseheight, 1, scales-1, 0, 3, gaussianhandle);
    }
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {sums_d, sizeof(float)*sumsSize(basewidth, baseheight), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
    });

    deviceHelper->device.memCpyDtoH(cmdBuff, sums, sums_d, sizeof(float)*sumsSize(basewidth, baseheight));
}
//libjxl's Msssim::Score from the sums written by placeAllscore_map, in double like libjxl.
double final_score(const float* sums, int64_t width, int64_t height){
    //measures[plane*36 + scale*6 + norm*3 + map], map 0: ssim, 1: artifact, 2: detail lost
    double measures[108] = {};
    const int scales = scoredScales(width, height);
    const float* workgroupSums = sums;
    for (int scale = 0; scale < scales; scale++){
        const int64_t workgroups = (width-1)/scoreThreads+1;
        const double onePerPixels = 1.0 / (height * width);
        for (int plane = 0; plane < 3; plane++){
            double total[sumsPerWorkgroup] = {};
            for (int64_t i = 0; i < workgroups; i++){
                for (int j = 0; j < sumsPerWorkgroup; j++) total[j] += workgroupSums[sumsPerWorkgroup*i + j];
            }
            workgroupSums += sumsPerWorkgroup*workgroups;
            //the shader's sums: ssim, artifact, detail lost, then their 4th powers
            for (int map = 0; map < 3; map++){
                measures[plane*36 + scale*6 + map] = onePerPixels * total[map];
                measures[plane*36 + scale*6 + 3 + map] = std::sqrt(std::sqrt(onePerPixels * total[3 + map]));
            }
        }
        width = (width-1)/2+1;
        height = (height-1)/2+1;
    }

    double ssim = 0.0;
    for (int i = 0; i < 108; i++){
        ssim += weights[i] * std::abs(measures[i]);
    }
    ssim = ssim * 0.9562382616834844;
    ssim = 2.326765642916932 * ssim - 0.020884521182843837 * ssim * ssim +
           6.248496625763138e-05 * ssim * ssim * ssim;
    if (ssim > 0) {
        ssim = 100.0 - 10.0 * std::pow(ssim, 0.6276336467831387);
    } else {
        ssim = 100.0;
    }
    return ssim;
}

}
