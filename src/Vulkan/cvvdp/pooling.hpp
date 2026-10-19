#pragma once

namespace cvvdp{

template<int power, bool applyPower, bool applyInversePower>
void placeReduceSum(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress dst, VkDeviceAddress src, int64_t size, bool divide, float scale, bool safePow, bool clampMin){
    constexpr uint th_x = 1024;
    int64_t bl_x = (size+th_x-1)/th_x;

    uint conditions = (clampMin ? 16 : 0)|(safePow ? 8 : 0)|(applyPower ? 4 : 0)|(applyInversePower ? 2 : 0)|(divide ? 1 : 0);
    VkPipeline pipeline = deviceHelper->cachedPipeline["poolingCVVDP.spv:reduceSum"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, dst, src, size, power, conditions, scale);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {dst, bl_x*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src, size*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

//the result will be at temp[0]. We suppose that temp is of the same size as src
//The inputs are multiplied by scale (and clamped at 0.01 if clampMin) before the power; safePow uses pycvvdp's
//safe_pow for both powers.
template<int power>
void placeComputeMean(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress src, VkDeviceAddress temp, int64_t size, bool divide, float scale = 1.f, bool safePow = false, bool clampMin = false){
    VkDeviceAddress final_dst = temp; //to contain temp[0]
    VkDeviceAddress tempbuffer[3] = {temp+sizeof(float)*1, temp+sizeof(float)*(1+(size+1023)/1024), src};
    int oscillator = 2; //corresponds to current source except the first time
    while (size > 1024){
        int destination = (oscillator == 2) ? 0 : (oscillator^1);
        if (oscillator == 2){
            placeReduceSum<power, true, false>(deviceHelper, cmdBuff, tempbuffer[destination], tempbuffer[oscillator], size, divide, scale, safePow, clampMin); 
        } else {
            placeReduceSum<power, false, false>(deviceHelper, cmdBuff, tempbuffer[destination], tempbuffer[oscillator], size, false, scale, safePow, clampMin); 
        }
        oscillator = destination;
        size = (size+1023)/1024;
    }
    if (oscillator == 2){
        placeReduceSum<power, true, true>(deviceHelper, cmdBuff, final_dst, tempbuffer[oscillator], size, divide, scale, safePow, clampMin); 
    } else {
        placeReduceSum<power, false, true>(deviceHelper, cmdBuff, final_dst, tempbuffer[oscillator], size, false, scale, safePow, clampMin);
    }
}

//A mean of placeComputeMean's to compute in a batch: the result goes to dst instead of temp[0]
struct MeanJob{
    VkDeviceAddress src;
    VkDeviceAddress dst;
    int64_t size;
    bool divide;
    float scale;
    bool safePow;
    bool clampMin;
};

//One reduceSum dispatch of a mean, as reduceSums in shaders/pooling.slang reads it
struct ReduceJob{
    VkDeviceAddress dst;
    VkDeviceAddress src;
    int64_t size;
    uint32_t conditions;
    float scale;
};
static_assert(sizeof(ReduceJob) == 32, "ReduceJob must match shaders/pooling.slang");

//placeComputeMean for several means at once, with the same sums in the same order: the dispatches of all the
//means are grouped by their rank, each group running as one reduceSums dispatch.
template<int power>
class BatchedMeans{
    VshipDeviceHelper* deviceHelper = NULL;
    VkDeviceAddress jobs_d = 0;
    ReduceJob* jobsCPU = NULL;
    std::vector<int64_t> passFirstJob; //the jobs of pass p are passFirstJob[p] .. passFirstJob[p+1]-1
    std::vector<int64_t> passBlocks; //blocks of 1024 inputs of the largest job of each pass
    VkDeviceAddress scratchBegin = 0;
    VkDeviceAddress scratchEnd = 0;
public:
    //scratch receives the means' partial sums, 2*ceil(size/1024) floats per mean, and must not overlap dst or src
    void init(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, const std::vector<MeanJob>& means, VkDeviceAddress scratch){
        this->deviceHelper = deviceHelper;
        scratchBegin = scratch;
        //the dispatches of each mean, as placeComputeMean places them
        std::vector<std::vector<ReduceJob>> passes;
        for (const MeanJob& mean : means){
            const uint32_t keep = (mean.clampMin ? 16 : 0)|(mean.safePow ? 8 : 0);
            const uint32_t divide = mean.divide ? 1 : 0;
            const int64_t partials = (mean.size+1023)/1024;
            VkDeviceAddress buffers[3] = {scratch, scratch+sizeof(float)*partials, mean.src};
            scratch += 2*sizeof(float)*partials;
            int oscillator = 2; //the source of the next dispatch: the input, then the partial sums back and forth
            int64_t size = mean.size;
            size_t pass = 0;
            while (true){
                const bool first = (oscillator == 2);
                const bool last = (size <= 1024);
                const int destination = first ? 0 : (oscillator^1);
                ReduceJob job;
                job.dst = last ? mean.dst : buffers[destination];
                job.src = buffers[oscillator];
                job.size = size;
                job.conditions = keep|(first ? 4|divide : 0)|(last ? 2 : 0);
                job.scale = mean.scale;
                if (passes.size() <= pass) passes.resize(pass+1);
                passes[pass].push_back(job);
                if (last) break;
                oscillator = destination;
                size = (size+1023)/1024;
                pass++;
            }
        }
        scratchEnd = scratch;

        int64_t jobCount = 0;
        for (const auto& pass : passes) jobCount += pass.size();
        jobs_d = deviceHelper->device.mallocDevice(sizeof(ReduceJob)*jobCount);
        jobsCPU = (ReduceJob*)deviceHelper->device.mallocHost(sizeof(ReduceJob)*jobCount);
        if (jobs_d == 0 || jobsCPU == NULL){
            throw VshipError(OutOfVRAM, __FILE__, __LINE__);
        }
        int64_t index = 0;
        for (const auto& pass : passes){
            passFirstJob.push_back(index);
            int64_t blocks = 0;
            for (const ReduceJob& job : pass){
                jobsCPU[index++] = job;
                blocks = std::max(blocks, (job.size+1023)/1024);
            }
            passBlocks.push_back(blocks);
        }
        passFirstJob.push_back(index);
        deviceHelper->device.memCpyHtoD(cmdBuff, jobs_d, jobsCPU, sizeof(ReduceJob)*jobCount);
        deviceHelper->device.placeMemoryBarriers(cmdBuff, {{jobs_d, sizeof(ReduceJob)*jobCount, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}});
    }
    //dst holds the extent of all the means' results, for the barrier after them
    void place(VkCommandBuffer cmdBuff, VkDeviceAddress dst, size_t dstSize){
        constexpr int blocksPerWorkgroup = 8; //TEAMS in shaders/pooling.slang
        VkPipeline pipeline = deviceHelper->cachedPipeline["poolingCVVDP.spv:reduceSums"];
        uint8_t kernelArguments[128];
        for (size_t pass = 0; pass+1 < passFirstJob.size(); pass++){
            const int64_t jobs = passFirstJob[pass+1]-passFirstJob[pass];
            fillKernelArguments(0, kernelArguments, jobs_d+sizeof(ReduceJob)*passFirstJob[pass], power);
            deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, (passBlocks[pass]+blocksPerWorkgroup-1)/blocksPerWorkgroup, jobs, 1);
            deviceHelper->device.placeMemoryBarriers(cmdBuff, {
                {scratchBegin, scratchEnd-scratchBegin, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
                {dst, dstSize, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
            });
        }
    }
    void destroy(){
        if (jobs_d != 0) deviceHelper->device.freeDevice(jobs_d);
        if (jobsCPU != NULL) deviceHelper->device.freeHost(jobsCPU);
    }
};

}