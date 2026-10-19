#pragma once

// GPU time of every kernel, for finding where the time goes (build with -DVSHIP_PROFILE only; normal builds compile
// none of this). Each command buffer gets a query pool: it is reset when the buffer is recorded, a timestamp is
// written before the buffer's first dispatch and after each dispatch, and the timestamps are read back when the
// fence of the submission is waited on. A kernel's time is the time between the end of the previous command and
// its own end, so it includes the barrier waits before it. The totals per kernel name are printed to stderr when the
// process exits.

#ifdef VSHIP_PROFILE

#include <algorithm>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vshipProfile {

static const uint32_t QUERIES = 4096; // timestamps per command buffer

struct Recording {
    VkQueryPool pool = VK_NULL_HANDLE;
    std::vector<VkPipeline> kernels; // in dispatch order; timestamp i+1 ends kernel i
    bool started = false;
};

struct Total {
    double ns = 0;
    uint64_t calls = 0;
};

struct Profiler {
    std::mutex lock;
    VkDevice device = VK_NULL_HANDLE;
    double nsPerTick = 1;
    std::unordered_map<VkPipeline, std::string> names;
    std::unordered_map<VkCommandBuffer, Recording> recordings;
    std::unordered_map<VkFence, std::vector<VkCommandBuffer>> pending; // submitted, waiting for their fence
    std::map<std::string, Total> totals;

    void init(VkDevice dev, VkPhysicalDevice physicalDevice){
        std::lock_guard<std::mutex> guard(lock);
        device = dev;
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        nsPerTick = properties.limits.timestampPeriod;
    }
    void name(VkPipeline pipeline, const std::string& entry){
        std::lock_guard<std::mutex> guard(lock);
        names[pipeline] = entry;
    }
    // A buffer being recorded again: its old timestamps were read when its fence was waited on.
    void begin(VkCommandBuffer cmdbuff){
        std::lock_guard<std::mutex> guard(lock);
        auto it = recordings.find(cmdbuff);
        if (it == recordings.end()) return;
        it->second.kernels.clear();
        it->second.started = false;
    }
    void beforeKernel(VkCommandBuffer cmdbuff){
        std::lock_guard<std::mutex> guard(lock);
        Recording& rec = recordings[cmdbuff];
        if (rec.pool == VK_NULL_HANDLE){
            VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            info.queryType = VK_QUERY_TYPE_TIMESTAMP;
            info.queryCount = QUERIES;
            vkCreateQueryPool(device, &info, nullptr, &rec.pool);
        }
        if (!rec.started){
            vkCmdResetQueryPool(cmdbuff, rec.pool, 0, QUERIES);
            vkCmdWriteTimestamp2(cmdbuff, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, rec.pool, 0);
            rec.started = true;
        }
    }
    void afterKernel(VkCommandBuffer cmdbuff, VkPipeline pipeline){
        std::lock_guard<std::mutex> guard(lock);
        Recording& rec = recordings[cmdbuff];
        if (rec.kernels.size() + 1 >= QUERIES) return;
        rec.kernels.push_back(pipeline);
        vkCmdWriteTimestamp2(cmdbuff, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, rec.pool, (uint32_t)rec.kernels.size());
    }
    void submitted(const VkCommandBuffer* cmdbuffs, uint32_t count, VkFence fence){
        std::lock_guard<std::mutex> guard(lock);
        for (uint32_t i = 0; i < count; i++)
            if (recordings.count(cmdbuffs[i])) pending[fence].push_back(cmdbuffs[i]);
    }
    // The fence was waited on: every buffer submitted with it has finished.
    void completed(VkFence fence){
        std::lock_guard<std::mutex> guard(lock);
        auto it = pending.find(fence);
        if (it == pending.end()) return;
        std::vector<uint64_t> ticks;
        for (VkCommandBuffer cmdbuff : it->second){
            Recording& rec = recordings[cmdbuff];
            const uint32_t n = (uint32_t)rec.kernels.size() + 1;
            ticks.resize(n);
            if (vkGetQueryPoolResults(device, rec.pool, 0, n, n*sizeof(uint64_t), ticks.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS) continue;
            for (uint32_t k = 0; k + 1 < n; k++){
                Total& t = totals[names.count(rec.kernels[k]) ? names[rec.kernels[k]] : "?"];
                t.ns += (double)(ticks[k+1] - ticks[k])*nsPerTick;
                t.calls++;
            }
        }
        pending.erase(it);
    }
    ~Profiler(){
        std::vector<std::pair<std::string, Total>> sorted(totals.begin(), totals.end());
        std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b){ return a.second.ns > b.second.ns; });
        double all = 0;
        for (auto& [n, t] : sorted) all += t.ns;
        fprintf(stderr, "vship profile: %.3f ms of GPU time in kernels\n", all*1e-6);
        for (auto& [n, t] : sorted)
            fprintf(stderr, "  %6.2f%%  %10.3f ms  %8llu calls  %9.2f us/call  %s\n", 100.0*t.ns/all, t.ns*1e-6, (unsigned long long)t.calls, t.ns*1e-3/t.calls, n.c_str());
    }
};

inline Profiler profiler;

}

#define VSHIP_PROFILE_INIT(device, physicalDevice) vshipProfile::profiler.init(device, physicalDevice)
#define VSHIP_PROFILE_NAME(pipeline, entry) vshipProfile::profiler.name(pipeline, entry)
#define VSHIP_PROFILE_BEGIN(cmdbuff) vshipProfile::profiler.begin(cmdbuff)
#define VSHIP_PROFILE_BEFORE_KERNEL(cmdbuff) vshipProfile::profiler.beforeKernel(cmdbuff)
#define VSHIP_PROFILE_AFTER_KERNEL(cmdbuff, pipeline) vshipProfile::profiler.afterKernel(cmdbuff, pipeline)
#define VSHIP_PROFILE_SUBMITTED(cmdbuffs, count, fence) vshipProfile::profiler.submitted(cmdbuffs, count, fence)
#define VSHIP_PROFILE_COMPLETED(fence) vshipProfile::profiler.completed(fence)

#else

#define VSHIP_PROFILE_INIT(device, physicalDevice)
#define VSHIP_PROFILE_NAME(pipeline, entry)
#define VSHIP_PROFILE_BEGIN(cmdbuff)
#define VSHIP_PROFILE_BEFORE_KERNEL(cmdbuff)
#define VSHIP_PROFILE_AFTER_KERNEL(cmdbuff, pipeline)
#define VSHIP_PROFILE_SUBMITTED(cmdbuffs, count, fence)
#define VSHIP_PROFILE_COMPLETED(fence)

#endif
