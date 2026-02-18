float3 XybLowFreqToVals_Device(float3 xyb) {
    const float xmuli = 33.832837186260f;
    const float ymuli = 14.458268100570f;
    const float bmuli = 49.87984651440f;
    const float y_to_b_muli = -0.362267051518f;
    xyb.z = xyb.z + y_to_b_muli * xyb.y;
    xyb.z = xyb.z * bmuli;
    xyb.x = xyb.x * xmuli;
    xyb.y = xyb.y * ymuli;
    return xyb;
}

float SuppressHfInBrightAreas_Device(float hf, float brightness, float mul, float reg) {
    float scaler = mul * reg / (reg + brightness);
    return hf * scaler;
}

float SuppressUhfInBrightAreas_Device(float hf, float brightness, float mul, float reg) {
    float scaler = mul * reg / (reg + brightness);
    return hf * scaler;
}

float MaximumClamp_Device(float v, float maxval) {
    const float kMul = 0.688059627878f;
    if (v >= maxval) {
        v -= maxval;
        v *= kMul;
        v += maxval;
    } else if (v < -maxval) {
        v += maxval;
        v *= kMul;
        v -= maxval;
    }
    return v;
}

float supressXbyY_Device(float x, float yval, float yw){
    const float s = 0.653020556257f;
    const float scaler = s + (yw * (1.0f - s)) / (yw + yval * yval);
    return x * scaler;
}

// Make area around zero more important (2x it until the limit).
float AmplifyRangeAroundZero_Device(float x, float w) {
    return x > w ? x + w : x < -w ? x - w : 2.0f * x;
}

// Make area around zero less important (remove it).
float RemoveRangeAroundZero_Device(float x, float w) {
    return x > w ? x - w : x < -w ? x + w : 0.0f;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void subarray_removerangearound0(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* first, uniform float* second, uniform uint64_t width, uniform float w){
    uint64_t x = threadIdx.x + blockIdx.x*256;

    if (x >= width) return;

    second[x] -= first[x];
    first[x] = RemoveRangeAroundZero_Device(first[x], w);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void removerangearound0(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* first, uniform uint64_t width, uniform float w){
    uint64_t x = threadIdx.x + blockIdx.x*256;

    if (x >= width) return;

    first[x] = RemoveRangeAroundZero_Device(first[x], w);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void subarray_amplifyrangearound0(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* first, uniform float* second, uniform uint64_t width, uniform float w){
    uint64_t x = threadIdx.x + blockIdx.x*256;

    if (x >= width) return;

    second[x] -= first[x];
    first[x] = AmplifyRangeAroundZero_Device(first[x], w);
    //printf("res : %f and %f\n", first[x], second[x]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void supressXbyY(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* first, uniform float* second, uniform uint64_t width, uniform float yw){
    uint64_t x = threadIdx.x + blockIdx.x*256;

    if (x >= width) return;

    first[x] = supressXbyY_Device(first[x], second[x], yw);
    //printf("res : %f and %f\n", first[x], second[x]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void separateHf_Uhf(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* hf, uniform float* uhf, uniform uint64_t width){
    uint64_t x = threadIdx.x + blockIdx.x*256;

    if (x >= width) return;

    const float kMaxclampHf = 28.4691806922f;
    const float kMaxclampUhf = 5.19175294647f;

    hf[x] = MaximumClamp_Device(hf[x], kMaxclampHf);
    uhf[x] -= hf[x];
    uhf[x] = MaximumClamp_Device(uhf[x], kMaxclampUhf);
    uhf[x] *= 2.69313763794f;
    hf[x] *= 2.155f;
    hf[x] = AmplifyRangeAroundZero_Device(hf[x], 0.132f);

    //printf("res : %f and %f\n", first[x], second[x]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void XybLowFreqToVals(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* xplane, uniform float* yplane, uniform float* bplane, uniform uint64_t width){
    uint64_t x = threadIdx.x + blockIdx.x*256;

    if (x >= width) return;

    
    float3 src = XybLowFreqToVals_Device({xplane[x], yplane[x], bplane[x]});
    xplane[x] = src.x;
    yplane[x] = src.y;
    bplane[x] = src.z;
}