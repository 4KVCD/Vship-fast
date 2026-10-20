import frequencyOpsModule;
import exactMathModule;

import xybLowFreqModule;

float SuppressHfInBrightAreas_Device(float hf, float brightness, float mul, float reg) {
    float scaler = mul * reg / (reg + brightness);
    return hf * scaler;
}

float SuppressUhfInBrightAreas_Device(float hf, float brightness, float mul, float reg) {
    float scaler = mul * reg / (reg + brightness);
    return hf * scaler;
}

// libjxl's SuppressXByY: its constant 1 - s and its two fused multiply-adds
float supressXbyY_Device(float x, float yval, float yw){
    const float s = 0.653020556257f;
    const float oneMinusS = 0.346979443743f;
    const float scaler = fma(divide(yw, fma(yval, yval, yw)), oneMinusS, s);
    return x * scaler;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void supressXbyY(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* first, uniform float* second, uniform uint64_t width, uniform float yw){
    uint64_t x = threadIdx.x + blockIdx.x*256;

    if (x >= width) return;

    first[x] = supressXbyY_Device(first[x], second[x], yw);
    //printf("res : %f and %f\n", first[x], second[x]);
}

