module frequencyOpsModule;

// The range operations of Butteraugli's frequency separation (libjxl's butteraugli.cc)
// libjxl's MaximumClamp: its constant and its fused multiply-add
public float MaximumClamp_Device(float v, float maxval) {
    const float kMul = 0.724216145665f;
    if (v >= maxval) return fma(v - maxval, kMul, maxval);
    if (v < -maxval) return fma(v + maxval, kMul, -maxval);
    return v;
}

// Make area around zero more important (2x it until the limit).
public float AmplifyRangeAroundZero_Device(float x, float w) {
    return x > w ? x + w : x < -w ? x - w : 2.0f * x;
}

// Make area around zero less important (remove it).
public float RemoveRangeAroundZero_Device(float x, float w) {
    return x > w ? x - w : x < -w ? x + w : 0.0f;
}

// The former separateHf_Uhf kernel's update of a pixel's high and ultra high frequencies.
public void separateHfUhf(inout float hf, inout float uhf){
    const float kMaxclampHf = 28.4691806922f;
    const float kMaxclampUhf = 5.19175294647f;

    hf = MaximumClamp_Device(hf, kMaxclampHf);
    uhf -= hf;
    uhf = MaximumClamp_Device(uhf, kMaxclampUhf);
    uhf *= 2.69313763794f;
    hf *= 2.155f;
    hf = AmplifyRangeAroundZero_Device(hf, 0.132f);
}
