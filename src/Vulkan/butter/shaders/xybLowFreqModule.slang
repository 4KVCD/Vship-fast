module xybLowFreqModule;

// libjxl's XybLowFreqToVals: the low frequency planes' values as Butteraugli compares them
public float3 XybLowFreqToVals_Device(float3 xyb) {
    const float xmuli = 33.832837186260f;
    const float ymuli = 14.458268100570f;
    const float bmuli = 49.87984651440f;
    const float y_to_b_muli = -0.362267051518f;
    xyb.z = fma(y_to_b_muli, xyb.y, xyb.z); // libjxl: MulAdd
    xyb.z = xyb.z * bmuli;
    xyb.x = xyb.x * xmuli;
    xyb.y = xyb.y * ymuli;
    return xyb;
}
