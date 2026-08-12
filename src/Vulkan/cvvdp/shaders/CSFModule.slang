module CSFModule;

static float sensitivity_correction = -0.2797423303127289;

//src is Lbkg of ref
public float computeSensitivityGPU(float src, int band, int channel, float* CSFMem_d){
    float* LUTx = CSFMem_d;
    float* LUTy = CSFMem_d+32*(1+band*4+channel);
    const float x = log10(src);
    float frac = 31*(x - LUTx[0])/(LUTx[31] - LUTx[0]);
    const int imin = max(0,min(30,(int)frac));
    const int imax = imin+1;
    frac = max(0.f, min(1.f, frac-imin));
    const float logS = LUTy[imin] * (1-frac) + LUTy[imax] * frac + sensitivity_correction/20.f;
    //if (threadIdx.x + blockIdx.x == 0) printf("SensitivityCompute: inp log: %f, imin %d imax %d frac %f -> logS %f, -> S %f\n", x, imin, imax, frac, logS, powf(10, logS));
    return pow(10, logS);
}
