// libjxl's MaltaDiffMap and MaltaDiffMapLF (lib/jxl/butteraugli/butteraugli.cc), bit-identical to libjxl's results. Built
// twice: with -DFP64=1 for GPUs with 64-bit floats, with -DFP64=0 for the others (see maltaDiff below).

import exactMathModule;

// A workgroup's 32x32 outputs need the diffs of a 40x40 window: 4 pixels more on each side, the lines' reach.
static const int MALTA_TILE = 32;
static const int MALTA_WINDOW = MALTA_TILE + 8;
groupshared float diffs[MALTA_WINDOW*MALTA_WINDOW];

// One diffs value of the window, dy rows and dx columns away from index o
float tap(int o, int dy, int dx) { return diffs[o + dy*MALTA_WINDOW + dx]; }

// The sum of the squared sums along 16 short lines through the pixel at index o: libjxl's MaltaUnit
// (lib/jxl/butteraugli/butteraugli.cc) with its lines, the order it adds their taps in and its fma.
// MaltaUnitLF's lines have 5 taps, MaltaUnit's 7 or 9.
float MaltaUnitLF(int o) {
    float retval;
    // x grows, y constant
    {
        const float sum = (tap(o, 0, -4) + tap(o, 0, -2)) + (tap(o, 0, 0) + (tap(o, 0, 2) + tap(o, 0, 4)));
        retval = sum*sum;
    }
    // y grows, x constant
    {
        const float sum = (tap(o, -4, 0) + tap(o, -2, 0)) + (tap(o, 0, 0) + (tap(o, 2, 0) + tap(o, 4, 0)));
        retval = fma(sum, sum, retval);
    }
    // both grow
    {
        const float sum = (tap(o, -3, -3) + tap(o, -2, -2)) + (tap(o, 0, 0) + (tap(o, 2, 2) + tap(o, 3, 3)));
        retval = fma(sum, sum, retval);
    }
    // y grows, x shrinks
    {
        const float sum = (tap(o, -3, 3) + tap(o, -2, 2)) + (tap(o, 0, 0) + (tap(o, 2, -2) + tap(o, 3, -3)));
        retval = fma(sum, sum, retval);
    }
    // y grows -4 to 4, x shrinks 1 -> -1
    {
        const float sum = (tap(o, -4, 1) + tap(o, -2, 1)) + (tap(o, 0, 0) + (tap(o, 2, -1) + tap(o, 4, -1)));
        retval = fma(sum, sum, retval);
    }
    //  y grows -4 to 4, x grows -1 -> 1
    {
        const float sum = (tap(o, -4, -1) + tap(o, -2, -1)) + (tap(o, 0, 0) + (tap(o, 2, 1) + tap(o, 4, 1)));
        retval = fma(sum, sum, retval);
    }
    // x grows -4 to 4, y grows -1 to 1
    {
        const float sum = (tap(o, -1, -4) + tap(o, -1, -2)) + (tap(o, 0, 0) + (tap(o, 1, 2) + tap(o, 1, 4)));
        retval = fma(sum, sum, retval);
    }
    // x grows -4 to 4, y shrinks 1 to -1
    {
        const float sum = (tap(o, 1, -4) + tap(o, 1, -2)) + (tap(o, 0, 0) + (tap(o, -1, 2) + tap(o, -1, 4)));
        retval = fma(sum, sum, retval);
    }
    /* 0_________
       1__*______
       2___*_____
       3_________
       4____0____
       5_________
       6_____*___
       7______*__
       8_________ */
    {
        const float sum = (tap(o, -3, -2) + tap(o, -2, -1)) + (tap(o, 0, 0) + (tap(o, 2, 1) + tap(o, 3, 2)));
        retval = fma(sum, sum, retval);
    }
    /* 0_________
       1______*__
       2_____*___
       3_________
       4____0____
       5_________
       6___*_____
       7__*______
       8_________ */
    {
        const float sum = (tap(o, -3, 2) + tap(o, -2, 1)) + (tap(o, 0, 0) + (tap(o, 2, -1) + tap(o, 3, -2)));
        retval = fma(sum, sum, retval);
    }
    /* 0_________
       1_________
       2_*_______
       3__*______
       4____0____
       5______*__
       6_______*_
       7_________
       8_________ */
    {
        const float sum = (tap(o, -2, -3) + tap(o, -1, -2)) + (tap(o, 0, 0) + (tap(o, 1, 2) + tap(o, 2, 3)));
        retval = fma(sum, sum, retval);
    }
    /* 0_________
       1_________
       2_______*_
       3______*__
       4____0____
       5__*______
       6_*_______
       7_________
       8_________ */
    {
        const float sum = (tap(o, -2, 3) + tap(o, -1, 2)) + (tap(o, 0, 0) + (tap(o, 1, -2) + tap(o, 2, -3)));
        retval = fma(sum, sum, retval);
    }
    /* 0_________
       1_________
       2________*
       3______*__
       4____0____
       5__*______
       6*________
       7_________
       8_________ */
    {
        const float sum = (tap(o, 2, -4) + tap(o, 1, -2)) + (tap(o, 0, 0) + (tap(o, -1, 2) + tap(o, -2, 4)));
        retval = fma(sum, sum, retval);
    }
    /* 0_________
       1_________
       2*________
       3__*______
       4____0____
       5______*__
       6________*
       7_________
       8_________ */
    {
        const float sum = (tap(o, -2, -4) + tap(o, -1, -2)) + (tap(o, 0, 0) + (tap(o, 1, 2) + tap(o, 2, 4)));
        retval = fma(sum, sum, retval);
    }
    /* 0__*______
       1_________
       2___*_____
       3_________
       4____0____
       5_________
       6_____*___
       7_________
       8______*__ */
    {
        const float sum = (tap(o, -4, -2) + tap(o, -2, -1)) + (tap(o, 0, 0) + (tap(o, 2, 1) + tap(o, 4, 2)));
        retval = fma(sum, sum, retval);
    }
    /* 0______*__
       1_________
       2_____*___
       3_________
       4____0____
       5_________
       6___*_____
       7_________
       8__*______ */
    {
        const float sum = (tap(o, -4, 2) + tap(o, -2, 1)) + (tap(o, 0, 0) + (tap(o, 2, -1) + tap(o, 4, -2)));
        retval = fma(sum, sum, retval);
    }
    return retval;
}

float MaltaUnit(int o) {
    float retval;
    // x grows, y constant
    {
        const float sum = (((tap(o, 0, -4) + tap(o, 0, -3)) + (tap(o, 0, -2) + tap(o, 0, -1))) + ((tap(o, 0, 0) + tap(o, 0, 1)) + (tap(o, 0, 2) + tap(o, 0, 3)))) + tap(o, 0, 4);
        retval = sum*sum;
    }
    // y grows, x constant
    {
        const float sum = (((tap(o, -4, 0) + tap(o, -3, 0)) + (tap(o, -2, 0) + tap(o, -1, 0))) + ((tap(o, 0, 0) + tap(o, 1, 0)) + (tap(o, 2, 0) + tap(o, 3, 0)))) + tap(o, 4, 0);
        retval = fma(sum, sum, retval);
    }
    // both grow
    {
        const float sum = (tap(o, -3, -3) + tap(o, -2, -2)) + (tap(o, -1, -1) + ((tap(o, 0, 0) + tap(o, 1, 1)) + (tap(o, 2, 2) + tap(o, 3, 3))));
        retval = fma(sum, sum, retval);
    }
    // y grows, x shrinks
    {
        const float sum = (tap(o, -3, 3) + tap(o, -2, 2)) + (tap(o, -1, 1) + ((tap(o, 0, 0) + tap(o, 1, -1)) + (tap(o, 2, -2) + tap(o, 3, -3))));
        retval = fma(sum, sum, retval);
    }
    // y grows -4 to 4, x shrinks 1 -> -1
    {
        const float sum = (((tap(o, -4, 1) + tap(o, -3, 1)) + (tap(o, -2, 1) + tap(o, -1, 0))) + ((tap(o, 0, 0) + tap(o, 1, 0)) + (tap(o, 2, -1) + tap(o, 3, -1)))) + tap(o, 4, -1);
        retval = fma(sum, sum, retval);
    }
    //  y grows -4 to 4, x grows -1 -> 1
    {
        const float sum = (((tap(o, -4, -1) + tap(o, -3, -1)) + (tap(o, -2, -1) + tap(o, -1, 0))) + ((tap(o, 0, 0) + tap(o, 1, 0)) + (tap(o, 2, 1) + tap(o, 3, 1)))) + tap(o, 4, 1);
        retval = fma(sum, sum, retval);
    }
    // x grows -4 to 4, y grows -1 to 1
    {
        const float sum = (((tap(o, -1, -4) + tap(o, -1, -3)) + (tap(o, -1, -2) + tap(o, 0, -1))) + ((tap(o, 0, 0) + tap(o, 0, 1)) + (tap(o, 1, 2) + tap(o, 1, 3)))) + tap(o, 1, 4);
        retval = fma(sum, sum, retval);
    }
    // x grows -4 to 4, y shrinks 1 to -1
    {
        const float sum = (((tap(o, 1, -4) + tap(o, 1, -3)) + (tap(o, 1, -2) + tap(o, 0, -1))) + ((tap(o, 0, 0) + tap(o, 0, 1)) + (tap(o, -1, 2) + tap(o, -1, 3)))) + tap(o, -1, 4);
        retval = fma(sum, sum, retval);
    }
    /* 0_________
       1__*______
       2___*_____
       3___*_____
       4____0____
       5_____*___
       6_____*___
       7______*__
       8_________ */
    {
        const float sum = (tap(o, -3, -2) + tap(o, -2, -1)) + (tap(o, -1, -1) + ((tap(o, 0, 0) + tap(o, 1, 1)) + (tap(o, 2, 1) + tap(o, 3, 2))));
        retval = fma(sum, sum, retval);
    }
    /* 0_________
       1______*__
       2_____*___
       3_____*___
       4____0____
       5___*_____
       6___*_____
       7__*______
       8_________ */
    {
        const float sum = (tap(o, -3, 2) + tap(o, -2, 1)) + (tap(o, -1, 1) + ((tap(o, 0, 0) + tap(o, 1, -1)) + (tap(o, 2, -1) + tap(o, 3, -2))));
        retval = fma(sum, sum, retval);
    }
    /* 0_________
       1_________
       2_*_______
       3__**_____
       4____0____
       5_____**__
       6_______*_
       7_________
       8_________ */
    {
        const float sum = (tap(o, -2, -3) + tap(o, -1, -2)) + (tap(o, -1, -1) + ((tap(o, 0, 0) + tap(o, 1, 1)) + (tap(o, 1, 2) + tap(o, 2, 3))));
        retval = fma(sum, sum, retval);
    }
    /* 0_________
       1_________
       2_______*_
       3_____**__
       4____0____
       5__**_____
       6_*_______
       7_________
       8_________ */
    {
        const float sum = (tap(o, -2, 3) + tap(o, -1, 2)) + (tap(o, -1, 1) + ((tap(o, 0, 0) + tap(o, 1, -1)) + (tap(o, 1, -2) + tap(o, 2, -3))));
        retval = fma(sum, sum, retval);
    }
    /* 0_________
       1_________
       2_________
       3______***
       4___*0*___
       5***______
       6_________
       7_________
       8_________ */
    {
        const float sum = (((tap(o, 1, -4) + tap(o, 1, -3)) + (tap(o, 1, -2) + tap(o, 0, -1))) + ((tap(o, 0, 0) + tap(o, 0, 1)) + (tap(o, -1, 2) + tap(o, -1, 3)))) + tap(o, -1, 4);
        retval = fma(sum, sum, retval);
    }
    /* 0_________
       1_________
       2_________
       3***______
       4___*0*___
       5______***
       6_________
       7_________
       8_________ */
    {
        const float sum = (((tap(o, -1, -4) + tap(o, -1, -3)) + (tap(o, -1, -2) + tap(o, 0, -1))) + ((tap(o, 0, 0) + tap(o, 0, 1)) + (tap(o, 1, 2) + tap(o, 1, 3)))) + tap(o, 1, 4);
        retval = fma(sum, sum, retval);
    }
    /* 0___*_____
       1___*_____
       2___*_____
       3____*____
       4____0____
       5____*____
       6_____*___
       7_____*___
       8_____*___ */
    {
        const float sum = (((tap(o, -4, -1) + tap(o, -3, -1)) + (tap(o, -2, -1) + tap(o, -1, 0))) + ((tap(o, 0, 0) + tap(o, 1, 0)) + (tap(o, 2, 1) + tap(o, 3, 1)))) + tap(o, 4, 1);
        retval = fma(sum, sum, retval);
    }
    /* 0_____*___
       1_____*___
       2____ *___
       3____*____
       4____0____
       5____*____
       6___*_____
       7___*_____
       8___*_____ */
    {
        const float sum = (((tap(o, -4, 1) + tap(o, -3, 1)) + (tap(o, -2, 1) + tap(o, -1, 0))) + ((tap(o, 0, 0) + tap(o, 1, 0)) + (tap(o, 2, -1) + tap(o, 3, -1)))) + tap(o, 4, -1);
        retval = fma(sum, sum, retval);
    }
    return retval;
}


// The value libjxl's MaltaDiffMap puts in its diffs image for one pixel (a of lum0, b of lum1), from its weights
// norm2_0gt1 and norm2_0lt1 (computed in double then rounded, as the host does) and norm1.
//
// libjxl computes the secondary objective in double: too_small = 0.55*|a| and too_big = 1.05*|a|, compared with b,
// then d -/+= scaler2*(the distance of b to them), the result rounded to float. Here it is computed in double-float
// (exactMathModule), which gives libjxl's float whenever the comparison and the rounding are decided with a margin
// far above both computations' errors; otherwise (rarely) the pixel is recomputed in double (FP64) or, on GPUs
// without 64-bit floats, the double-float result is kept, an approximation.
static const float MARGIN = 9.094947e-13f; // 2^-40, relative to the magnitudes involved

float maltaDiff(float a, float b, float norm2_0gt1, float norm2_0lt1, float norm1, uint zero){
    const float absval = 0.5f*(abs(a) + abs(b));
    const float diff = a - b;
    const float scaler = divide(norm2_0gt1, norm1 + absval);
    // primary symmetric quadratic objective
    float d = scaler*diff;
    const float scaler2 = divide(norm2_0lt1, norm1 + absval);

    // secondary half-open quadratic objectives: which distance applies (mode) and whether d gains or loses it
    const float fabs0 = abs(a);
    // a = b = 0, as in areas where both images are flat: too_small and too_big are 0 and b is neither below nor above
    // them, so d is libjxl's result. The tests below cannot decide this case (their margins are 0), and sending it to
    // the double computation makes whole waves take that path, which is slow on GPUs with slow doubles.
    if (fabs0 == 0.0f && b == 0.0f) return d;
    const FF tooSmall = ffMulFloat(ff(0.550000011920929f, -1.1920929132713809e-08f), fabs0, zero);
    const FF tooBig = ffMulFloat(ff(1.0499999523162842f, 4.7683716530855236e-08f), fabs0, zero);
    const float conditionMargin = MARGIN*(tooBig.hi + abs(b));
    bool certain = true;
    bool c;
    FF term = ff(0.0f, 0.0f);
    int sign = 0;
    if (a < 0.0f){
        const FF x = ffAddFloat(tooSmall, b, zero);               // b + too_small > 0: b > -too_small
        const int s = signDecided(x, conditionMargin, c); certain = certain && c;
        if (s > 0){ term = x; sign = -1; }
        else {
            const FF y = ffAddFloat(ffNeg(tooBig), -b, zero);     // -b - too_big > 0: b < -too_big
            const int s2 = signDecided(y, conditionMargin, c); certain = certain && c;
            if (s2 > 0){ term = y; sign = 1; }
        }
    } else {
        const FF x = ffAddFloat(tooSmall, -b, zero);              // too_small - b > 0: b < too_small
        const int s = signDecided(x, conditionMargin, c); certain = certain && c;
        if (s > 0){ term = x; sign = 1; }
        else {
            const FF y = ffAddFloat(ffNeg(tooBig), b, zero);      // b - too_big > 0: b > too_big
            const int s2 = signDecided(y, conditionMargin, c); certain = certain && c;
            if (s2 > 0){ term = y; sign = -1; }
        }
    }
    if (certain && sign == 0) return d;
    if (certain){
        const FF impact = ffMulFloat(term, scaler2, zero);
        const FF result = ffAddFloat((sign > 0) ? impact : ffNeg(impact), d, zero);
        if (decided(result, MARGIN*(abs(d) + abs(impact.hi) + abs(scaler2)*tooBig.hi))) return result.hi;
    }
#if FP64
    // undecided: libjxl's double computation
    const double fabs0d = double(fabs0);
    const double too_small = asdouble(0x9999999au, 0x3fe19999u)*fabs0d; // 0.55
    const double too_big = asdouble(0xcccccccdu, 0x3ff0ccccu)*fabs0d;   // 1.05
    const double bd = double(b);
    const double s2d = double(scaler2);
    if (a < 0.0f){
        if (bd > -too_small) d = float(double(d) - s2d*(bd + too_small));
        else if (bd < -too_big) d = float(double(d) + s2d*(-bd - too_big));
    } else {
        if (bd < too_small) d = float(double(d) + s2d*(too_small - bd));
        else if (bd > too_big) d = float(double(d) - s2d*(bd - too_big));
    }
    return d;
#else
    if (sign == 0) return d;
    const FF impact = ffMulFloat(term, scaler2, zero);
    return ffAddFloat((sign > 0) ? impact : ffNeg(impact), d, zero).hi;
#endif
}

// diffs of the workgroup's window (blockIdx), 0 outside the image, as libjxl's padded units read them
void loadDiffs(uint3 threadIdx, uint3 blockIdx, float* lum0, float* lum1, uint64_t width, uint64_t height, float norm2_0gt1, float norm2_0lt1, float norm1, uint zero){
    const int left = (int)blockIdx.x*MALTA_TILE - 4;
    const int top = (int)blockIdx.y*MALTA_TILE - 4;
    for (uint i = threadIdx.x + threadIdx.y*16; i < MALTA_WINDOW*MALTA_WINDOW; i += 256){
        const int x = left + (int)(i % MALTA_WINDOW);
        const int y = top + (int)(i / MALTA_WINDOW);
        diffs[i] = (x < 0 || x >= (int)width || y < 0 || y >= (int)height) ? 0.0f : maltaDiff(lum0[y*width + x], lum1[y*width + x], norm2_0gt1, norm2_0lt1, norm1, zero);
    }
    GroupMemoryBarrierWithGroupSync();
}

// block_diff_ac += libjxl's MaltaDiffMap: each thread adds the MaltaUnit of 4 of the workgroup's 32x32 outputs
[shader("compute")]
[numthreads(16, 16, 1)]
void MaltaDiffMap(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* lum0, uniform float* lum1, uniform float* block_diff_ac, uniform uint64_t width, uniform uint64_t height, uniform float norm2_0gt1, uniform float norm2_0lt1, uniform float norm1, uniform uint zero) {
    loadDiffs(threadIdx, blockIdx, lum0, lum1, width, height, norm2_0gt1, norm2_0lt1, norm1, zero);
    [ForceUnroll]
    for (int k = 0; k < 4; k++){
        const int ox = (int)threadIdx.x + 16*(k % 2);
        const int oy = (int)threadIdx.y + 16*(k / 2);
        const uint64_t x = blockIdx.x*MALTA_TILE + ox;
        const uint64_t y = blockIdx.y*MALTA_TILE + oy;
        const float result = MaltaUnit((oy + 4)*MALTA_WINDOW + ox + 4);
        if (x < width && y < height) block_diff_ac[y*width + x] += result;
    }
}

// block_diff_ac += libjxl's MaltaDiffMapLF, as MaltaDiffMap with MaltaUnitLF's lines
[shader("compute")]
[numthreads(16, 16, 1)]
void MaltaDiffMapLF(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* lum0, uniform float* lum1, uniform float* block_diff_ac, uniform uint64_t width, uniform uint64_t height, uniform float norm2_0gt1, uniform float norm2_0lt1, uniform float norm1, uniform uint zero) {
    loadDiffs(threadIdx, blockIdx, lum0, lum1, width, height, norm2_0gt1, norm2_0lt1, norm1, zero);
    [ForceUnroll]
    for (int k = 0; k < 4; k++){
        const int ox = (int)threadIdx.x + 16*(k % 2);
        const int oy = (int)threadIdx.y + 16*(k / 2);
        const uint64_t x = blockIdx.x*MALTA_TILE + ox;
        const uint64_t y = blockIdx.y*MALTA_TILE + oy;
        const float result = MaltaUnitLF((oy + 4)*MALTA_WINDOW + ox + 4);
        if (x < width && y < height) block_diff_ac[y*width + x] += result;
    }
}