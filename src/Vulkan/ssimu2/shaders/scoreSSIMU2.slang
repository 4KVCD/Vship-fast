// SSIMULACRA2's error maps of one plane at one scale, computed like libjxl's tools/ssimulacra2.cc, the reference
// the scores are compared with.
//
// The 5 signals SSIMMap and EdgeDiffMap need are blurred: img1*img1, img2*img2, img1*img2, img1 and img2. The blur
// is libjxl's recursive Gaussian (tools/gauss_blur.cc): a recurrence along each row (blurRows) then along each
// column (blurColumnsAndScore), zero outside the image. Both passes run libjxl's arithmetic in the same order (the
// rows in libjxl's blocks of 4 outputs) so that they round like libjxl's. blurColumnsAndScore uses each blurred
// pixel right away for the SSIM and edge difference maps and sums them, and the scores are finished on the CPU.
//
// The coefficient buffer holds mul_in, mul_prev and mul_prev2 (3 terms of 4 lanes each), then n2 and d1
// (GaussianHandle in gaussianblur.hpp).

static const int RADIUS = 5;  // recursiveGaussianRadius
static const int SIGNALS = 5; // img1*img1, img2*img2, img1*img2, img1, img2
static const int THREADS = 64;
static const float kC2 = 0.0009f;

// Returns x. Its bits go through an OR with a zero only known at run time, so the compiler cannot fuse the
// multiplication that produced x into an addition: libjxl rounds these products before adding them.
float unfused(float x, uint zero){
    return asfloat(asuint(x) | zero);
}

// The value of signal s for the pixel values a (img1) and b (img2).
float signal(int s, float a, float b, uint zero){
    if (s == 0) return unfused(a*a, zero);
    if (s == 1) return unfused(b*b, zero);
    if (s == 2) return unfused(a*b, zero);
    if (s == 3) return a;
    return b;
}

// a/b correctly rounded, like the CPU's division; Vulkan allows the GPU's own division to be 2.5 ulp off.
float divide(float a, float b){
    const float r = 1.0f/b;
    const float q = a*r;
    const float e = fma(-q, b, a);
    return fma(e, r, q);
}

// Horizontal pass, libjxl's FastGaussian1D: one thread per row computes the 5 signals' blurred rows into rows
// (5 planes of width*height).
[shader("compute")]
[numthreads(THREADS, 1, 1)]
void blurRows(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* img1, uniform float* img2, uniform float* rows, uniform float* coefficients, uniform int64_t width64, uniform int64_t height64, uniform uint zero){
    const int width = (int)width64;
    const int height = (int)height64;
    const int y = threadIdx.x + blockIdx.x*THREADS;
    if (y >= height) return;
    const uint64_t planeSize = (uint64_t)width*(uint64_t)height;
    float* row1 = img1 + (uint64_t)y*(uint64_t)width;
    float* row2 = img2 + (uint64_t)y*(uint64_t)width;
    float* out = rows + (uint64_t)y*(uint64_t)width;

    float4 mulIn[3], mulPrev[3], mulPrev2[3];
    [ForceUnroll]
    for (int k = 0; k < 3; k++){
        mulIn[k] = float4(coefficients[4*k], coefficients[4*k+1], coefficients[4*k+2], coefficients[4*k+3]);
        mulPrev[k] = float4(coefficients[12+4*k], coefficients[12+4*k+1], coefficients[12+4*k+2], coefficients[12+4*k+3]);
        mulPrev2[k] = float4(coefficients[24+4*k], coefficients[24+4*k+1], coefficients[24+4*k+2], coefficients[24+4*k+3]);
    }

    float prev[SIGNALS][3];
    float prev2[SIGNALS][3];
    [ForceUnroll]
    for (int s = 0; s < SIGNALS; s++){
        [ForceUnroll]
        for (int k = 0; k < 3; k++){
            prev[s][k] = 0.0f;
            prev2[s][k] = 0.0f;
        }
    }

    // One output at a time, with zero outside the row: the first outputs and those near the end.
    int n = -RADIUS + 1;
    const int firstAligned = 8; // libjxl's RoundUpTo(RADIUS + 1, 4)
    for (int pass = 0; pass < 2; pass++){
        const int end = (pass == 0) ? min(firstAligned, width) : width;
        for (; n < end; n++){
            const int left = n - RADIUS - 1;
            const int right = n + RADIUS - 1;
            const float a0 = (left >= 0) ? row1[left] : 0.0f;
            const float b0 = (left >= 0) ? row2[left] : 0.0f;
            const float a1 = (right < width) ? row1[right] : 0.0f;
            const float b1 = (right < width) ? row2[right] : 0.0f;
            [ForceUnroll]
            for (int s = 0; s < SIGNALS; s++){
                const float sum = ((left >= 0) ? signal(s, a0, b0, zero) : 0.0f) + ((right < width) ? signal(s, a1, b1, zero) : 0.0f);
                float o[3];
                [ForceUnroll]
                for (int k = 0; k < 3; k++){
                    o[k] = sum*mulIn[k].x;
                    o[k] = fma(mulPrev2[k].x, prev2[s][k], o[k]);
                    prev2[s][k] = prev[s][k];
                    o[k] = fma(mulPrev[k].x, prev[s][k], o[k]);
                    prev[s][k] = o[k];
                }
                if (n >= 0) out[s*planeSize + n] = o[0] + (o[1] + o[2]);
            }
        }
        if (pass == 1) break;

        // 4 outputs at a time without bounds checks, as libjxl's unrolled loop: output n+L gets the 4 sums of
        // the block, sum j with coefficient mulIn[L-j] (none for j > L), then the previous two outputs.
        for (; n < width - RADIUS + 1 - 3; n += 4){
            float a[8], b[8];
            [ForceUnroll]
            for (int j = 0; j < 4; j++){
                a[j] = row1[n + j - RADIUS - 1];
                b[j] = row2[n + j - RADIUS - 1];
                a[4+j] = row1[n + j + RADIUS - 1];
                b[4+j] = row2[n + j + RADIUS - 1];
            }
            [ForceUnroll]
            for (int s = 0; s < SIGNALS; s++){
                float4 sum;
                [ForceUnroll]
                for (int j = 0; j < 4; j++) sum[j] = signal(s, a[j], b[j], zero) + signal(s, a[4+j], b[4+j], zero);
                float4 o[3];
                [ForceUnroll]
                for (int k = 0; k < 3; k++){
                    o[k] = sum.x*mulIn[k];
                    [ForceUnroll]
                    for (int L = 1; L < 4; L++){
                        [ForceUnroll]
                        for (int j = 1; j < 4; j++) if (j <= L) o[k][L] = fma(mulIn[k][L-j], sum[j], o[k][L]);
                    }
                    o[k] = fma(mulPrev2[k], float4(prev2[s][k]), o[k]);
                    o[k] = fma(mulPrev[k], float4(prev[s][k]), o[k]);
                    prev2[s][k] = o[k].z;
                    prev[s][k] = o[k].w;
                }
                const float4 result = o[0] + (o[1] + o[2]);
                [ForceUnroll]
                for (int L = 0; L < 4; L++) out[s*planeSize + n + L] = result[L];
            }
        }
    }
}

// Vertical pass, libjxl's FastGaussianVertical, one thread per column, and the error maps of SSIMMap and
// EdgeDiffMap for each blurred pixel. Each workgroup writes the sums over its columns of the ssim error, the
// ringing (artifact) and the blurring (detail lost) and of their 4th powers into sums[6*blockIdx.x...].
groupshared float partialSums[6][THREADS];

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void blurColumnsAndScore(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* img1, uniform float* img2, uniform float* rows, uniform float* coefficients, uniform float* sums, uniform int64_t width64, uniform int64_t height64, uniform uint zero){
    const int width = (int)width64;
    const int height = (int)height64;
    const int x = threadIdx.x + blockIdx.x*THREADS;
    const uint64_t planeSize = (uint64_t)width*(uint64_t)height;

    float n2[3], d1[3];
    [ForceUnroll]
    for (int k = 0; k < 3; k++){
        n2[k] = coefficients[36+k];
        d1[k] = coefficients[39+k];
    }

    // Compensated (Kahan) sums: a column can be thousands of pixels long.
    float total[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float compensation[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    if (x < width){
        float y1[SIGNALS][3]; // outputs n-1 and n-2 of each term of each signal
        float y2[SIGNALS][3];
        [ForceUnroll]
        for (int s = 0; s < SIGNALS; s++){
            [ForceUnroll]
            for (int k = 0; k < 3; k++){
                y1[s][k] = 0.0f;
                y2[s][k] = 0.0f;
            }
        }

        for (int n = -RADIUS + 1; n < height; n++){
            const int top = n - RADIUS - 1;
            const int bottom = n + RADIUS - 1;
            float blurred[SIGNALS];
            [ForceUnroll]
            for (int s = 0; s < SIGNALS; s++){
                float* plane = rows + s*planeSize + x;
                const float input = ((top >= 0) ? plane[(uint64_t)top*(uint64_t)width] : 0.0f) + ((bottom < height) ? plane[(uint64_t)bottom*(uint64_t)width] : 0.0f);
                [ForceUnroll]
                for (int k = 0; k < 3; k++){
                    const float yk = fma(n2[k], input, fma(-d1[k], y1[s][k], -y2[s][k]));
                    y2[s][k] = y1[s][k];
                    y1[s][k] = yk;
                }
                blurred[s] = y1[s][0] + (y1[s][1] + y1[s][2]);
            }
            if (n >= 0){ //the first RADIUS-1 steps only fill the recurrence
                const uint64_t pixel = (uint64_t)n*(uint64_t)width + (uint64_t)x;
                const float mu1 = blurred[3];
                const float mu2 = blurred[4];

                // SSIMMap
                const float mu11 = unfused(mu1*mu1, zero);
                const float mu22 = unfused(mu2*mu2, zero);
                const float mu12 = unfused(mu1*mu2, zero);
                const float muDiff = mu1 - mu2;
                const float num_m = 1.0f - unfused(muDiff*muDiff, zero);
                const float num_s = 2.0f*(blurred[2] - mu12) + kC2;
                const float denom_s = ((blurred[0] - mu11) + (blurred[1] - mu22)) + kC2;
                const float ssim = max(1.0f - divide(unfused(num_m*num_s, zero), denom_s), 0.0f);

                // EdgeDiffMap: (1 + |img2 - mu2|)/(1 + |img1 - mu1|) - 1, written as a difference over 1 + |img1 - mu1|
                // so that float keeps the precision libjxl gets from double.
                const float edge1 = abs(img1[pixel] - mu1);
                const float edge2 = abs(img2[pixel] - mu2);
                const float d = divide(edge2 - edge1, 1.0f + edge1);
                const float artifact = max(d, 0.0f);
                const float detailLost = max(-d, 0.0f);

                const float values[6] = {ssim, artifact, detailLost, (ssim*ssim)*(ssim*ssim), (artifact*artifact)*(artifact*artifact), (detailLost*detailLost)*(detailLost*detailLost)};
                [ForceUnroll]
                for (int i = 0; i < 6; i++){
                    const float v = values[i] - compensation[i];
                    const float t = total[i] + v;
                    compensation[i] = (t - total[i]) - v;
                    total[i] = t;
                }
            }
        }
    }

    [ForceUnroll]
    for (int i = 0; i < 6; i++) partialSums[i][threadIdx.x] = total[i];
    GroupMemoryBarrierWithGroupSync();
    for (int next = THREADS/2; next > 0; next /= 2){
        if (threadIdx.x < next){
            [ForceUnroll]
            for (int i = 0; i < 6; i++) partialSums[i][threadIdx.x] += partialSums[i][threadIdx.x + next];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (threadIdx.x == 0){
        [ForceUnroll]
        for (int i = 0; i < 6; i++) sums[6*blockIdx.x + i] = partialSums[i][0];
    }
}
