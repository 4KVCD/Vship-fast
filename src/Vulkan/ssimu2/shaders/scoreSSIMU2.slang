// SSIMULACRA2's error maps of each plane at each scale, computed like libjxl's tools/ssimulacra2.cc, the reference
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

// A dispatch scores the planes firstPlane .. firstPlane + planes - 1 (blockIdx.y) of the scales firstScale ..
// (blockIdx.z), each such job in its own part of the rows buffer: the full scale one plane at a time, and the smaller
// scales, whose rows fit together in the same buffer, all at once so that the GPU has enough work. The images hold
// each plane's scales one after the other, planeStride floats apart, and the sums are in libjxl's order: for each
// scale, for each plane, the sums of each workgroup.
struct Job {
    int width;
    int height;
    uint64_t image;     // offset of the plane at this scale in img1 and img2
    uint64_t rows;      // offset of the job's 5 blurred planes in rows
    uint64_t sums;      // offset of the job's sums in sums
};

Job findJob(int64_t planeStride, int64_t baseWidth, int64_t baseHeight, int firstScale, int firstPlane, int planes, uint3 blockIdx){
    const int scale = firstScale + (int)blockIdx.z;
    const int slot = (int)blockIdx.y;
    Job job;
    uint64_t w = (uint64_t)baseWidth;
    uint64_t h = (uint64_t)baseHeight;
    job.image = 0;
    job.rows = 0;
    job.sums = 0;
    for (int sc = 0; sc < scale; sc++){
        job.image += w*h;
        job.sums += 3*6*((w - 1)/THREADS + 1);
        if (sc >= firstScale) job.rows += (uint64_t)planes*SIGNALS*w*h;
        w = (w - 1)/2 + 1;
        h = (h - 1)/2 + 1;
    }
    const int plane = firstPlane + slot;
    job.image += (uint64_t)plane*(uint64_t)planeStride;
    job.rows += (uint64_t)slot*SIGNALS*w*h;
    job.sums += (uint64_t)plane*6*((w - 1)/THREADS + 1);
    job.width = (int)w;
    job.height = (int)h;
    return job;
}

// Horizontal pass, libjxl's FastGaussian1D. Its recurrence is a sum of TERMS independent recurrences, added up per
// output: a workgroup takes ROW_GROUP rows and gives each (signal, term) pair ROW_GROUP threads, one row each. The
// rows go through shared memory CHUNK columns at a time so that the image is read and the 5 blurred planes written
// coalesced.
static const int TERMS = 3;
static const int ROW_GROUP = 16;
// 16 rather than 32: the shared memory of a workgroup halves, so twice as many fit on a Radeon 8060S to hide the
// latency of the serial recurrence. A multiple of 4, so libjxl's blocks of 4 outputs never straddle two chunks.
static const int CHUNK = 16;
static const int CHUNK_IN = CHUNK + 2*RADIUS;  // inputs n-RADIUS-1 .. n+RADIUS-1 of the chunk's outputs
static const int IN_PITCH = CHUNK_IN + 1;      // odd, so that lanes reading one column each of their rows do not conflict
static const int OUT_PITCH = CHUNK + 1;
static const int ROW_THREADS = ROW_GROUP*SIGNALS*TERMS;
groupshared float signalsIn[SIGNALS][ROW_GROUP*IN_PITCH];
groupshared float termsOut[SIGNALS*TERMS][ROW_GROUP*OUT_PITCH]; // term k of signal s at s*TERMS + k

// A chunk's inputs, read into registers by loadChunk with all the loads in flight at once, and stored as the 5 signals
// into signalsIn by storeChunk: the next chunk is read while the current one is blurred, so the wait for memory is
// hidden, and each signal value is computed once instead of by each of its terms' threads.
static const int CHUNK_LOADS = (ROW_GROUP*CHUNK_IN + ROW_THREADS - 1)/ROW_THREADS;

// Element i of the chunk's inputs is column c0 - RADIUS - 1 + i % CHUNK_IN of row y0 + i / CHUNK_IN, 0 outside the image.
void loadChunk(int thread, int c0, int y0, int width, int height, float* img1, float* img2, out float in1[CHUNK_LOADS], out float in2[CHUNK_LOADS]){
    [ForceUnroll]
    for (int j = 0; j < CHUNK_LOADS; j++){
        const int i = thread + j*ROW_THREADS;
        const int r = i / CHUNK_IN;
        const int x = c0 - RADIUS - 1 + i - r*CHUNK_IN;
        const int y = y0 + r;
        const bool inside = i < ROW_GROUP*CHUNK_IN && x >= 0 && x < width && y < height;
        const uint64_t pixel = inside ? (uint64_t)y*(uint64_t)width + (uint64_t)x : 0;
        in1[j] = inside ? img1[pixel] : 0.0f;
        in2[j] = inside ? img2[pixel] : 0.0f;
    }
}

void storeChunk(int thread, float in1[CHUNK_LOADS], float in2[CHUNK_LOADS], uint zero){
    [ForceUnroll]
    for (int j = 0; j < CHUNK_LOADS; j++){
        const int i = thread + j*ROW_THREADS;
        const int r = i / CHUNK_IN;
        if (i < ROW_GROUP*CHUNK_IN){
            [ForceUnroll]
            for (int s = 0; s < SIGNALS; s++) signalsIn[s][r*IN_PITCH + i - r*CHUNK_IN] = signal(s, in1[j], in2[j], zero);
        }
    }
}

[shader("compute")]
[numthreads(ROW_GROUP, SIGNALS*TERMS, 1)]
void blurRows(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* img1Base, uniform float* img2Base, uniform float* rowsBase, uniform float* coefficients, uniform int64_t planeStride, uniform int64_t baseWidth, uniform int64_t baseHeight, uniform int firstScale, uniform int firstPlane, uniform int planes, uniform uint zero){
    const Job job = findJob(planeStride, baseWidth, baseHeight, firstScale, firstPlane, planes, blockIdx);
    const int width = job.width;
    const int height = job.height;
    if ((int)blockIdx.x*ROW_GROUP >= height) return; // the dispatch has the workgroups of its largest job
    float* img1 = img1Base + job.image;
    float* img2 = img2Base + job.image;
    float* rows = rowsBase + job.rows;
    const int lane = threadIdx.x; // the row of this thread
    const int s = threadIdx.y / TERMS;
    const int k = threadIdx.y % TERMS;
    const int thread = threadIdx.x + threadIdx.y*ROW_GROUP;
    const int y0 = blockIdx.x*ROW_GROUP;
    const uint64_t planeSize = (uint64_t)width*(uint64_t)height;

    const float4 mulIn = float4(coefficients[4*k], coefficients[4*k+1], coefficients[4*k+2], coefficients[4*k+3]);
    const float4 mulPrev = float4(coefficients[12+4*k], coefficients[12+4*k+1], coefficients[12+4*k+2], coefficients[12+4*k+3]);
    const float4 mulPrev2 = float4(coefficients[24+4*k], coefficients[24+4*k+1], coefficients[24+4*k+2], coefficients[24+4*k+3]);
    float prev = 0.0f;
    float prev2 = 0.0f;
    const int termIndex = threadIdx.y; // s*TERMS + k

    const int firstAligned = 8;                    // libjxl's RoundUpTo(RADIUS + 1, 4): outputs before it are one at a time
    const int blocksEnd = width - RADIUS + 1 - 3;  // libjxl's blocks of 4 start before this
    int n = -RADIUS + 1;
    float in1[CHUNK_LOADS], in2[CHUNK_LOADS];
    loadChunk(thread, 0, y0, width, height, img1, img2, in1, in2);
    for (int c0 = 0; c0 < width; c0 += CHUNK){
        // signalsIn[.][r*IN_PITCH + i] is column c0 - RADIUS - 1 + i of row y0 + r
        storeChunk(thread, in1, in2, zero);
        GroupMemoryBarrierWithGroupSync();
        if (c0 + CHUNK < width) loadChunk(thread, c0 + CHUNK, y0, width, height, img1, img2, in1, in2);

        const int rowBase = lane*IN_PITCH - (c0 - RADIUS - 1); // signalsIn index of column 0 of this lane's row
        const int end = min(c0 + CHUNK, width);
        while (n < end){
            if (n >= firstAligned && n < blocksEnd){
                // 4 outputs at a time, as libjxl's unrolled loop: output n+L gets the 4 sums of the block, sum j
                // with coefficient mulIn[L-j] (none for j > L), then the previous two outputs.
                float4 sum;
                [ForceUnroll]
                for (int j = 0; j < 4; j++) sum[j] = signalsIn[s][rowBase + n + j - RADIUS - 1] + signalsIn[s][rowBase + n + j + RADIUS - 1];
                float4 o = sum.x*mulIn;
                [ForceUnroll]
                for (int L = 1; L < 4; L++){
                    [ForceUnroll]
                    for (int j = 1; j < 4; j++) if (j <= L) o[L] = fma(mulIn[L-j], sum[j], o[L]);
                }
                o = fma(mulPrev2, float4(prev2), o);
                o = fma(mulPrev, float4(prev), o);
                prev2 = o.z;
                prev = o.w;
                [ForceUnroll]
                for (int L = 0; L < 4; L++) termsOut[termIndex][lane*OUT_PITCH + n - c0 + L] = o[L];
                n += 4;
            } else {
                // one output, with zero outside the row
                const int left = n - RADIUS - 1;
                const int right = n + RADIUS - 1;
                const float sum = ((left >= 0) ? signalsIn[s][rowBase + left] : 0.0f) + ((right < width) ? signalsIn[s][rowBase + right] : 0.0f);
                float o = sum*mulIn.x;
                o = fma(mulPrev2.x, prev2, o);
                prev2 = prev;
                o = fma(mulPrev.x, prev, o);
                prev = o;
                if (n >= 0) termsOut[termIndex][lane*OUT_PITCH + n - c0] = o;
                n++;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // the outputs, term 0 + (term 1 + term 2) as libjxl adds them, written one row of a signal at a time
        for (int i = thread; i < SIGNALS*ROW_GROUP*CHUNK; i += ROW_THREADS){
            const int signalIndex = i / (ROW_GROUP*CHUNK);
            const int r = (i / CHUNK) % ROW_GROUP;
            const int column = i % CHUNK;
            const int y = y0 + r;
            const int x = c0 + column;
            const int index = r*OUT_PITCH + column;
            if (y < height && x < width) rows[signalIndex*planeSize + (uint64_t)y*(uint64_t)width + (uint64_t)x] = termsOut[signalIndex*TERMS][index] + (termsOut[signalIndex*TERMS + 1][index] + termsOut[signalIndex*TERMS + 2][index]);
        }
        // no barrier needed here: termsOut is written again only after the next chunk's first barrier
    }
}

// Vertical pass, libjxl's FastGaussianVertical, and the error maps of SSIMMap and EdgeDiffMap for each blurred
// pixel. A workgroup takes COLUMN_GROUP columns: each of its SIGNALS pairs of warps runs one signal's recurrence,
// one column per thread, ROW_CHUNK rows at a time. Then all threads compute the maps of these rows and the first
// pair of warps adds them up, one column per thread in row order. Each workgroup writes the sums over its columns
// of the ssim error, the ringing (artifact) and the blurring (detail lost) and of their 4th powers into
// sums[6*blockIdx.x...].
static const int COLUMN_GROUP = THREADS;
// 8 rather than 16, for the same reason as CHUNK in blurRows
static const int ROW_CHUNK = 8;
static const int MAP_ROWS = (ROW_CHUNK + SIGNALS - 1)/SIGNALS; // the rows of a chunk whose maps one thread computes
groupshared float blurredChunk[SIGNALS][ROW_CHUNK*COLUMN_GROUP];
groupshared float valuesChunk[3][ROW_CHUNK*COLUMN_GROUP]; // ssim, artifact, detail lost
groupshared float columnSums[6][COLUMN_GROUP];

// A chunk's memory reads are issued together into registers, the next chunk's while the current one is computed,
// so that the wait for memory is hidden.

// The recurrence inputs of the outputs r0 .. r0 + ROW_CHUNK - 1: rows n - RADIUS - 1 and n + RADIUS - 1, 0 outside
// the image.
void loadColumnChunk(float* plane, int r0, int width, int height, out float topIn[ROW_CHUNK], out float bottomIn[ROW_CHUNK]){
    [ForceUnroll]
    for (int i = 0; i < ROW_CHUNK; i++){
        const int top = r0 + i - RADIUS - 1;
        const int bottom = r0 + i + RADIUS - 1;
        topIn[i] = (top >= 0 && top < height) ? plane[(uint64_t)top*(uint64_t)width] : 0.0f;
        bottomIn[i] = (bottom < height) ? plane[(uint64_t)bottom*(uint64_t)width] : 0.0f;
    }
}

// The pixels of the rows r0 + s + m*SIGNALS whose maps thread s computes
void loadMapPixels(float* img1, float* img2, int s, int r0, int x, int width, int height, out float pixels1[MAP_ROWS], out float pixels2[MAP_ROWS]){
    [ForceUnroll]
    for (int m = 0; m < MAP_ROWS; m++){
        const int y = r0 + s + m*SIGNALS;
        const uint64_t pixel = (uint64_t)min(y, height - 1)*(uint64_t)width + (uint64_t)x;
        pixels1[m] = img1[pixel];
        pixels2[m] = img2[pixel];
    }
}

[shader("compute")]
[numthreads(COLUMN_GROUP, SIGNALS, 1)]
void blurColumnsAndScore(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* img1Base, uniform float* img2Base, uniform float* rowsBase, uniform float* coefficients, uniform float* sumsBase, uniform int64_t planeStride, uniform int64_t baseWidth, uniform int64_t baseHeight, uniform int firstScale, uniform int firstPlane, uniform int planes, uniform uint zero){
    const Job job = findJob(planeStride, baseWidth, baseHeight, firstScale, firstPlane, planes, blockIdx);
    const int width = job.width;
    const int height = job.height;
    if ((int)blockIdx.x*COLUMN_GROUP >= width) return; // the dispatch has the workgroups of its largest job
    float* img1 = img1Base + job.image;
    float* img2 = img2Base + job.image;
    float* rows = rowsBase + job.rows;
    float* sums = sumsBase + job.sums;
    const int lane = threadIdx.x;
    const int s = threadIdx.y; // the signal this thread blurs
    const int x = lane + blockIdx.x*COLUMN_GROUP;
    const bool inside = x < width;
    const int column = inside ? x : 0; // a column read but not used by the threads outside the image
    const uint64_t planeSize = (uint64_t)width*(uint64_t)height;

    float n2[3], d1[3];
    [ForceUnroll]
    for (int k = 0; k < 3; k++){
        n2[k] = coefficients[36+k];
        d1[k] = coefficients[39+k];
    }
    float y1[3] = {0.0f, 0.0f, 0.0f}; // outputs n-1 and n-2 of each term
    float y2[3] = {0.0f, 0.0f, 0.0f};
    float* plane = rows + s*planeSize + column;

    // the first RADIUS-1 steps, outputs -RADIUS+1 .. -1, only fill the recurrence
    for (int n = -RADIUS + 1; n < 0; n++){
        const int bottom = n + RADIUS - 1;
        const float input = 0.0f + ((bottom < height) ? plane[(uint64_t)bottom*(uint64_t)width] : 0.0f);
        [ForceUnroll]
        for (int k = 0; k < 3; k++){
            const float yk = fma(n2[k], input, fma(-d1[k], y1[k], -y2[k]));
            y2[k] = y1[k];
            y1[k] = yk;
        }
    }

    // Compensated (Kahan) sums of the first pair of warps: a column can be thousands of pixels long.
    float total[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float compensation[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    float topIn[ROW_CHUNK], bottomIn[ROW_CHUNK], pixels1[MAP_ROWS], pixels2[MAP_ROWS];
    loadColumnChunk(plane, 0, width, height, topIn, bottomIn);
    loadMapPixels(img1, img2, s, 0, column, width, height, pixels1, pixels2);
    for (int r0 = 0; r0 < height; r0 += ROW_CHUNK){
        const int rowsInChunk = min(ROW_CHUNK, height - r0);
        [ForceUnroll]
        for (int i = 0; i < ROW_CHUNK; i++){
            if (i < rowsInChunk){
                const float input = topIn[i] + bottomIn[i];
                [ForceUnroll]
                for (int k = 0; k < 3; k++){
                    const float yk = fma(n2[k], input, fma(-d1[k], y1[k], -y2[k]));
                    y2[k] = y1[k];
                    y1[k] = yk;
                }
                blurredChunk[s][i*COLUMN_GROUP + lane] = y1[0] + (y1[1] + y1[2]);
            }
        }
        GroupMemoryBarrierWithGroupSync();

        float chunkPixels1[MAP_ROWS] = pixels1, chunkPixels2[MAP_ROWS] = pixels2;
        if (r0 + ROW_CHUNK < height){
            loadColumnChunk(plane, r0 + ROW_CHUNK, width, height, topIn, bottomIn);
            loadMapPixels(img1, img2, s, r0 + ROW_CHUNK, column, width, height, pixels1, pixels2);
        }

        [ForceUnroll]
        for (int m = 0; m < MAP_ROWS; m++){
            const int rr = s + m*SIGNALS;
            if (rr < rowsInChunk){
                const int index = rr*COLUMN_GROUP + lane;
                const float mu1 = blurredChunk[3][index];
                const float mu2 = blurredChunk[4][index];

                // SSIMMap
                const float mu11 = unfused(mu1*mu1, zero);
                const float mu22 = unfused(mu2*mu2, zero);
                const float mu12 = unfused(mu1*mu2, zero);
                const float muDiff = mu1 - mu2;
                const float num_m = 1.0f - unfused(muDiff*muDiff, zero);
                const float num_s = 2.0f*(blurredChunk[2][index] - mu12) + kC2;
                const float denom_s = ((blurredChunk[0][index] - mu11) + (blurredChunk[1][index] - mu22)) + kC2;
                const float ssim = max(1.0f - divide(unfused(num_m*num_s, zero), denom_s), 0.0f);

                // EdgeDiffMap: (1 + |img2 - mu2|)/(1 + |img1 - mu1|) - 1, written as a difference over 1 + |img1 - mu1|
                // so that float keeps the precision libjxl gets from double.
                const float edge1 = abs(chunkPixels1[m] - mu1);
                const float edge2 = abs(chunkPixels2[m] - mu2);
                const float d = divide(edge2 - edge1, 1.0f + edge1);

                valuesChunk[0][index] = ssim;
                valuesChunk[1][index] = max(d, 0.0f);  // artifact
                valuesChunk[2][index] = max(-d, 0.0f); // detail lost
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (s == 0 && inside){
            for (int rr = 0; rr < rowsInChunk; rr++){
                const float ssim = valuesChunk[0][rr*COLUMN_GROUP + lane];
                const float artifact = valuesChunk[1][rr*COLUMN_GROUP + lane];
                const float detailLost = valuesChunk[2][rr*COLUMN_GROUP + lane];
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
        GroupMemoryBarrierWithGroupSync();
    }

    // sum of the columns
    if (s == 0){
        [ForceUnroll]
        for (int i = 0; i < 6; i++) columnSums[i][lane] = total[i];
    }
    GroupMemoryBarrierWithGroupSync();
    for (int next = COLUMN_GROUP/2; next > 0; next /= 2){
        if (s == 0 && lane < next){
            [ForceUnroll]
            for (int i = 0; i < 6; i++) columnSums[i][lane] += columnSums[i][lane + next];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (s == 0 && lane == 0){
        [ForceUnroll]
        for (int i = 0; i < 6; i++) sums[6*blockIdx.x + i] = columnSums[i][0];
    }
}
