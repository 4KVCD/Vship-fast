// Butteraugli's Gaussian blur for the windows 3, 4 and 18 in one kernel: a workgroup blurs a TILE_X x TILE_Y tile,
// first along the rows of the tile and of its 2*WINDOW halo rows into shared memory, then along the columns, so the
// image between the two passes never goes through memory. Compiled once per window size (-DWINDOW=3, 4 or 18) so
// that the loops over the 2*WINDOW+1 taps unroll and the kernel weights stay in scalar registers.
//
// Every output is the same float as with the former separate passes (horizontalBlur then verticalBlur): the same
// fused multiply-adds in the same order, then the division by the same weight.
// - The pixels outside the image are zero and every tap is summed. The former passes skipped the taps outside the
//   image instead, which gives the same float: the sum starts at +0, and fma(+0, k, s) is s for any s, +0 included.
// - The row pass of a row outside the image gives +0, as the skipped taps of the column pass did.
// - Each row result is a float in shared memory, as it was in the temporary plane between the two passes.
// - The taps are explicit fma() calls. AMD's compiler reorders an unrolled chain of "out += a*b" (it moved the
//   first tap to the end of some sums), which a loop with a run-time bound, like the former passes', prevents.
// - Each sum starts from the argument zero, which is 0: a literal 0 lets the compiler turn the first fma into a
//   multiplication, whose result differs from fma(a, k, +0) when a*k is -0.
//
// For window 3, opsinDynamicsBlur blurs the 3 planes of an image the same way and computes the opsin dynamics from
// them (opsinModule.slang), each blurred value through opaque() as it came from memory before.
//
// postOp makes separableBlur also do the pass that followed the blur, writing dst and second (never src: other
// workgroups still read its tile borders). The blurred value b goes through opaque() as it went through memory:
// - POST_RESIDUAL: second = src - b (the subArray pass after the window-18 blur);
// - POST_REMOVE_RANGES (window 4, X channel): dst = RemoveRangeAroundZero(b, 1.5) and
//   second = RemoveRangeAroundZero(src - b, 0.04) (subarray_removerangearound0 then removerangearound0);
// - POST_SEPARATE_HF_UHF (window 4, Y channel): dst and second are separateHf_Uhf of b and src.

import opsinModule;
import frequencyOpsModule;

#define CONCAT2(a, b) a##b
#define CONCAT(a, b) CONCAT2(a, b)

static const int G = WINDOW;
static const int TAPS = 2*G + 1;
// 32x64 rather than 64x32: window 18 is 10% faster on a Radeon 8060S (fewer halo rows per output), the others the same
static const int TILE_X = 32;
static const int TILE_Y = 64;
static const int THREADS = 256;

// the passes separableBlur can do after the blur (postOp)
static const int POST_NONE = 0;
static const int POST_RESIDUAL = 1;
static const int POST_REMOVE_RANGES = 2;
static const int POST_SEPARATE_HF_UHF = 3;

static const int ROWS = TILE_Y + 2*G;          // rows of the row pass: the tile and its halo
static const int IN_PITCH = TILE_X + 2*G + 1;  // odd, so that the lanes reading a column each do not conflict
static const int ROW_PITCH = TILE_X + 1;
static const int TILE_LOADS = (ROWS*(TILE_X + 2*G) + THREADS - 1)/THREADS; // input loads per thread
// The row pass: each task computes ROW_RUN consecutive outputs of one row.
static const int ROW_RUN = 8;
static const int ROW_TASKS = ROWS*(TILE_X/ROW_RUN);
static const int TASKS_PER_THREAD = (ROW_TASKS + THREADS - 1)/THREADS;
// The column pass: each thread computes COLUMN_RUN consecutive outputs of one column.
static const int COLUMN_RUN = TILE_Y*TILE_X/THREADS;

// The input tile, then (in the same memory, once it has been read) the results of the row pass, whose pitch is
// smaller.
groupshared float tile[ROWS*IN_PITCH];

// x unchanged, through an operation the compiler cannot see through (zero is 0)
float opaque(float x, uint zero){
    return asfloat(asuint(x) | zero);
}

// The weight of the window of pixel n of a line of the given length: the sum of the kernel over the taps inside the
// line, as the integral gives it.
float windowWeight(float* integral, int n, int length){
    // a pixel past the end only feeds outputs outside the image; clamped so that its weight reads inside the integral
    n = min(n, length - 1);
    const int begin = max(n - G, 0);
    const int end = min(n + G, length - 1);
    return integral[G+end+1-n] - integral[G+begin-n];
}

// The blur of the tile at (x0, y0) of src: thread tid's COLUMN_RUN outputs, of column x0 + tid % TILE_X and rows
// y0 + COLUMN_RUN*(tid / TILE_X) and the COLUMN_RUN-1 after (0 outside the image). Every thread of the workgroup calls
// it, and may call it again for another plane.
void blurTile(float* src, int64_t width, int64_t height, int x0, int y0, int tid, float k[TAPS], float* gaussiankernel_integral, float zero, bool afterAnotherTile, out float blurred[COLUMN_RUN]){
    const int w = (int)width;
    const int h = (int)height;

    // the input tile: rows y0-G .. y0+TILE_Y+G-1 and columns x0-G .. x0+TILE_X+G-1, 0 outside the image. The loads
    // are all issued before the first store, so that their waits overlap.
    float loaded[TILE_LOADS];
    [ForceUnroll]
    for (int m = 0; m < TILE_LOADS; m++){
        const int i = tid + m*THREADS;
        const int r = i/(TILE_X + 2*G);
        const int c = i - r*(TILE_X + 2*G);
        const int y = y0 - G + r;
        const int x = x0 - G + c;
        loaded[m] = (i < ROWS*(TILE_X + 2*G) && x >= 0 && x < w && y >= 0 && y < h) ? src[(int64_t)y*width + x] : 0.0f;
    }
    // a former call's column pass may still read the tile
    if (afterAnotherTile) GroupMemoryBarrierWithGroupSync();
    [ForceUnroll]
    for (int m = 0; m < TILE_LOADS; m++){
        const int i = tid + m*THREADS;
        const int r = i/(TILE_X + 2*G);
        if (i < ROWS*(TILE_X + 2*G)) tile[r*IN_PITCH + i - r*(TILE_X + 2*G)] = loaded[m];
    }
    GroupMemoryBarrierWithGroupSync();

    // row pass, kept in registers until every thread has read the input tile
    float rowOut[TASKS_PER_THREAD][ROW_RUN];
    [ForceUnroll]
    for (int m = 0; m < TASKS_PER_THREAD; m++){
        const int task = tid + m*THREADS;
        if (task >= ROW_TASKS) break;
        const int r = task/(TILE_X/ROW_RUN);
        const int c0 = (task - r*(TILE_X/ROW_RUN))*ROW_RUN;
        float in[ROW_RUN + 2*G];
        [ForceUnroll]
        for (int i = 0; i < ROW_RUN + 2*G; i++) in[i] = tile[r*IN_PITCH + c0 + i];
        [ForceUnroll]
        for (int n = 0; n < ROW_RUN; n++){
            const int x = x0 + c0 + n;
            float out = zero;
            [ForceUnroll]
            for (int j = 0; j < TAPS; j++) out = fma(in[n+j], k[j], out);
            // a column outside the image only feeds outputs outside the image
            rowOut[m][n] = (x < w) ? out/windowWeight(gaussiankernel_integral, x, w) : 0.0f;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    [ForceUnroll]
    for (int m = 0; m < TASKS_PER_THREAD; m++){
        const int task = tid + m*THREADS;
        if (task >= ROW_TASKS) break;
        const int r = task/(TILE_X/ROW_RUN);
        const int c0 = (task - r*(TILE_X/ROW_RUN))*ROW_RUN;
        [ForceUnroll]
        for (int n = 0; n < ROW_RUN; n++) tile[r*ROW_PITCH + c0 + n] = rowOut[m][n];
    }
    GroupMemoryBarrierWithGroupSync();

    // column pass
    const int c = tid % TILE_X;
    const int rr0 = (tid/TILE_X)*COLUMN_RUN;
    float column[COLUMN_RUN + 2*G];
    [ForceUnroll]
    for (int i = 0; i < COLUMN_RUN + 2*G; i++) column[i] = tile[(rr0 + i)*ROW_PITCH + c];
    [ForceUnroll]
    for (int n = 0; n < COLUMN_RUN; n++){
        const int y = y0 + rr0 + n;
        float out = zero;
        [ForceUnroll]
        for (int j = 0; j < TAPS; j++) out = fma(column[n+j], k[j], out);
        blurred[n] = (x0 + c < w && y < h) ? out/windowWeight(gaussiankernel_integral, y, h) : 0.0f;
    }
}

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void CONCAT(separableBlur, WINDOW)(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* src, uniform float* dst, uniform int64_t width, uniform int64_t height, uniform float* gaussiankernel, uniform float* gaussiankernel_integral, uniform float zero, uniform float* second, uniform int postOp){
    const int tid = (int)threadIdx.x;
    const int x0 = (int)blockIdx.x*TILE_X;
    const int y0 = (int)blockIdx.y*TILE_Y;
    float k[TAPS];
    [ForceUnroll]
    for (int j = 0; j < TAPS; j++) k[j] = gaussiankernel[j];

    float blurred[COLUMN_RUN];
    blurTile(src, width, height, x0, y0, tid, k, gaussiankernel_integral, zero, false, blurred);

    const int x = x0 + tid % TILE_X;
    if (x >= (int)width) return;
    [ForceUnroll]
    for (int n = 0; n < COLUMN_RUN; n++){
        const int y = y0 + (tid/TILE_X)*COLUMN_RUN + n;
        if (y >= (int)height) return;
        const int64_t pixel = (int64_t)y*width + x;
        if (postOp == POST_NONE){
            dst[pixel] = blurred[n];
            continue;
        }
        const float b = opaque(blurred[n], asuint(zero));
        if (postOp == POST_RESIDUAL){
            dst[pixel] = blurred[n];
            second[pixel] = src[pixel] - b;
        } else if (postOp == POST_REMOVE_RANGES){
            dst[pixel] = RemoveRangeAroundZero_Device(b, 1.5f);
            second[pixel] = RemoveRangeAroundZero_Device(opaque(src[pixel] - b, asuint(zero)), 0.04f);
        } else {
            float hf = b;
            float uhf = src[pixel];
            separateHfUhf(hf, uhf);
            dst[pixel] = hf;
            second[pixel] = uhf;
        }
    }
}

#if WINDOW == 3
// Butteraugli's opsin dynamics of an image (src1, src2, src3: linear RGB) into dst1, dst2, dst3: the 3 planes are
// blurred (window 3) and combined with the image in one pass. dst must not be src, as other workgroups still read
// src's tile borders.
[shader("compute")]
[numthreads(THREADS, 1, 1)]
void opsinDynamicsBlur(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* src1, uniform float* src2, uniform float* src3, uniform float* dst1, uniform float* dst2, uniform float* dst3, uniform int64_t width, uniform int64_t height, uniform float* gaussiankernel, uniform float* gaussiankernel_integral, uniform float zero, uniform float intensity_multiplier){
    const int tid = (int)threadIdx.x;
    const int x0 = (int)blockIdx.x*TILE_X;
    const int y0 = (int)blockIdx.y*TILE_Y;
    float k[TAPS];
    [ForceUnroll]
    for (int j = 0; j < TAPS; j++) k[j] = gaussiankernel[j];

    float blurred1[COLUMN_RUN], blurred2[COLUMN_RUN], blurred3[COLUMN_RUN];
    blurTile(src1, width, height, x0, y0, tid, k, gaussiankernel_integral, zero, false, blurred1);
    blurTile(src2, width, height, x0, y0, tid, k, gaussiankernel_integral, zero, true, blurred2);
    blurTile(src3, width, height, x0, y0, tid, k, gaussiankernel_integral, zero, true, blurred3);

    const int x = x0 + tid % TILE_X;
    if (x >= (int)width) return;
    const uint z = asuint(zero);
    [ForceUnroll]
    for (int n = 0; n < COLUMN_RUN; n++){
        const int y = y0 + (tid/TILE_X)*COLUMN_RUN + n;
        if (y >= (int)height) return;
        const int64_t pixel = (int64_t)y*width + x;
        const float3 xyb = opsinDynamics(float3(src1[pixel], src2[pixel], src3[pixel]), float3(opaque(blurred1[n], z), opaque(blurred2[n], z), opaque(blurred3[n], z)), intensity_multiplier);
        dst1[pixel] = xyb.x;
        dst2[pixel] = xyb.y;
        dst3[pixel] = xyb.z;
    }
}
#endif
