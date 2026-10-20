// Butteraugli's Gaussian blurs with libjxl's arithmetic (lib/jxl/butteraugli/butteraugli.cc, the reference the scores
// are compared with), one kernel per radius: a workgroup blurs a TILE_X x TILE_Y tile, first along the rows of the
// tile and of its 2*R halo rows into shared memory, then along the columns, so the image between the two passes
// never goes through memory. Compiled once per radius (-DRADIUS=2, 3, 6, 7 or 16, libjxl's ComputeKernel radii for
// the sigmas 1.2, 1.564, 2.7, 3.22 and 7.156), so that the loops over the taps unroll and the weights stay in
// registers.
//
// - Radius 2 is libjxl's Separable5: the borders are mirrored (Mirror: -1 -> 0, -2 -> 1, size -> size - 1), the row
//   pass is fma(l2 + r2, w2, fma(l1 + r1, w1, c*w0)) and the column pass combines 5 row results the same way.
//   libjxl computes the last width % 8 columns of an image differently (a scalar tail summing all 25 taps); this
//   kernel gives them the vector formula, which is the same for widths that are a multiple of 8.
// - The other radii are libjxl's ConvolutionWithTranspose, applied to the rows then to the columns. Inside the image
//   the taps are added in pairs around the centre, each pair times its weight scaled by 1/(sum of the weights), the
//   pairs spread over 4 partial sums in turn; near a border (ConvolveBorderColumn) only the taps inside the image
//   are summed, with their unscaled weights, then the sum is scaled by 1/(sum of those weights).
// - Every operation is rounded as libjxl's compiled tools round it: a multiplication and addition that libjxl writes
//   in one C++ expression (sum += a*b, x*k + s) is fused, written here as fma(); every other operation is rounded on
//   its own (the build marks them all NoContraction, spvFileToCppHeader.cpp). This is what matches libjxl 0.12.0's
//   butteraugli_main to its printed digits.
//
// The kernel buffer holds the 2*R + 1 weights, then the R + 1 scaled weights (GaussianHandle in gaussianblur.hpp).
//
// For radius 2, opsinDynamicsBlur blurs the 3 planes of an image and computes the opsin dynamics from them
// (opsinModule.slang); it can also write the image's half scale from the input tiles in shared memory.
//
// postOp makes separableBlur also do the pass that followed the blur, writing dst and second (never src: other
// workgroups still read its tile borders), with b the blurred value:
// - POST_RESIDUAL: second = src - b;
// - POST_REMOVE_RANGES: dst = RemoveRangeAroundZero(b, 1.5) and second = RemoveRangeAroundZero(src - b, 0.04);
// - POST_SEPARATE_HF_UHF: dst and second are separateHf_Uhf of b and src;
// - POST_SUBTRACT_REMOVE_RANGE: second = src - b and dst = RemoveRangeAroundZero(b, w);
// - POST_SUBTRACT_AMPLIFY_RANGE: second = src - b and dst = AmplifyRangeAroundZero(b, w).

import opsinModule;
import frequencyOpsModule;

#define CONCAT2(a, b) a##b
#define CONCAT(a, b) CONCAT2(a, b)

static const int R = RADIUS;
static const int TAPS = 2*R + 1;
static const bool SEPARABLE5 = (R == 2);
static const int FIRST_PAIRS = (R < 4) ? R : 4; // the pairs that start libjxl's 4 partial sums
// 32x64 rather than 64x32: fewer halo rows per output for the large radii
static const int TILE_X = 32;
static const int TILE_Y = 64;
static const int THREADS = 256;

// the passes separableBlur can do after the blur (postOp), as BlurPostOp in gaussianblur.hpp
static const int POST_NONE = 0;
static const int POST_RESIDUAL = 1;
static const int POST_REMOVE_RANGES = 2;
static const int POST_SEPARATE_HF_UHF = 3;
static const int POST_SUBTRACT_REMOVE_RANGE = 4;
static const int POST_SUBTRACT_AMPLIFY_RANGE = 5;

static const int ROWS = TILE_Y + 2*R;          // rows of the row pass: the tile and its halo
static const int IN_PITCH = TILE_X + 2*R + 1;  // odd, so that the lanes reading a column each do not conflict
static const int ROW_PITCH = TILE_X + 1;
static const int TILE_LOADS = (ROWS*(TILE_X + 2*R) + THREADS - 1)/THREADS; // input loads per thread
// The row pass: each task computes ROW_RUN consecutive outputs of one row.
static const int ROW_RUN = 8;
static const int ROW_TASKS = ROWS*(TILE_X/ROW_RUN);
static const int TASKS_PER_THREAD = (ROW_TASKS + THREADS - 1)/THREADS;
// The column pass: each thread computes COLUMN_RUN consecutive outputs of one column.
static const int COLUMN_RUN = TILE_Y*TILE_X/THREADS;

// The input tile, then (in the same memory, once it has been read) the results of the row pass, whose pitch is
// smaller.
groupshared float tile[ROWS*IN_PITCH];

// libjxl's Mirror: the position x of a line of the given size, mirrored into it (-1 -> 0, size -> size - 1)
int mirror(int x, int size){
    while (x < 0 || x >= size){
        x = (x < 0) ? -x - 1 : 2*size - 1 - x;
    }
    return x;
}

// The blurred value at position pos of a line of length size: in[j] is position pos - R + j (0 outside the line).
// k holds the weights, scaled the scaled weights.
float blurPosition(float in[TAPS], int pos, int size, float k[TAPS], float scaled[R + 1]){
    if (SEPARABLE5){
        // HorzConvolve (in is mirrored at the borders)
        return fma(in[0] + in[4], scaled[0], fma(in[1] + in[3], scaled[1], in[2]*scaled[2]));
    }
    if (pos >= R && pos < size - R){
        // ConvolutionWithTranspose's loop for kernel.size() == TAPS
        float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        [ForceUnroll]
        for (int j = 0; j < FIRST_PAIRS; j++) partial[j] = (in[j] + in[TAPS - 1 - j])*scaled[j];
        [ForceUnroll]
        for (int j = 4; j < R; j++) partial[j % 4] = fma(in[j] + in[TAPS - 1 - j], scaled[j], partial[j % 4]);
        if (R == 3){
            // case 7: in[3]*sk3 + sum0 + sum1 + sum2, one expression
            return (fma(in[3], scaled[3], partial[0]) + partial[1]) + partial[2];
        }
        // the other cases: sum = centre*weight, then sum + sum0 + sum1 + sum2 + sum3
        const float centre = in[R]*scaled[R];
        return (((centre + partial[0]) + partial[1]) + partial[2]) + partial[3];
    }
    // ConvolveBorderColumn: the taps inside the line, weights and values in increasing position
    float weight = 0.0f;
    float sum = 0.0f;
    [ForceUnroll]
    for (int j = 0; j < TAPS; j++){
        const int p = pos - R + j;
        if (p >= 0 && p < size){
            weight += k[j];
            sum = fma(in[j], k[j], sum);
        }
    }
    return sum*(1.0f/weight);
}

// Element i of the input tile is row y0 - R + i / (TILE_X + 2*R) and column x0 - R + i % (TILE_X + 2*R) of the image:
// the tile at (x0, y0) and its halo. Thread tid holds the elements tid + m*THREADS, m < TILE_LOADS.
int tileRow(int i){
    return i/(TILE_X + 2*R);
}

int tileColumn(int i){
    return i - tileRow(i)*(TILE_X + 2*R);
}

// Thread tid's elements of the input tile of src: outside the image, mirrored pixels for radius 2 and 0 for the
// others (whose border formula does not read them). The loads are all issued before the first store (storeTile), so
// that their waits overlap.
void loadTile(float* src, int64_t width, int64_t height, int x0, int y0, int tid, out float loaded[TILE_LOADS]){
    [ForceUnroll]
    for (int m = 0; m < TILE_LOADS; m++){
        const int i = tid + m*THREADS;
        int y = y0 - R + tileRow(i);
        int x = x0 - R + tileColumn(i);
        if (SEPARABLE5){
            y = mirror(y, (int)height);
            x = mirror(x, (int)width);
        }
        loaded[m] = (i < ROWS*(TILE_X + 2*R) && x >= 0 && x < (int)width && y >= 0 && y < (int)height) ? src[(int64_t)y*width + x] : 0.0f;
    }
}

// Puts thread tid's elements of the input tile into tile, for blurStoredTile. afterAnotherTile: a former call's
// column pass may still read tile.
void storeTile(int tid, float loaded[TILE_LOADS], bool afterAnotherTile){
    if (afterAnotherTile) GroupMemoryBarrierWithGroupSync();
    [ForceUnroll]
    for (int m = 0; m < TILE_LOADS; m++){
        const int i = tid + m*THREADS;
        if (i < ROWS*(TILE_X + 2*R)) tile[tileRow(i)*IN_PITCH + tileColumn(i)] = loaded[m];
    }
    GroupMemoryBarrierWithGroupSync();
}

// The input tile's pixel (x, y) of the image, in the tile at (x0, y0) or its halo, after storeTile.
float storedTilePixel(int x0, int y0, int x, int y){
    return tile[(y - y0 + R)*IN_PITCH + x - x0 + R];
}

// The blur of the input tile in tile (storeTile): thread tid's COLUMN_RUN outputs, of column x0 + tid % TILE_X and
// rows y0 + COLUMN_RUN*(tid / TILE_X) and the COLUMN_RUN-1 after (0 outside the image). Every thread of the workgroup
// calls it.
void blurStoredTile(int64_t width, int64_t height, int x0, int y0, int tid, float k[TAPS], float scaled[R + 1], out float blurred[COLUMN_RUN]){
    const int w = (int)width;
    const int h = (int)height;

    // row pass, kept in registers until every thread has read the input tile
    float rowOut[TASKS_PER_THREAD][ROW_RUN];
    [ForceUnroll]
    for (int m = 0; m < TASKS_PER_THREAD; m++){
        const int task = tid + m*THREADS;
        if (task >= ROW_TASKS) break;
        const int r = task/(TILE_X/ROW_RUN);
        const int c0 = (task - r*(TILE_X/ROW_RUN))*ROW_RUN;
        float line[ROW_RUN + 2*R];
        [ForceUnroll]
        for (int i = 0; i < ROW_RUN + 2*R; i++) line[i] = tile[r*IN_PITCH + c0 + i];
        [ForceUnroll]
        for (int n = 0; n < ROW_RUN; n++){
            const int x = x0 + c0 + n;
            float in[TAPS];
            [ForceUnroll]
            for (int j = 0; j < TAPS; j++) in[j] = line[n + j];
            // a column outside the image only feeds outputs outside the image
            rowOut[m][n] = (x < w) ? blurPosition(in, x, w, k, scaled) : 0.0f;
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

    // column pass: for radius 2 the halo rows hold the row results of the mirrored rows, as libjxl's row pointers do
    const int c = tid % TILE_X;
    const int rr0 = (tid/TILE_X)*COLUMN_RUN;
    float column[COLUMN_RUN + 2*R];
    [ForceUnroll]
    for (int i = 0; i < COLUMN_RUN + 2*R; i++) column[i] = tile[(rr0 + i)*ROW_PITCH + c];
    [ForceUnroll]
    for (int n = 0; n < COLUMN_RUN; n++){
        const int y = y0 + rr0 + n;
        float in[TAPS];
        [ForceUnroll]
        for (int j = 0; j < TAPS; j++) in[j] = column[n + j];
        blurred[n] = (x0 + c < w && y < h) ? blurPosition(in, y, h, k, scaled) : 0.0f;
    }
}

// The weights and scaled weights of the kernel buffer (GaussianHandle)
void loadWeights(float* kernel, out float k[TAPS], out float scaled[R + 1]){
    [ForceUnroll]
    for (int j = 0; j < TAPS; j++) k[j] = kernel[j];
    [ForceUnroll]
    for (int j = 0; j <= R; j++) scaled[j] = kernel[TAPS + j];
}

// The blur of the tile at (x0, y0) of src (blurStoredTile's outputs). Every thread of the workgroup calls it, and may
// call it again for another plane (afterAnotherTile).
void blurTile(float* src, int64_t width, int64_t height, int x0, int y0, int tid, float k[TAPS], float scaled[R + 1], bool afterAnotherTile, out float blurred[COLUMN_RUN]){
    float loaded[TILE_LOADS];
    loadTile(src, width, height, x0, y0, tid, loaded);
    storeTile(tid, loaded, afterAnotherTile);
    blurStoredTile(width, height, x0, y0, tid, k, scaled, blurred);
}

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void CONCAT(separableBlur, RADIUS)(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* src, uniform float* dst, uniform int64_t width, uniform int64_t height, uniform float* gaussiankernel, uniform float* second, uniform int postOp, uniform float w){
    const int tid = (int)threadIdx.x;
    const int x0 = (int)blockIdx.x*TILE_X;
    const int y0 = (int)blockIdx.y*TILE_Y;
    float k[TAPS], scaled[R + 1];
    loadWeights(gaussiankernel, k, scaled);

    float blurred[COLUMN_RUN];
    blurTile(src, width, height, x0, y0, tid, k, scaled, false, blurred);

    const int x = x0 + tid % TILE_X;
    if (x >= (int)width) return;
    [ForceUnroll]
    for (int n = 0; n < COLUMN_RUN; n++){
        const int y = y0 + (tid/TILE_X)*COLUMN_RUN + n;
        if (y >= (int)height) return;
        const int64_t pixel = (int64_t)y*width + x;
        const float b = blurred[n];
        if (postOp == POST_NONE){
            dst[pixel] = b;
        } else if (postOp == POST_RESIDUAL){
            dst[pixel] = b;
            second[pixel] = src[pixel] - b;
        } else if (postOp == POST_REMOVE_RANGES){
            dst[pixel] = RemoveRangeAroundZero_Device(b, 1.5f);
            second[pixel] = RemoveRangeAroundZero_Device(src[pixel] - b, 0.04f);
        } else if (postOp == POST_SEPARATE_HF_UHF){
            float hf = b;
            float uhf = src[pixel];
            separateHfUhf(hf, uhf);
            dst[pixel] = hf;
            second[pixel] = uhf;
        } else {
            second[pixel] = src[pixel] - b;
            dst[pixel] = (postOp == POST_SUBTRACT_REMOVE_RANGE) ? RemoveRangeAroundZero_Device(b, w) : AmplifyRangeAroundZero_Device(b, w);
        }
    }
}

#if RADIUS == 2
// Butteraugli's downsample of the tile at (x0, y0) into halfPlane, from the input tile in shared memory (storeTile): the
// mean of each 2x2 block, the last row and column repeated, added up in downsample's order. A tile's blocks stay
// inside it, as x0 and y0 are even. TILE_X*TILE_Y/4 is a multiple of THREADS.
void storeHalfScale(float* halfPlane, int64_t width, int64_t height, int x0, int y0, int tid){
    const int64_t halfWidth = (width - 1)/2 + 1;
    const int64_t halfHeight = (height - 1)/2 + 1;
    [ForceUnroll]
    for (int q = 0; q < TILE_X*TILE_Y/4/THREADS; q++){
        const int j = tid + q*THREADS;
        const int64_t hx = x0/2 + j % (TILE_X/2);
        const int64_t hy = y0/2 + j / (TILE_X/2);
        if (hx >= halfWidth || hy >= halfHeight) continue;
        const int left = (int)min(2*hx, width - 1);
        const int right = (int)min(2*hx + 1, width - 1);
        const int top = (int)min(2*hy, height - 1);
        const int bottom = (int)min(2*hy + 1, height - 1);
        float sum = 0.0f;
        sum += storedTilePixel(x0, y0, left, top);
        sum += storedTilePixel(x0, y0, left, bottom);
        sum += storedTilePixel(x0, y0, right, top);
        sum += storedTilePixel(x0, y0, right, bottom);
        sum *= 0.25f;
        halfPlane[hy*halfWidth + hx] = sum;
    }
}

// blurTile of one plane of the opsin dynamics' image, which also writes the plane's half scale into halfPlane when
// writeHalf is set (storeHalfScale).
void blurTileAndHalve(float* src, float* halfPlane, bool writeHalf, int64_t width, int64_t height, int x0, int y0, int tid, float k[TAPS], float scaled[R + 1], bool afterAnotherTile, out float blurred[COLUMN_RUN]){
    float loaded[TILE_LOADS];
    loadTile(src, width, height, x0, y0, tid, loaded);
    storeTile(tid, loaded, afterAnotherTile);
    if (writeHalf) storeHalfScale(halfPlane, width, height, x0, y0, tid);
    blurStoredTile(width, height, x0, y0, tid, k, scaled, blurred);
}

// Butteraugli's opsin dynamics of an image (src1, src2, src3: linear RGB) into dst1, dst2, dst3: the 3 planes are
// blurred (radius 2) and combined with the image in one pass. dst must not be src, as other workgroups still read
// src's tile borders. When writeHalfScale is set, the kernel also writes the image's half scale (Butteraugli's
// downsample) into halfScale, its 3 planes one after the other.
[shader("compute")]
[numthreads(THREADS, 1, 1)]
void opsinDynamicsBlur(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* src1, uniform float* src2, uniform float* src3, uniform float* dst1, uniform float* dst2, uniform float* dst3, uniform int64_t width, uniform int64_t height, uniform float* gaussiankernel, uniform float intensity_multiplier, uniform float* halfScale, uniform int writeHalfScale){
    const int tid = (int)threadIdx.x;
    const int x0 = (int)blockIdx.x*TILE_X;
    const int y0 = (int)blockIdx.y*TILE_Y;
    float k[TAPS], scaled[R + 1];
    loadWeights(gaussiankernel, k, scaled);

    const bool writeHalf = writeHalfScale != 0;
    const int64_t halfSize = ((width - 1)/2 + 1)*((height - 1)/2 + 1);
    float blurred1[COLUMN_RUN], blurred2[COLUMN_RUN], blurred3[COLUMN_RUN];
    blurTileAndHalve(src1, halfScale, writeHalf, width, height, x0, y0, tid, k, scaled, false, blurred1);
    blurTileAndHalve(src2, halfScale + halfSize, writeHalf, width, height, x0, y0, tid, k, scaled, true, blurred2);
    blurTileAndHalve(src3, halfScale + 2*halfSize, writeHalf, width, height, x0, y0, tid, k, scaled, true, blurred3);

    const int x = x0 + tid % TILE_X;
    if (x >= (int)width) return;
    [ForceUnroll]
    for (int n = 0; n < COLUMN_RUN; n++){
        const int y = y0 + (tid/TILE_X)*COLUMN_RUN + n;
        if (y >= (int)height) return;
        const int64_t pixel = (int64_t)y*width + x;
        const float3 xyb = opsinDynamics(float3(src1[pixel], src2[pixel], src3[pixel]), float3(blurred1[n], blurred2[n], blurred3[n]), intensity_multiplier);
        dst1[pixel] = xyb.x;
        dst2[pixel] = xyb.y;
        dst3[pixel] = xyb.z;
    }
}

#endif
