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

#define CONCAT2(a, b) a##b
#define CONCAT(a, b) CONCAT2(a, b)

static const int G = WINDOW;
static const int TAPS = 2*G + 1;
static const int TILE_X = 64;
static const int TILE_Y = 32;
static const int THREADS = 256;

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

// The weight of the window of pixel n of a line of the given length: the sum of the kernel over the taps inside the
// line, as the integral gives it.
float windowWeight(float* integral, int n, int length){
    const int begin = max(n - G, 0);
    const int end = min(n + G, length - 1);
    return integral[G+end+1-n] - integral[G+begin-n];
}

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void CONCAT(separableBlur, WINDOW)(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* src, uniform float* dst, uniform int64_t width, uniform int64_t height, uniform float* gaussiankernel, uniform float* gaussiankernel_integral, uniform float zero){
    const int w = (int)width;
    const int h = (int)height;
    const int tid = (int)threadIdx.x;
    const int x0 = (int)blockIdx.x*TILE_X;
    const int y0 = (int)blockIdx.y*TILE_Y;

    float k[TAPS];
    [ForceUnroll]
    for (int j = 0; j < TAPS; j++) k[j] = gaussiankernel[j];

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

    // column pass: thread tid takes column tid % TILE_X, rows COLUMN_RUN*(tid / TILE_X) and the COLUMN_RUN-1 after
    const int c = tid % TILE_X;
    const int rr0 = (tid/TILE_X)*COLUMN_RUN;
    const int x = x0 + c;
    if (x >= w) return;
    float column[COLUMN_RUN + 2*G];
    [ForceUnroll]
    for (int i = 0; i < COLUMN_RUN + 2*G; i++) column[i] = tile[(rr0 + i)*ROW_PITCH + c];
    [ForceUnroll]
    for (int n = 0; n < COLUMN_RUN; n++){
        const int y = y0 + rr0 + n;
        if (y >= h) return;
        float out = zero;
        [ForceUnroll]
        for (int j = 0; j < TAPS; j++) out = fma(column[n+j], k[j], out);
        dst[(int64_t)y*width + x] = out/windowWeight(gaussiankernel_integral, y, h);
    }
}
