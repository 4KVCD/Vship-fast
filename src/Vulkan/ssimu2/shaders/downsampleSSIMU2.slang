import makeXYBSSIMU2;
import "../../gpuColorToLinear/shaders/vshipColorModule";
import "../../gpuColorToLinear/shaders/primaries";
import "../../gpuColorToLinear/shaders/inverseTransferModule";
import "../../gpuColorToLinear/shaders/YUVToLinRGB";

// Halves linear RGB like libjxl's Downsample (each output the mean of a 2x2 block, the last row and column
// repeated) and converts the source scale to positive XYB in place, the smaller scale too when lastScale is set.
[shader("compute")]
[numthreads(16, 16, 1)]
void downsample(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* src1, uniform float* src2, uniform float* src3, uniform float* dst1, uniform float* dst2, uniform float* dst3, uniform uint64_t width, uniform uint64_t height, uniform uint lastScale, uniform uint zero){
    uint64_t x = threadIdx.x + blockIdx.x*16; // < width >> 1 +1
    uint64_t y = threadIdx.y + blockIdx.y*16; // < height >> 1 +1

    int64_t newh = (height-1)/2 + 1;
    int64_t neww = (width-1)/2 + 1;

    if (x >= neww || y >= newh) return;

    float3 srcData[4];
    for (int dy = 0; dy < 2; dy++){
        for (int dx = 0; dx < 2; dx++){
            srcData[dy*2+dx].x = src1[min((int)(2*y + dy), (int)(height-1)) * width + min((int)(2*x+dx), (int)(width-1))];
            srcData[dy*2+dx].y = src2[min((int)(2*y + dy), (int)(height-1)) * width + min((int)(2*x+dx), (int)(width-1))];
            srcData[dy*2+dx].z = src3[min((int)(2*y + dy), (int)(height-1)) * width + min((int)(2*x+dx), (int)(width-1))];
        }
    }

    float3 dst = {0.f, 0.f, 0.f};
    for (int i = 0; i < 4; i++){
        dst += srcData[i];
        //to later write srcData in XYB
        srcData[i] = rgb_to_positive_xyb_d(srcData[i], zero);
    }
    dst *= 0.25f;

    //the smaller scale is converted by the next downsample, except for the last one
    if (lastScale != 0){
        dst = rgb_to_positive_xyb_d(dst, zero);
    }

    dst1[y * neww + x] = dst.x;
    dst2[y * neww + x] = dst.y;
    dst3[y * neww + x] = dst.z;

    //write on source the XYB version directly with in registers data to reduce memory reads
    for (int dy = 0; dy < 2; dy++){
        for (int dx = 0; dx < 2; dx++){
            if (2*y+dy < height && 2*x+dx < width){
                src1[(2*y+dy)*width+2*x+dx] = srcData[dy*2+dx].x;
                src2[(2*y+dy)*width+2*x+dx] = srcData[dy*2+dx].y;
                src3[(2*y+dy)*width+2*x+dx] = srcData[dy*2+dx].z;
            }
        }
    }
}

float3 unfused3(float3 v, uint zero){
    return float3(unfused(v.x, zero), unfused(v.y, zero), unfused(v.z, zero));
}

// The pixel (x, y) of the image in linear RGB, as YUVToLinRGBUpsampled computes it from the luma (lumaAt) and the
// horizontally upsampled chroma (Converter::placeChromaForPixelConversion), then converted to the destination
// primaries when toPrimaries is set, as primariesToPrimaries does. Each step's result went through memory, so it
// goes through unfused() and the compiler cannot combine the steps' arithmetic.
float3 linearRGB(float* luma, uint8_t* rawLuma, int rawLumaFormat, float rawLumaScale, float* chroma1, float* chroma2, int64_t x, int64_t y, int64_t width, int64_t height, int verticalMode, Vship_YUVMatrix_t matrix, Vship_TransferFunction_t transfer, int libjxlArithmetic, int toPrimaries, Vship_Primaries_t primariesSource, Vship_Primaries_t primariesDestination, uint zero){
    const int64_t i = y*width + x;
    float3 val = {lumaAt(luma, rawLuma, rawLumaFormat, rawLumaScale, i, width, zero), unfused(upsampledChroma(chroma1, x, y, width, height, verticalMode), zero), unfused(upsampledChroma(chroma2, x, y, width, height, verticalMode), zero)};
    val = inverseMatrixNCL(val, matrix);
    val = inverseTransfer(val, transfer, libjxlArithmetic != 0);
    val = inverseMatrixCL(val, matrix);
    val = unfused3(val, zero);
    if (toPrimaries != 0) val = unfused3(primariesToPrimaries_impl(val, primariesSource, primariesDestination), zero);
    return val;
}

// downsample's mean of a 2x2 block, its pixels in the order downsample adds them (top left, top right, bottom left,
// bottom right)
float3 blockMean(float3 block[4]){
    float3 mean = {0.f, 0.f, 0.f};
    for (int i = 0; i < 4; i++) mean += block[i];
    mean *= 0.25f;
    return mean;
}

// The offset of each scale in a plane holding the 6 scales one after the other, and the scales' sizes.
struct Scales {
    int64_t offset[6];
    int64_t width[6];
    int64_t height[6];
};

Scales scalesOf(int64_t width, int64_t height){
    Scales scales;
    int64_t offset = 0;
    for (int k = 0; k < 6; k++){
        scales.offset[k] = offset;
        scales.width[k] = width;
        scales.height[k] = height;
        offset += width*height;
        width = (width - 1)/2 + 1;
        height = (height - 1)/2 + 1;
    }
    return scales;
}

void storeXYB(float* p0, float* p1, float* p2, int64_t index, float3 xyb){
    p0[index] = xyb.x;
    p1[index] = xyb.y;
    p2[index] = xyb.z;
}

// The 6 scales of one image in positive XYB, from its luma and chroma, in one pass: what the converter's
// YUVToLinRGBUpsampled (and primariesToPrimaries), then downsample for the scales 1 to 5, wrote and read through
// memory. A workgroup takes a tile of 32x32 pixels: the 2x2 blocks a scale halves stay inside the tile down to the
// 1x1 of scale 5, the last row and column of each scale being repeated as downsample repeats them. Each thread
// converts PYRAMID_TILE/ROWS pixels of scale 0 (yuvToPyramidBody; a wave a row, so that the loads and stores are
// coalesced) and puts their linear RGB in shared memory, from which each smaller scale is made from the one above
// it. Every value is computed with downsample's arithmetic, and the values that went through memory go through
// unfused(). p0..p2 get the scales one after the other (scalesOf); luma and chroma1/chroma2 may be p0 and p1/p2, as
// each pixel of scale 0 is read and then written by the same thread only.
static const int PYRAMID_TILE = 32; // scale 0 pixels per side of a tile
// the linear RGB of the tile at scale 0 (one array per channel), then of the scale being halved
groupshared float tileRGB[3][PYRAMID_TILE*PYRAMID_TILE];

float3 tilePixel(int index){
    return float3(tileRGB[0][index], tileRGB[1][index], tileRGB[2][index]);
}

void setTilePixel(int index, float3 rgb){
    tileRGB[0][index] = rgb.x;
    tileRGB[1][index] = rgb.y;
    tileRGB[2][index] = rgb.z;
}

// yuvToPyramid's body for workgroups of 32 x ROWS threads, each thread converting the rows ty, ty + ROWS, ... of
// the tile. A 32x32 workgroup would be 1024 threads, of which an NVIDIA SM holds only one, idle behind the barriers
// of the smaller scales: 32x8 is the fastest on an RTX 3070 and 32x16 on a Radeon 8060S (each 2-6% ahead of the
// other), so the host picks the entry point by vendor. The work of a thread differs, its arithmetic does not.
void yuvToPyramidBody<let ROWS : int>(uint3 threadIdx, uint3 blockIdx, float* luma, float* chroma1, float* chroma2, float* p0, float* p1, float* p2, uint64_t width64, uint64_t height64, int verticalMode, Vship_YUVMatrix_t matrix, Vship_TransferFunction_t transfer, int libjxlArithmetic, int toPrimaries, Vship_Primaries_t primariesSource, Vship_Primaries_t primariesDestination, uint zero, uint8_t* rawLuma, int rawLumaFormat, float rawLumaScale){
    const Scales scales = scalesOf((int64_t)width64, (int64_t)height64);
    const int threadIndex = threadIdx.y*PYRAMID_TILE + threadIdx.x;

    // scale 0: this thread's pixels
    [ForceUnroll]
    for (int r = 0; r < PYRAMID_TILE/ROWS; r++){
        const int tx = threadIdx.x;
        const int ty = threadIdx.y + r*ROWS;
        const int64_t x0 = blockIdx.x*PYRAMID_TILE + tx;
        const int64_t y0 = blockIdx.y*PYRAMID_TILE + ty;
        if (x0 < scales.width[0] && y0 < scales.height[0]){
            const float3 rgb = linearRGB(luma, rawLuma, rawLumaFormat, rawLumaScale, chroma1, chroma2, x0, y0, scales.width[0], scales.height[0], verticalMode, matrix, transfer, libjxlArithmetic, toPrimaries, primariesSource, primariesDestination, zero);
            setTilePixel(ty*PYRAMID_TILE + tx, rgb);
            storeXYB(p0, p1, p2, y0*scales.width[0] + x0, rgb_to_positive_xyb_d(rgb, zero));
        }
    }

    // scales 1 to 5, each from the tile of the scale above it in tileRGB: n x n threads (at most 256, all the
    // workgroup's or a part), each reading its 2x2 block (clamped to the scale's last row and column) before any of them
    // writes its mean back. The thread mapping uses shifts and masks: slangc 2026.10.2 crashes on % and / by n here.
    [ForceUnroll]
    for (int k = 1; k < 6; k++){
        const int n = PYRAMID_TILE >> k;
        const int tx = threadIndex & (n - 1); // n = 2^(5-k)
        const int ty = threadIndex >> (5 - k);
        const int64_t xk = blockIdx.x*n + tx;
        const int64_t yk = blockIdx.y*n + ty;
        const bool active = threadIndex < n*n && xk < scales.width[k] && yk < scales.height[k];
        GroupMemoryBarrierWithGroupSync();
        float3 mean = {0.f, 0.f, 0.f};
        if (active){
            float3 block[4];
            for (int dy = 0; dy < 2; dy++){
                for (int dx = 0; dx < 2; dx++){
                    // the tile of scale k-1 starts at 2n*blockIdx in that scale
                    const int64_t x = min(2*xk + dx, scales.width[k-1] - 1) - 2*n*(int64_t)blockIdx.x;
                    const int64_t y = min(2*yk + dy, scales.height[k-1] - 1) - 2*n*(int64_t)blockIdx.y;
                    block[dy*2 + dx] = tilePixel((int)y*PYRAMID_TILE + (int)x);
                }
            }
            mean = blockMean(block);
        }
        GroupMemoryBarrierWithGroupSync();
        if (active){
            // scales 1 to 4 went through memory, to be halved and converted by the next downsample; scale 5 was
            // converted by its own downsample (lastScale), straight from the mean
            if (k < 5){
                mean = unfused3(mean, zero);
                setTilePixel(ty*PYRAMID_TILE + tx, mean);
            }
            storeXYB(p0, p1, p2, scales.offset[k] + yk*scales.width[k] + xk, rgb_to_positive_xyb_d(mean, zero));
        }
    }
}

[shader("compute")]
[numthreads(PYRAMID_TILE, 8, 1)]
void yuvToPyramid8(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* luma, uniform float* chroma1, uniform float* chroma2, uniform float* p0, uniform float* p1, uniform float* p2, uniform uint64_t width64, uniform uint64_t height64, uniform int verticalMode, uniform Vship_YUVMatrix_t matrix, uniform Vship_TransferFunction_t transfer, uniform int libjxlArithmetic, uniform int toPrimaries, uniform Vship_Primaries_t primariesSource, uniform Vship_Primaries_t primariesDestination, uniform uint zero, uniform uint8_t* rawLuma, uniform int rawLumaFormat, uniform float rawLumaScale){
    yuvToPyramidBody<8>(threadIdx, blockIdx, luma, chroma1, chroma2, p0, p1, p2, width64, height64, verticalMode, matrix, transfer, libjxlArithmetic, toPrimaries, primariesSource, primariesDestination, zero, rawLuma, rawLumaFormat, rawLumaScale);
}

[shader("compute")]
[numthreads(PYRAMID_TILE, 16, 1)]
void yuvToPyramid16(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* luma, uniform float* chroma1, uniform float* chroma2, uniform float* p0, uniform float* p1, uniform float* p2, uniform uint64_t width64, uniform uint64_t height64, uniform int verticalMode, uniform Vship_YUVMatrix_t matrix, uniform Vship_TransferFunction_t transfer, uniform int libjxlArithmetic, uniform int toPrimaries, uniform Vship_Primaries_t primariesSource, uniform Vship_Primaries_t primariesDestination, uniform uint zero, uniform uint8_t* rawLuma, uniform int rawLumaFormat, uniform float rawLumaScale){
    yuvToPyramidBody<16>(threadIdx, blockIdx, luma, chroma1, chroma2, p0, p1, p2, width64, height64, verticalMode, matrix, transfer, libjxlArithmetic, toPrimaries, primariesSource, primariesDestination, zero, rawLuma, rawLumaFormat, rawLumaScale);
}
