import CSFModule;

static const float kernel_a = 0.4f;
static const float gaussPyrKernel[5] = {0.25-kernel_a/2.,0.25, kernel_a, 0.25, 0.25-kernel_a/2.};

// A pixel of the plane as pycvvdp's gausspyr_reduce sees it around the image (lpyr_dec.py). The rows are padded
// symmetrically (-1 -> 0, height -> height-1). The columns too on the left, but on the right gausspyr_reduce picks
// its padding from the parity of the height instead of the width; when the two parities differ:
// - odd width: only the second column past the edge gets the last column, the first one is 0;
// - even width: the last output also adds the last column once more with the weight of the tap next to the
//   center, which the first column past the edge (weight K4) carries as K3/K4 times the last column.
float reducePadded(float* plane, int x, int y, int width, int height){
    if (y < 0) y = -y - 1;
    if (y >= height) y = 2*height - 1 - y;
    y = clamp(y, 0, height - 1); //only for planes under 3 pixels
    float* row = plane + (int64_t)y*width;
    if (x < 0) x = -x - 1;
    if (x >= width){
        const bool oddWidth = (width%2) == 1;
        const bool oddHeight = (height%2) == 1;
        if (oddWidth == oddHeight){
            x = 2*width - 1 - x;
        } else if (oddWidth){
            if (x == width) return 0.f;
            x = width - 1;
        } else {
            if (x == width) return row[max(width - 2, 0)] + (gaussPyrKernel[3]/gaussPyrKernel[4])*row[width - 1];
            x = width - 1;
        }
    }
    return row[clamp(x, 0, width - 1)];
}

//pyramidSize is the stride of a "big" plane as a whole. mem_d needs to be offset prior using scaleOffset
//memo: mem_d+scaleOffset+pyramidSize*n = mem_d+pyramidSize*n + scaleOffset (-> we only need mem_d+scaleOffset and not mem_d)
//this kernel is called for all planes and then all scales successively except the smallest one
//both sources are treated at once; a dispatch does channels firstChannel.. (blockIdx.z): channel 0 alone, as it
//writes the Lbkg the others read, then channels 1 to 3 together
[shader("compute")]
[numthreads(16, 16, 1)]
void fromPyrReduceToCSF(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* mem_d, uniform float* mem_d2, uniform float* CSFMem_d, uniform int64_t pyramidSize, uniform int64_t width, uniform int64_t height, uniform int scale, uniform int firstChannel){
    const int channel = firstChannel + (int)blockIdx.z;
    const int64_t base_x = blockIdx.x * 16;
    const int64_t base_y = blockIdx.y * 16;
    const int thx = threadIdx.x;
    const int thy = threadIdx.y;
    const int thid = thy*16+thx;

    const int64_t small_width = (width+1)/2;
    const int64_t small_height = (height+1)/2;

    //memory layout helpers
    float* inputPlane1 = mem_d+channel*pyramidSize; //read from
    float* inputPlane2 = mem_d2+channel*pyramidSize; //read from
    float* Lbkg1 = mem_d+4*pyramidSize; //gets written by channel 0, read by others
    float* Lbkg2 = mem_d2+4*pyramidSize; //gets written by channel 0, read by others
    float* smallScale1 = inputPlane1+width*height; //gets written to
    float* smallScale2 = inputPlane2+width*height; //gets written to
    float* outputPlane1 = mem_d+(channel+5)*pyramidSize; //gets written to
    float* outputPlane2 = mem_d2+(channel+5)*pyramidSize; //gets written to

    static groupshared float2 bufferInput[23*23];
    static groupshared float2 bufferSmallScale[10*10];

    //load input in memory: a thread's loads (3 pixels at most) are all issued before its first store, so that their
    //waits overlap
    float2 loaded[3];
    [ForceUnroll]
    for (int m = 0; m < 3; m++){
        const int i = thid + 256*m;
        if (i < 23*23){
            const int64_t x = base_x-4 + (i%23);
            const int64_t y = base_y-4 + (i/23);
            loaded[m].x = reducePadded(inputPlane1, (int)x, (int)y, (int)width, (int)height);
            loaded[m].y = reducePadded(inputPlane2, (int)x, (int)y, (int)width, (int)height);
        }
    }
    [ForceUnroll]
    for (int m = 0; m < 3; m++){
        const int i = thid + 256*m;
        if (i < 23*23) bufferInput[i] = loaded[m];
    }
    GroupMemoryBarrierWithGroupSync();

    //compute into bufferSmallScale
    if (thx < 10 && thy < 10){
        bufferSmallScale[thy*10+thx] = {0.f, 0.f};

        for (int ix = 0; ix < 5; ix++){
            const float kernelx = gaussPyrKernel[ix];

            for (int iy = 0; iy < 5; iy++){
                const float kernel = kernelx*gaussPyrKernel[iy];
                //for thx = 9, base column is 18
                bufferSmallScale[thy*10+thx] += bufferInput[(2*thy+iy)*23+2*thx+ix]*kernel;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    //export [1x1, 8x8] smallscale into plane
    if (thx < 8 && thy < 8){
        const int64_t new_x = (blockIdx.x*8 + thx);
        const int64_t new_y = (blockIdx.y*8 + thy);
        //if (new_x == 4 && new_y == 7) printf("got %f in channel %d band %d\n", bufferSmallScale[(thy+1)*10+thx+1], channel, scale);
        if (new_x < small_width && new_y < small_height){
            smallScale1[new_y*small_width+new_x] = bufferSmallScale[(thy+1)*10+thx+1].x;
            smallScale2[new_y*small_width+new_x] = bufferSmallScale[(thy+1)*10+thx+1].y;
        }
    }
    //no need to syncthreads because bufferSmallScale won't be written to again

    //use [0x0, 9x9] smallscale to expand and get local value for global x global y
    const int64_t glx = base_x + thx;
    const int64_t gly = base_y + thy;
    float2 expanded = {0.f, 0.f};

    const int parity_x = thx%2;
    const int parity_y = thy%2;

    //bufferSmallScale[0] is the small scale's pixel base/2 - 1. Like pycvvdp's gausspyr_expand, the small scale
    //repeats its edge pixels outside the image.
    const int64_t smallBase_x = base_x/2 - 1;
    const int64_t smallBase_y = base_y/2 - 1;
    //either in [0,3] or [0, 2]
    for (int dx = 0; dx < 3-parity_x; dx++){
        const float kernelx = 2*gaussPyrKernel[2*dx+parity_x];
        const int64_t small_x = clamp(smallBase_x + (thx+1)/2 + dx, (int64_t)0, small_width - 1);
        for (int dy = 0; dy < 3-parity_y; dy++){
            const float kernel = kernelx*2*gaussPyrKernel[2*dy+parity_y];
            const int64_t small_y = clamp(smallBase_y + (thy+1)/2 + dy, (int64_t)0, small_height - 1);
            expanded += bufferSmallScale[(small_y - smallBase_y) * 10 + (small_x - smallBase_x)]*kernel;
        }
    }

    //clamp + export this value as Lbkg if channel = 0 or read otherwise
    float2 lbkg = {0.f, 0.f};
    if (channel == 0){
        lbkg = max(expanded, 0.01f);
        if (glx < width && gly < height) {
            Lbkg1[gly*width+glx] = lbkg.x;
            Lbkg2[gly*width+glx] = lbkg.y;
        }
    } else {
        int64_t ref_ind_x = glx;
        int64_t ref_ind_y = gly;

        if (ref_ind_x < 0) ref_ind_x = -ref_ind_x-1;
        if (ref_ind_x >= width) ref_ind_x = 2*width - ref_ind_x - 2;

        if (ref_ind_y < 0) ref_ind_y = -ref_ind_y-1;
        if (ref_ind_y >= height) ref_ind_y = 2*height - ref_ind_y - 2;

        lbkg.x = Lbkg1[ref_ind_y*width+ref_ind_x];
        lbkg.y = Lbkg2[ref_ind_y*width+ref_ind_x];
    }

    //substract + refine to get final contrast on this local value
    const float2 Ldiff = (bufferInput[(thy+4)*23 + thx+4] - expanded);
    float2 contrast = min(Ldiff/lbkg, 1000.f);

    if (scale != 0) contrast *= 2;

    //apply gaussian precompute (notably CSF)
    const float ch_gain = (channel == 1) ? 1.45f : 1.f;

    const float S = computeSensitivityGPU(lbkg.x, scale, channel, CSFMem_d);
    contrast *= S * ch_gain;

    //if (glx == 6 && gly == 4) printf("contrast %f on channel %d and scale %d\n", contrast, channel, scale);
    if (glx < width && gly < height){
        outputPlane1[gly*width+glx] = contrast.x;
        outputPlane2[gly*width+glx] = contrast.y;
    }
}
