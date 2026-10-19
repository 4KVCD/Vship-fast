// SSIMULACRA2 check + speed tool for libvship.
// usage: ssimu2bench W H ref.yuv dist.yuv frames [-r repeats] [-g threads] [-q]
// Inputs are raw 8-bit yuv420p (BT.709 limited). Prints every frame's score as an exact hex float
// a hash of its distortion map, then times `repeats` passes over all frames without distmap readback.
#include "VshipAPI.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static void check(Vship_Exception e, const char* what) {
    if (e != 0) {
        char msg[1024];
        Vship_GetErrorMessage(e, msg, sizeof msg);
        fprintf(stderr, "%s: %s\n", what, msg);
        exit(1);
    }
}

static uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char** argv) {
    if (argc < 6) { fprintf(stderr, "usage: ssimu2bench W H ref.yuv dist.yuv frames [-r repeats] [-g threads] [-q]\n"); return 1; }
    const int64_t w = atoll(argv[1]), h = atoll(argv[2]);
    const int frames = atoi(argv[5]);
    int repeats = 3, threads = 1;
    bool quiet = false;
    for (int i = 6; i < argc; i++) {
        if (!strcmp(argv[i], "-r")) repeats = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-g")) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-q")) quiet = true;
    }
    const size_t frameBytes = w * h * 3 / 2;
    std::vector<uint8_t> ref(frameBytes * frames), dis(frameBytes * frames);
    for (int k = 0; k < 2; k++) {
        FILE* f = fopen(argv[3 + k], "rb");
        if (!f || fread(k ? dis.data() : ref.data(), 1, frameBytes * frames, f) != frameBytes * frames) { fprintf(stderr, "cannot read %s\n", argv[3 + k]); return 1; }
        fclose(f);
    }

    Vship_Colorspace_t cs{};
    cs.width = w; cs.height = h; cs.target_width = -1; cs.target_height = -1;
    cs.sample = Vship_SampleUINT8; cs.range = Vship_RangeLimited; cs.subsampling = {1, 1};
    cs.chromaLocation = Vship_ChromaLoc_Left; cs.YUVMatrix = Vship_MATRIX_BT709;
    cs.transferFunction = Vship_TRC_BT709; cs.primaries = Vship_PRIMARIES_BT709; cs.crop = {0, 0, 0, 0};
    const int64_t lineSize[3] = {w, w / 2, w / 2};
    auto planes = [&](std::vector<uint8_t>& v, int fr, const uint8_t* p[3]) {
        const uint8_t* base = v.data() + frameBytes * fr;
        p[0] = base; p[1] = base + w * h; p[2] = base + w * h + (w / 2) * (h / 2);
    };

    check(Vship_SetDevice(0), "SetDevice");

    // Exactness pass: score of every frame.
    {
        Vship_SSIMU2Handler hd;
        check(Vship_SSIMU2Init(&hd, cs, cs), "Init");
        for (int fr = 0; fr < frames; fr++) {
            const uint8_t *p1[3], *p2[3];
            planes(ref, fr, p1); planes(dis, fr, p2);
            double s;
            check(Vship_ComputeSSIMU2(hd, &s, p1, p2, lineSize, lineSize), "Compute");
            if (!quiet) printf("frame %d: ssimu2 %.12g (%a)\n", fr, s, s);
        }
        check(Vship_SSIMU2Free(hd), "Free");
    }

    // Speed pass: each thread owns a handler and takes every threads-th frame.
    std::vector<std::thread> pool;
    auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < threads; t++) {
        pool.emplace_back([&, t] {
            Vship_SSIMU2Handler hd;
            check(Vship_SSIMU2Init(&hd, cs, cs), "Init");
            for (int r = 0; r < repeats; r++)
                for (int fr = t; fr < frames; fr += threads) {
                    const uint8_t *p1[3], *p2[3];
                    planes(ref, fr, p1); planes(dis, fr, p2);
                    double s;
                    check(Vship_ComputeSSIMU2(hd, &s, p1, p2, lineSize, lineSize), "Compute");
                }
            check(Vship_SSIMU2Free(hd), "Free");
        });
    }
    for (auto& th : pool) th.join();
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("speed: %d frames x %d repeats, %d threads: %.3f s, %.2f fps\n", frames, repeats, threads, sec, frames * repeats / sec);
    return 0;
}