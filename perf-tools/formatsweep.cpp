// Exactness sweep over input formats for libvship: builds two frames of every listed format (sample type, range,
// chroma subsampling and location, matrix, transfer, primaries, odd sizes, crop, resize) from an 8-bit 4:2:0 clip,
// scores them with SSIMULACRA2, Butteraugli and CVVDP and prints every score as an exact hex float, so that the
// outputs of two builds can be compared line by line.
// usage: formatsweep W H ref.yuv dist.yuv   (raw 8-bit yuv420p, at least 2 frames)
#include "VshipAPI.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string lastError;
static bool ok(Vship_Exception e) {
    if (e == 0) return true;
    char msg[1024];
    Vship_GetErrorMessage(e, msg, sizeof msg);
    lastError = msg;
    return false;
}

struct Format {
    const char* name;
    int width, height;
    Vship_Sample_t sample;
    Vship_Range_t range;
    int subw, subh;
    Vship_ChromaLocation_t location;
    Vship_YUVMatrix_t matrix;
    Vship_TransferFunction_t transfer;
    Vship_Primaries_t primaries;
    Vship_CropRectangle_t crop;
    int targetWidth, targetHeight; // -1: no resize
};

// The source: 8-bit 4:2:0 frames
struct Source {
    int w, h;
    std::vector<uint8_t> data;
    const uint8_t* Y(int f) const { return data.data() + (size_t)f * w * h * 3 / 2; }
    const uint8_t* U(int f) const { return Y(f) + w * h; }
    const uint8_t* V(int f) const { return U(f) + (w / 2) * (h / 2); }
};

// One frame of the format: 3 planes with a padded stride
struct Frame {
    std::vector<uint8_t> planes[3];
    int64_t lineSize[3];
    const uint8_t* ptr[3];
};

static int bytesOf(Vship_Sample_t s) { return s == Vship_SampleUINT8 ? 1 : (s == Vship_SampleFLOAT ? 4 : 2); }

// The 8-bit value v of pixel (x, y) in the format's sample type; low bits vary with the position so that they are
// exercised too
static void store(uint8_t* dst, Vship_Sample_t s, int v, int x, int y) {
    switch (s) {
        case Vship_SampleUINT8: *dst = (uint8_t)v; break;
        case Vship_SampleUINT10: { uint16_t q = (uint16_t)((v << 2) | ((x + 3 * y) & 3)); memcpy(dst, &q, 2); break; }
        case Vship_SampleUINT12: { uint16_t q = (uint16_t)((v << 4) | ((x * 5 + y) & 15)); memcpy(dst, &q, 2); break; }
        case Vship_SampleUINT16: { uint16_t q = (uint16_t)((v << 8) | ((x * 7 + y * 3) & 255)); memcpy(dst, &q, 2); break; }
        case Vship_SampleFLOAT: { float f = (v + ((x + y) & 7) / 8.f) / 256.f; memcpy(dst, &f, 4); break; }
        default: break;
    }
}

static Frame makeFrame(const Source& src, int f, const Format& fm) {
    Frame fr;
    const int b = bytesOf(fm.sample);
    for (int p = 0; p < 3; p++) {
        const int pw = p == 0 ? fm.width : ((fm.width - 1) >> fm.subw) + 1;
        const int ph = p == 0 ? fm.height : ((fm.height - 1) >> fm.subh) + 1;
        fr.lineSize[p] = (int64_t)pw * b + 64; // padded stride
        fr.planes[p].assign(fr.lineSize[p] * ph, 0);
        for (int y = 0; y < ph; y++) {
            for (int x = 0; x < pw; x++) {
                // the pixel of the full-resolution plane this sample stands for, clamped to the source
                const int fx = std::min(x << (p == 0 ? 0 : fm.subw), src.w - 1);
                const int fy = std::min(y << (p == 0 ? 0 : fm.subh), src.h - 1);
                int v;
                if (p == 0) v = src.Y(f)[fy * src.w + fx];
                else v = (p == 1 ? src.U(f) : src.V(f))[(fy / 2) * (src.w / 2) + fx / 2];
                store(fr.planes[p].data() + y * fr.lineSize[p] + (size_t)x * b, fm.sample, v, x, y);
            }
        }
        fr.ptr[p] = fr.planes[p].data();
    }
    return fr;
}

static Vship_Colorspace_t colorspace(const Format& fm) {
    Vship_Colorspace_t cs{};
    cs.width = fm.width; cs.height = fm.height;
    cs.target_width = fm.targetWidth; cs.target_height = fm.targetHeight;
    cs.sample = fm.sample; cs.range = fm.range;
    cs.subsampling = {fm.subw, fm.subh}; cs.chromaLocation = fm.location;
    cs.colorFamily = fm.matrix == Vship_MATRIX_RGB ? Vship_ColorRGB : Vship_ColorYUV;
    cs.YUVMatrix = fm.matrix; cs.transferFunction = fm.transfer; cs.primaries = fm.primaries;
    cs.crop = fm.crop;
    return cs;
}

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: formatsweep W H ref.yuv dist.yuv\n"); return 1; }
    Source ref{atoi(argv[1]), atoi(argv[2])}, dis{ref.w, ref.h};
    const size_t frameBytes = (size_t)ref.w * ref.h * 3 / 2;
    for (int k = 0; k < 2; k++) {
        Source& s = k ? dis : ref;
        s.data.resize(frameBytes * 2);
        FILE* f = fopen(argv[3 + k], "rb");
        if (!f || fread(s.data.data(), 1, s.data.size(), f) != s.data.size()) { fprintf(stderr, "cannot read %s\n", argv[3 + k]); return 1; }
        fclose(f);
    }

    const Vship_CropRectangle_t noCrop{0, 0, 0, 0};
    const Format formats[] = {
        {"u8-420-left-709", 640, 360, Vship_SampleUINT8, Vship_RangeLimited, 1, 1, Vship_ChromaLoc_Left, Vship_MATRIX_BT709, Vship_TRC_BT709, Vship_PRIMARIES_BT709, noCrop, -1, -1},
        {"u8-420-center", 640, 360, Vship_SampleUINT8, Vship_RangeLimited, 1, 1, Vship_ChromaLoc_Center, Vship_MATRIX_BT709, Vship_TRC_BT709, Vship_PRIMARIES_BT709, noCrop, -1, -1},
        {"u8-420-topleft", 640, 360, Vship_SampleUINT8, Vship_RangeLimited, 1, 1, Vship_ChromaLoc_TopLeft, Vship_MATRIX_BT709, Vship_TRC_BT709, Vship_PRIMARIES_BT709, noCrop, -1, -1},
        {"u8-420-top", 640, 360, Vship_SampleUINT8, Vship_RangeLimited, 1, 1, Vship_ChromaLoc_Top, Vship_MATRIX_BT709, Vship_TRC_BT709, Vship_PRIMARIES_BT709, noCrop, -1, -1},
        {"u8-422-left", 640, 360, Vship_SampleUINT8, Vship_RangeLimited, 1, 0, Vship_ChromaLoc_Left, Vship_MATRIX_BT709, Vship_TRC_BT709, Vship_PRIMARIES_BT709, noCrop, -1, -1},
        {"u8-444", 640, 360, Vship_SampleUINT8, Vship_RangeLimited, 0, 0, Vship_ChromaLoc_Left, Vship_MATRIX_BT709, Vship_TRC_BT709, Vship_PRIMARIES_BT709, noCrop, -1, -1},
        {"u8-440-center", 640, 360, Vship_SampleUINT8, Vship_RangeLimited, 0, 1, Vship_ChromaLoc_Center, Vship_MATRIX_BT709, Vship_TRC_BT709, Vship_PRIMARIES_BT709, noCrop, -1, -1},
        {"u8-410-left", 640, 360, Vship_SampleUINT8, Vship_RangeLimited, 2, 1, Vship_ChromaLoc_Left, Vship_MATRIX_BT709, Vship_TRC_BT709, Vship_PRIMARIES_BT709, noCrop, -1, -1},
        {"u8-411-center", 640, 360, Vship_SampleUINT8, Vship_RangeLimited, 2, 0, Vship_ChromaLoc_Center, Vship_MATRIX_BT709, Vship_TRC_BT709, Vship_PRIMARIES_BT709, noCrop, -1, -1},
        {"u8-sub22-center", 640, 360, Vship_SampleUINT8, Vship_RangeLimited, 2, 2, Vship_ChromaLoc_Center, Vship_MATRIX_BT709, Vship_TRC_BT709, Vship_PRIMARIES_BT709, noCrop, -1, -1},
        {"u8-sub22-top", 640, 360, Vship_SampleUINT8, Vship_RangeLimited, 2, 2, Vship_ChromaLoc_Top, Vship_MATRIX_BT709, Vship_TRC_BT709, Vship_PRIMARIES_BT709, noCrop, -1, -1},
        {"u8-sub12-topleft", 640, 360, Vship_SampleUINT8, Vship_RangeLimited, 1, 2, Vship_ChromaLoc_TopLeft, Vship_MATRIX_BT709, Vship_TRC_BT709, Vship_PRIMARIES_BT709, noCrop, -1, -1},
        {"u8-420-odd-center", 637, 357, Vship_SampleUINT8, Vship_RangeLimited, 1, 1, Vship_ChromaLoc_Center, Vship_MATRIX_BT709, Vship_TRC_BT709, Vship_PRIMARIES_BT709, noCrop, -1, -1},
        {"u8-420-odd-top", 637, 357, Vship_SampleUINT8, Vship_RangeLimited, 1, 1, Vship_ChromaLoc_Top, Vship_MATRIX_BT709, Vship_TRC_BT709, Vship_PRIMARIES_BT709, noCrop, -1, -1},
        {"u8-470bg-601", 640, 360, Vship_SampleUINT8, Vship_RangeLimited, 1, 1, Vship_ChromaLoc_Left, Vship_MATRIX_BT470_BG, Vship_TRC_BT601, Vship_PRIMARIES_BT470_BG, noCrop, -1, -1},
        {"u8-ycgco-srgb-full", 640, 360, Vship_SampleUINT8, Vship_RangeFull, 1, 1, Vship_ChromaLoc_Left, Vship_MATRIX_YCGCO, Vship_TRC_sRGB, Vship_PRIMARIES_BT709, noCrop, -1, -1},
        {"u10-2020ncl-pq", 640, 360, Vship_SampleUINT10, Vship_RangeLimited, 1, 1, Vship_ChromaLoc_TopLeft, Vship_MATRIX_BT2020_NCL, Vship_TRC_PQ, Vship_PRIMARIES_BT2020, noCrop, -1, -1},
        {"u10-2020ncl-hlg-full", 640, 360, Vship_SampleUINT10, Vship_RangeFull, 1, 1, Vship_ChromaLoc_Left, Vship_MATRIX_BT2020_NCL, Vship_TRC_HLG, Vship_PRIMARIES_BT2020, noCrop, -1, -1},
        {"u10-2020cl", 640, 360, Vship_SampleUINT10, Vship_RangeLimited, 1, 1, Vship_ChromaLoc_Left, Vship_MATRIX_BT2020_CL, Vship_TRC_BT709, Vship_PRIMARIES_BT2020, noCrop, -1, -1},
        {"u12-ictcp-pq", 640, 360, Vship_SampleUINT12, Vship_RangeLimited, 1, 1, Vship_ChromaLoc_TopLeft, Vship_MATRIX_BT2100_ICTCP, Vship_TRC_PQ, Vship_PRIMARIES_BT2020, noCrop, -1, -1},
        {"u16-422-srgb-full", 640, 360, Vship_SampleUINT16, Vship_RangeFull, 1, 0, Vship_ChromaLoc_Center, Vship_MATRIX_BT709, Vship_TRC_sRGB, Vship_PRIMARIES_DisplayP3, noCrop, -1, -1},
        {"f32-444-rgb-linear", 640, 360, Vship_SampleFLOAT, Vship_RangeFull, 0, 0, Vship_ChromaLoc_Left, Vship_MATRIX_RGB, Vship_TRC_Linear, Vship_PRIMARIES_BT709, noCrop, -1, -1},
        {"f32-444-rgb-srgb", 640, 360, Vship_SampleFLOAT, Vship_RangeFull, 0, 0, Vship_ChromaLoc_Left, Vship_MATRIX_RGB, Vship_TRC_sRGB, Vship_PRIMARIES_BT709, noCrop, -1, -1},
        {"u8-444-rgb-srgb", 640, 360, Vship_SampleUINT8, Vship_RangeFull, 0, 0, Vship_ChromaLoc_Left, Vship_MATRIX_RGB, Vship_TRC_sRGB, Vship_PRIMARIES_BT709, noCrop, -1, -1},
        {"u8-420-crop", 640, 360, Vship_SampleUINT8, Vship_RangeLimited, 1, 1, Vship_ChromaLoc_Center, Vship_MATRIX_BT709, Vship_TRC_BT709, Vship_PRIMARIES_BT709, {4, 6, 8, 2}, -1, -1},
        {"u8-420-resize", 640, 360, Vship_SampleUINT8, Vship_RangeLimited, 1, 1, Vship_ChromaLoc_Left, Vship_MATRIX_BT709, Vship_TRC_BT709, Vship_PRIMARIES_BT709, noCrop, 480, 270},
        {"u8-420-resize-crop", 640, 360, Vship_SampleUINT8, Vship_RangeLimited, 1, 1, Vship_ChromaLoc_Top, Vship_MATRIX_BT709, Vship_TRC_BT709, Vship_PRIMARIES_BT709, {2, 2, 4, 4}, 800, 450},
    };

    if (!ok(Vship_SetDevice(0))) { fprintf(stderr, "SetDevice: %s\n", lastError.c_str()); return 1; }
    for (const Format& fm : formats) {
        const Vship_Colorspace_t cs = colorspace(fm);
        Frame r[2] = {makeFrame(ref, 0, fm), makeFrame(ref, 1, fm)};
        Frame d[2] = {makeFrame(dis, 0, fm), makeFrame(dis, 1, fm)};
        std::string line = fm.name;
        char buf[256];

        Vship_SSIMU2Handler s;
        if (ok(Vship_SSIMU2Init(&s, cs, cs))) {
            for (int f = 0; f < 2; f++) {
                double v;
                if (ok(Vship_ComputeSSIMU2(s, &v, r[f].ptr, d[f].ptr, r[f].lineSize, d[f].lineSize))) snprintf(buf, sizeof buf, " ssimu2 %a", v);
                else snprintf(buf, sizeof buf, " ssimu2 error(%s)", lastError.c_str());
                line += buf;
            }
            Vship_SSIMU2Free(s);
        } else line += " ssimu2 init error(" + lastError + ")";

        Vship_ButteraugliHandler b;
        if (ok(Vship_ButteraugliInit(&b, cs, cs, 5, 203.f))) {
            for (int f = 0; f < 2; f++) {
                Vship_ButteraugliScore v;
                if (ok(Vship_ComputeButteraugli(b, &v, NULL, 0, r[f].ptr, d[f].ptr, r[f].lineSize, d[f].lineSize))) snprintf(buf, sizeof buf, " butter %a %a %a", v.normQ, v.norm3, v.norminf);
                else snprintf(buf, sizeof buf, " butter error(%s)", lastError.c_str());
                line += buf;
            }
            Vship_ButteraugliFree(b);
        } else line += " butter init error(" + lastError + ")";

        Vship_CVVDPHandler c;
        if (ok(Vship_CVVDPInit(&c, cs, cs, 24.f, false, "standard_fhd"))) {
            for (int f = 0; f < 2; f++) {
                double v;
                if (ok(Vship_ComputeCVVDP(c, &v, NULL, 0, r[f].ptr, d[f].ptr, r[f].lineSize, d[f].lineSize))) snprintf(buf, sizeof buf, " cvvdp %a", v);
                else snprintf(buf, sizeof buf, " cvvdp error(%s)", lastError.c_str());
                line += buf;
            }
            Vship_CVVDPFree(c);
        } else line += " cvvdp init error(" + lastError + ")";
        printf("%s\n", line.c_str());
        fflush(stdout);
    }
    return 0;
}
