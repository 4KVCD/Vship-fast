module vshipColorModule;

public enum Vship_Sample_t{
    Vship_SampleFLOAT = 0,
//    Vship_SampleHALF = 1,
    Vship_SampleUINT8 = 2,
    Vship_SampleUINT9 = 3,
    Vship_SampleUINT10 = 5,
    Vship_SampleUINT12 = 7,
    Vship_SampleUINT14 = 9,
    Vship_SampleUINT16 = 11,
};

public enum Vship_Range_t{
    Vship_RangeLimited = 0,
    Vship_RangeFull = 1,
};

public enum Vship_ColorFamily_t{
    Vship_ColorYUV,
    Vship_ColorRGB,
};

public enum Vship_YUVMatrix_t{
    Vship_MATRIX_RGB = 0,
    Vship_MATRIX_BT709 = 1,
    Vship_MATRIX_BT470_BG = 5,
    Vship_MATRIX_ST170_M = 6, //same as 5
    Vship_MATRIX_YCGCO = 8,
    Vship_MATRIX_BT2020_NCL = 9,
    Vship_MATRIX_BT2020_CL = 10,
    Vship_MATRIX_BT2100_ICTCP = 14,
    Vship_MATRIX_YCGCO_RE = 16,
    Vship_MATRIX_YCGCO_RO = 17,
};

public enum Vship_TransferFunction_t{
    Vship_TRC_BT709 = 1,
    Vship_TRC_BT470_M = 4,
    Vship_TRC_BT470_BG = 5,
    Vship_TRC_BT601 = 6, //same as 5
    Vship_TRC_ST240_M = 7,
    Vship_TRC_Linear = 8,
    Vship_TRC_sRGB = 13,
    Vship_TRC_PQ = 16,
    Vship_TRC_ST428 = 17,
    Vship_TRC_HLG = 18,
};

public enum Vship_Primaries_t{
    Vship_PRIMARIES_INTERNAL = -1, //corresponds to XYZ really
    Vship_PRIMARIES_BT709 = 1,
    Vship_PRIMARIES_BT470_M = 4,
    Vship_PRIMARIES_BT470_BG = 5,
    Vship_PRIMARIES_ST170_M = 6,
	Vship_PRIMARIES_ST240_M = 7,  // Equivalent to 6.
    Vship_PRIMARIES_BT2020 = 9,
    Vship_PRIMARIES_DisplayP3 = 12,
};

public int bytesizeSample(Vship_Sample_t sampleType){
    switch (sampleType){
        case Vship_Sample_t.Vship_SampleUINT8:
            return 1;
        case Vship_Sample_t.Vship_SampleUINT9:
        case Vship_Sample_t.Vship_SampleUINT10:
        case Vship_Sample_t.Vship_SampleUINT12:
        case Vship_Sample_t.Vship_SampleUINT14:
        case Vship_Sample_t.Vship_SampleUINT16:
//        case Vship_Sample_t.Vship_SampleHALF:
            return 2;
        case Vship_Sample_t.Vship_SampleFLOAT:
            return 4;
    }
    return 0;
}

public int bitprecisionSample(Vship_Sample_t sampleType){
    switch (sampleType){
        case Vship_Sample_t.Vship_SampleUINT8:
            return 8;
        case Vship_Sample_t.Vship_SampleUINT9:
            return 9;
        case Vship_Sample_t.Vship_SampleUINT10:
            return 10;
        case Vship_Sample_t.Vship_SampleUINT12:
            return 12;
        case Vship_Sample_t.Vship_SampleUINT14:
            return 14;
        case Vship_Sample_t.Vship_SampleUINT16:
            return 16;
        default:
            return 1;
    }
}