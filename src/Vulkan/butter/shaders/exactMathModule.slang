module exactMathModule;

// Division and square root rounded like the CPU's (libjxl uses the CPU's, which IEEE 754 rounds correctly). Vulkan
// allows the GPU's own division 2.5 ulp of error and its square root more, so each result is corrected with one
// fused multiply-add step from the exact remainder.

// a/b correctly rounded
public float divide(float a, float b){
    const float r = 1.0f/b;
    const float q = a*r;
    const float e = fma(-q, b, a);
    return fma(e, r, q);
}

// sqrt(x) correctly rounded (sqrt's own result for 0, negative, infinite and NaN x)
public float sqrtRounded(float x){
    const float s = sqrt(x);
    if (!(x > 0.0f) || isinf(x)) return s;
    const float remainder = fma(-s, s, x);
    return fma(remainder, 0.5f/s, s);
}

// Double-float arithmetic: a value held as hi + lo, two floats, |lo| at most half an ulp of hi, about 46 bits of
// precision. It reproduces libjxl's double computations at float speed: a result is taken from it only when it is
// far enough from a float rounding boundary that libjxl's double result, rounded to float, cannot differ (decided()),
// and the rare other cases are recomputed in double. These algorithms need every operation rounded as written: the
// driver keeps them so (the NoContraction decorations of spvFileToCppHeader.cpp), and slangc, which simplifies float
// expressions as if they were exact (it turns twoSum's error term into 0), is kept from it by passing each rounded
// sum or product through opaque() with zero, a run-time 0 given to the kernel.
public struct FF {
    public float hi;
    public float lo;
};

public FF ff(float hi, float lo){
    FF r;
    r.hi = hi;
    r.lo = lo;
    return r;
}

// x unchanged, through an operation slangc cannot see through (zero is 0 at run time)
public float opaque(float x, uint zero){
    return asfloat(asuint(x) | zero);
}

// a + b exactly, when |a| >= |b| or a is 0
FF quickTwoSum(float a, float b, uint zero){
    const float s = opaque(a + b, zero);
    return ff(s, b - (s - a));
}

// a + b exactly
public FF twoSum(float a, float b, uint zero){
    const float s = opaque(a + b, zero);
    const float bb = opaque(s - a, zero);
    return ff(s, (a - (s - bb)) + (b - bb));
}

// a*b exactly
public FF twoProd(float a, float b, uint zero){
    const float p = opaque(a*b, zero);
    return ff(p, fma(a, b, -p));
}

public FF ffAdd(FF a, FF b, uint zero){
    const FF s = twoSum(a.hi, b.hi, zero);
    return quickTwoSum(s.hi, s.lo + (a.lo + b.lo), zero);
}

public FF ffAddFloat(FF a, float b, uint zero){
    const FF s = twoSum(a.hi, b, zero);
    return quickTwoSum(s.hi, s.lo + a.lo, zero);
}

public FF ffMulFloat(FF a, float b, uint zero){
    const FF p = twoProd(a.hi, b, zero);
    return quickTwoSum(p.hi, fma(a.lo, b, p.lo), zero);
}

public FF ffMul(FF a, FF b, uint zero){
    const FF p = twoProd(a.hi, b.hi, zero);
    return quickTwoSum(p.hi, fma(a.hi, b.lo, fma(a.lo, b.hi, p.lo)), zero);
}

public FF ffDiv(FF a, FF b, uint zero){
    const float q1 = opaque(a.hi/b.hi, zero);
    const FF r = ffAdd(a, ffMulFloat(b, -q1, zero), zero);
    const float q2 = r.hi/b.hi;
    return quickTwoSum(q1, q2, zero);
}

public FF ffNeg(FF a){
    return ff(-a.hi, -a.lo);
}
// Whether every real number within errorBound of v rounds to the same float as v.hi, so that a double computation of
// v with an absolute error under errorBound rounds to v.hi.
public bool decided(FF v, float errorBound){
    const uint bits = asuint(abs(v.hi));
    if (bits == 0u || bits >= 0x7f800000u) return false;
    const float ulpUp = asfloat(bits + 1u) - abs(v.hi);
    // the nearest boundary is on lo's side (both sides when lo is 0); the gap below a power of two is half the gap
    // above it
    const bool towardZero = (v.lo == 0.0f) || ((v.lo < 0.0f) == (v.hi > 0.0f));
    const float gap = (towardZero && (bits & 0x7fffffu) == 0u) ? 0.5f*ulpUp : ulpUp;
    return 0.5f*gap - abs(v.lo) > errorBound;
}

// The sign of v (-1, 0 or 1) when it is certain for any value within errorBound of v, else 0 with certain false
public int signDecided(FF v, float errorBound, out bool certain){
    certain = abs(v.hi) > errorBound;
    return (v.hi > 0.0f) ? 1 : ((v.hi < 0.0f) ? -1 : 0);
}

#if FP64
// a/b correctly rounded, in double
public double divide64(double a, double b){
    const double r = 1.0/b;
    const double q = a*r;
    const double e = fma(-q, b, a);
    return fma(e, r, q);
}
#endif
