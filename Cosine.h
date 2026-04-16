/*
* *
**Filename: Cosine.h
* *
* *Description:
* * This file contains the implementation of the Discrete Cosine Transform (DCT)
* *  and Modified Discrete Cosine Transform (MDCT) classes, which decomposes
* *  a signal into a sum of cosine functions oscillating at different frequencies. The DCT is widely
* *  used in signal processing and data compression applications, such as JPEG image compression and MP3 audio compression. 
* *  The MDCT is a variant of the DCT that is used in audio coding, particularly in the context of perceptual audio codecs 
* *  like MP3 and AAC, where it provides better energy compaction and reduced blocking artifacts compared to the standard DCT.
* *
* *
** Author:
**  JEP, J.Enrique Peraza
* *
* *
*/
#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <complex>
#include <thread>
#include <future>
#include <chrono>
#include <stdexcept>
#include <optional>
#include "DSPWindows.h"
#include "Fourier.h"

// ----------------------------------------------------------------------------
// The discrete Cosine Transform (DCT) part of FCWTransforms.h
// Discrete Cosine / Modified Discrete Cosine Transform. 
// Supports DCT-I, DCT-II, DCT-III, and DCT-IV (1-D, real-valued).
// Supports MDCT/IMDCT (Princen-Bradley sine window pair).
// Uses the existing SpectralOps<T>::FFTStride/IFFTStride for O(N*log(N))
// No heap allocations inside the hot path (caller passes working buffers).
// --------------------------------------------------------------------------- 
template<typename T=float>
class DCT
{
  public:
    enum class Type { I, II, III, IV,MDCT,IMDCT };

    // ----------- Forward 1-D transforms ----------------------------------- 
    static vector<T> Transform (
      const vector<T>& x,               // The input signal to transform.
      Type t,                           // The transform type.
      SpectralOps<T>& engine)           // Our spectral engine.
    {                                   // ----------- Transform -------------
      switch (t)                        // Act according to transform type...
      {                                 // 
        case Type::I:     return DCTI(x,engine);  // DCT-I
        case Type::II:    return DCTII(x,engine); // DCT-II
        case Type::III:   return DCTIII(x,engine);// DCT-III
        case Type::IV:    return DCTIV(x,engine); // DCT-IV
        case Type::MDCT:  {
      // Default to PB sine window for MDCT if none is provided
      sig::spectral::Window<T> W; 
      using WT=typename sig::spectral::Window<T>::WindowType;
      W.SetWindowType(WT::MLTSine, x.size());
      return MDCT(x, W, engine);  // MDCT (length N/2 coefficients)
    }
    case Type::IMDCT: {
      // Default to PB sine window for IMDCT (output length=2*X.size())
      const size_t wlen=2*x.size();
      sig::spectral::Window<T> W;
      using WT=typename sig::spectral::Window<T>::WindowType;
      W.SetWindowType(WT::MLTSine, wlen);
      return IMDCT(x, W, engine); // IMDCT (returns 2N block for OLA)
    }
    default:          return DCTIV(x,engine); // Default to DCT-IV
      }                                 // Done dispatching transform          
    }                                   // ----------- Transform -------------
    static vector<T> Forward (
      const vector<T>& timeBlock,       // The time-domain block to transform.
      const Window<T>& analysisW,       // The analysis window to apply.
       SpectralOps<T>& engine)          // Our spectral engine.
    {                                   // ----------- Forward --------------- //
      return MDCT(timeBlock,analysisW,engine); // Use MDCT for forward transform.
    }                                   // ----------- Forward --------------- //
    // ----------- Inverse 1-D transforms -----------------------------------
    static vector<T> Inverse(
      const vector<T>& X,               // The frequency-domain block to transform.
      Type t,                           // The transform type.
    SpectralOps<T>& engine)             // Our spectral engine.
    {                                   // ----------- Inverse ----------------
      switch(t)                         // Dispatch according to type.
      {                                 //
        case Type::I:     return DCTI(X,engine,/*inverse=*/true); // DCT-I
        case Type::II:    return DCTIII(X,engine); // -1 dual
        case Type::III:   return DCTII(X,engine); // -1 dual
        case Type::IV:    return DCTIV(X,engine); // Self-inverse
        default:          return DCTIV(X,engine); // Default to DCT-II
      }                                 // Done dispatching inverse transform
    }                                   // ----------- Inverse ----------------
    static vector<T> Inverse(
      const vector<T>& coeffs,          // The frequency-domain coefficients to transform.
      const Window<T>& synthesisW,      // The synthesis window to apply.
      SpectralOps<T>& engine)           // Our spectral engine.      
    {                                   // ----------- Inverse ----------------
      return IMDCT(coeffs, synthesisW, engine); // Use IMDCT for inverse transform.
    }                                   // ----------- Inverse ----------------
    // ---------------- Discrete Cosine Transforms ----------------
    // All four classical DCTs are computed through a single length-2N complex FFT.
    // following the well-known even/odd embeddings (see Britanak & Rao, ch. 6).
    // We use the existing SpectralOps<T>::FFTStride/IFFTStride for O(N*log(N))
    // So no extra radix-2 code is duplicated here.
    // 
    static vector<T> DCTII (
      const vector<T>& x,               // The input signal to transform.
      SpectralOps<T>& engine,           // Our spectral engine.
      const bool inverse=false)         // If it is the inverse.           
    {                                   // ----------- DCT-II ----------------
      const size_t N=x.size();          // Get the size of the input signal.
      if (N==0) return {};              // Guard: empty
      vector<std::complex<T>> w(2*N);   // Create a complex vector of size 2*N.
      // Even-symmetrical extensions (x, reversed(x))
      for (size_t n=0;n<N;++n)          // For all samples.
      {                                 // Set the even-symmetrical extensions.
        w[n]={x[n],0};                  // Fill the first N points with x[n].
        w[2*N-1-n]={x[n],0};            // Fill the last N points with x[n].
      }                                 // Done symmetrical extensions.
      // Compute the FFT of the complex vector w.
      auto W=engine.FFTStride(w);       // Compute the FFT of the complex vector w.
      // Post-twiddle & packing.
      vector<T> X(N);                   // Create a vector of size N for the output.
      const T scale=inverse?T(2)/T(N):T(1); // Orthonormal pair (II <--> III).
      const T factor=T(M_PI)/(T(2)*T(N));   // The factor to apply to the output.
      for (size_t k=0;k<N;++k)          // For all frequency bins...
      {                                 // Apply the post-twiddle and scale/pack.
        std::complex<T> c=std::exp(std::complex<T>(0,-factor*T(k)));
        X[k]=T(2)*(W[k]*c).real()*scale;// Apply the post-twiddle and scale.
      }                                 // Done post-twiddle and packing.
      return X;                         // Return the DCT-II coefficients.
    }                                   // ----------- DCT-II ----------------
    static std::vector<T> DCTIII(
      const std::vector<T>& x,          // The input signal to transform.
      SpectralOps<T>& engine)           // Our spectral engine.
    {                                   // ----------- DCT-III ---------------
      // DCT-III is the inverse of DCT-II. Call DCTII with the inverse flag
      return DCTII(x,engine,true);      // Call DCT-II with the inverse flag.
    }                                   // ----------- DCT-III ---------------
    static vector<T> DCTI(
      const std::vector<T>& x,          // The input signal to transform.
      SpectralOps<T>& engine,           // Our spectral engine.
      const bool inverse=false)         // If it is the inverse.
    {                                   // ----------- DCT-I -----------------
      // Even-even extensions --> length-2(N-1) FFT.
      const size_t N=x.size();          // Get the size of the input signal.
      if (N<2) return x;                // If the input signal is too short, return it.
      vector<std::complex<T>> w(2*(N-1)); // Create a complex vector of size 2*(N-1).
      // Fill the complex vector with the even-even extensions.
      for (size_t n=0;n<N-1;++n)        // For all samples.
      {                                 // Set the even-even extensions.
        w[n]={x[n],0};                  // Fill the first N-1 points with x[n].
        w[2*(N-1)-1-n]={x[N-2-n],0};    // Fill the last N-1 points with x[N-2-n].
      }                                 // Done even-even extensions.
      w[N-1]={x.back(),0};              // Fill the middle point with x[N-1].
      auto W=engine.FFTStride(w);       // Compute the FFT of the complex vector w.
      vector<T> X(N);                   // Create a vector of size N for the output.
      const T scale=inverse?T(1)/T(N-1):T(1); // Orthonormal pair (I <--> IV).
      for (size_t k=0;k<N;++k)          // For all frequency bins...
        X[k]=W[k].real()*scale;         // Apply the post-twiddle and scale.
      return X;                         // Return the DCT-I coefficients.
    }                                   // ----------- DCT-I -----------------
    static vector<T> DCTIV(
      const vector<T>& x,               // The input signal to transform.
      SpectralOps<T>& engine)           // Our spectral engine.
    {                                   // ----------- DCT-IV ----------------
      const size_t N=x.size();          // Get the size of the input signal.
      if (N==0) return {};              // Guard: empty
      vector<std::complex<T>> w(2*N);   // Create a complex vector of size 2*N.
      // Fill the complex vector with the even-even extensions.
      for (size_t n=0;n<N;++n)          // For all samples...
      {
        w[n]={x[n],0};                  // Fill the first N points with x[n].
        w[2*N-1-n]={-x[n],0};           // Fill the last N points with -x[n].
      }                                 // Done symmetrical extensions.
      auto W=engine.FFTStride(w);       // Compute the FFT of the complex vector w.
      vector<T> X(N);                   // Create a vector of size N for the output.
      const T factor=T(M_PI)/(T(4)*T(N)); // The factor to apply to the output.
      for (size_t k=0;k<N;++k)          // For all frequency bins....
      {                                 
        std::complex<T> c=std::exp(std::complex<T>(0,-factor*(T(2)*T(k)+T(1))));
        X[k]=T(2)*(W[k].real()*c.real()-W[k].imag()*c.imag());// Apply the post-twiddle and scale.
      }                                 // Done post-twiddle and packing.
      return X;                         // Self-inverse (orthogonal, no extra scaling).
    }                                   // ----------- DCT-IV ----------------
  // --------------------- MDCT/IMDCT ---------------------
  // MDCT length is N (produces N/2 coefficients from a 2N-sample block).
  // IMDCT returns a 2N block that the caller overlap-adds by N samples.
  // -----------------------------------------------------------
  static vector<T> MDCT(
    const vector<T>& timeBlock,         // The time-domain block to transform.
    const Window<T>& win,               // The window to apply before MDCT.
    SpectralOps<T>& engine)             // Our spectral engine.
  {                                     // ------------ MDCT ---------------
    // Preconditions:
  const size_t n2=timeBlock.size();   // Get the size of the input block.
    if (n2%2)                           // If a remainder appears after division by 2....
      return vector<T>();               // Return an empty vector.
    const size_t N=n2/2;                // N is the number of MDCT coefficients.
    // -------------------------------- //
    // Windowing+pre-twiddle (+ phase shift n+0.5).
    // -------------------------------- //
    vector<std::complex<T>> xwin(n2);   // Create a complex vector of size n2.
    for (size_t n=0;n<n2;++n)           // Fpr all samples...
      xwin[n]={timeBlock[n]*win[n],0}; // Apply the window to the time block.
    // -------------------------------- //
    // Rearrange into even-odd groups for FFT of length 2N.
    // -------------------------------- //
    vector<std::complex<T>> v(n2);      //  Create a complex vector of size n2.
    for (size_t n=0;n<N;++n)            // For all samples....
    {                                   // Fill the complex vector with the even-odd groups.
      v[n]=xwin[n]+xwin[n2-1-n];        // Even part: x[n]+w[n2-1-n].
      v[n+N]=(xwin[n]-xwin[n2-1-n])*std::complex<T>{0,-1};     // Odd part: x[n]-w[n2-1-n].
    }                                   //
    auto V=engine.FFTStride(v);         // Compute the FFT of the complex vector v.
    // Post-twiddle: take real part, multiply by exp(-j(pi*k+0.5)/N)                                                
    vector<T> X(N);                    // Create a vector of size N for the output.
    const T factor=T(M_PI)/(T(2)*T(N)); // The factor to apply to the output.
    for (size_t k=0;k<N;++k)            // For all frequency bins...
    {                                   // Apply the post-twiddle and scale.
      std::complex<T> c=std::exp(std::complex<T>(0,-factor*(T(k)+T(0.5))));
      X[k]=(V[k]*c).real();             // Take the real part and apply the post-twiddle.
    }                                   // Done post-twiddle and packing.
    return X;                           // Return the MDCT coefficients.
  }                                     // ------------ MDCT ---------------
  static vector<T> IMDCT(
    const vector<T>& X,                 // The frequency-domain block to transform.
    const Window<T>& win,               // The window to apply after IMDCT.
    SpectralOps<T>& engine)             // Our spectral engine.
  {                                     // ------------ IMDCT ---------------
    const size_t N=X.size();            // Get the size of the input block.
    const size_t n2=N*2;                //
    // Pre-twiddle
    vector<std::complex<T>> V(n2);      // Create a complex vector of size n2.
    const T factor=T(M_PI)/(T(2)*T(N)); // The factor to apply to the output.
    for (size_t k=0;k<N;++k)            // For all frequency bins....
    {
      std::complex<T> c=std::exp(std::complex<T>(0,factor*(T(k)+T(0.5))));
      V[k]=X[k]*c;                      // Apply the pre-twiddle.
      V[k+N]=-X[k]*c;                   // odd symmetry: V[k+N]=-X[k]*c.
    }                                   // 
    // IFFT but actually FFT of size 2N, then scale/flip.
    auto v=engine.IFFTStride(V);        // Compute the IFFT of the complex vector V.
    // Post-twiddle+windowing.
    vector<T> y(n2);                     // Output signal.
    for (size_t n=0;n<n2;++n)           // For all samples...
      y[n]=T(2)*(v[(n+N/2)%n2].real())*win[n]; // Apply the post-twiddle and windowing.
    return y;           // Caller must perform 50% OLA with previous frame (block).
  }                                     // ------------ IMDCT ---------------
};  

//============================================================================
// MDCT Beat Granulator (free function)
// Depends on: Window<T>, DCT<T>::MDCT/IMDCT and a SpectralOps<T>& engine.
//============================================================================
//===================== MDCT Beat Granulator (with Pitch & Stutter)=====================
template<typename T=float>
class MDCTBeatGranulator {
public:
  struct Config {
    double bpm        {120.0};   // musical tempo
    double stutterBeats {0.0};   // 0=off; else duration in beats for a frozen grain
    double pitchRatio {1.0};     // 1.0=no shift; 2.0=+12 st; 0.5=-12 st
    size_t windowLen  {2048};    // MDCT operates on 2N samples; windowLen== 2N
    size_t hop        {0};       // if 0, defaults to 50% overlap (N)
  };

  MDCTBeatGranulator(const Config& cfg,
                     typename Window<T>::WindowType wtype=Window<T>::WindowType::MLTSine)
  : cfg_(cfg)
  {
    if(cfg_.windowLen % 2) cfg_.windowLen++;            // force even 2N
    N_=cfg_.windowLen/2;                             // MDCT length
    hop_= (cfg_.hop==0) ? N_ : cfg_.hop;                // default 50% OLA
    W_.SetWindowType(wtype, cfg_.windowLen);            // PB-compatible sine by default
    lastFreezeStart_=0;
    framesPerStutter_=0; // computed at prepare()
  }

  // Call this once you know sample rate and before processing
  void prepare(double sampleRate) {
    fs_=sampleRate;
    // how many *output* samples per beat:
    const double samplesPerBeat=(60.0 / cfg_.bpm)*fs_;
    // how many analysis hops fit in a stutter chunk:
    if (cfg_.stutterBeats > 0.0) {
      const double stutterSamples=cfg_.stutterBeats*samplesPerBeat;
      framesPerStutter_=std::max<size_t>(1, static_cast<size_t>(std::round(stutterSamples / hop_)));
    } else {
      framesPerStutter_=0;
    }
    writePtr_=0;
    frozen_ =false;
    frozenX_.clear();
    xFrame_.assign(cfg_.windowLen, T(0));
    ola_.assign(cfg_.windowLen, T(0));
    ring_.clear();
    ring_.reserve(8); // tiny cache of recent frames? MDCT coeffs
  }

  // Process a block of audio. This consumes input with hop-sized increments and returns OLA?d output.
  // You can feed this in a pull model: push hop samples at a time and collect hop samples back.
  std::vector<T> processHop(const std::vector<T>& inHop,
                            SpectralOps<T>& engine)
  {
    if (inHop.size()!=hop_) throw std::runtime_error("MDCTBeatGranulator: inHop size must equal hop.");
    // Slide the analysis window buffer
    std::rotate(xFrame_.begin(), xFrame_.begin()+hop_, xFrame_.end());
    std::copy(inHop.begin(), inHop.end(), xFrame_.end()-hop_);

    // 1) MDCT
    auto X=DCT<T>::MDCT(xFrame_, W_, engine);           // length N_

    // 2) optionally enter/maintain stutter
    if (framesPerStutter_>0) {
      if (!frozen_) {
        // start a freeze chunk at interval
        if ((frameCount_-lastFreezeStart_) >= framesPerStutter_) {
          lastFreezeStart_=frameCount_;
          frozen_ =true;
          frozenX_=X;      // capture the grain
          freezeIdx_= 0;
        }
      }
    }

    // 3) choose source spectrum (live or frozen)
    const std::vector<T>& src=(frozen_ ? frozenX_ : X);

    // 4) apply pitch shift in MDCT domain (bin resample)

    std::vector<T> Xp=(cfg_.pitchRatio==1.0) ? src : pitchShiftBins(src, cfg_.pitchRatio);

    // 5) IMDCT ? overlap-add
    auto y2N=DCT<T>::IMDCT(Xp, W_, engine);             // returns 2N samples
    // standard 50% OLA: emit last hop from the middle
    // keep an OLA buffer so windows sum to 1
    for (size_t n=0;n<cfg_.windowLen;++n) {
      ola_[n] += y2N[n];
    }
    std::vector<T> out(hop_);
    std::copy(ola_.begin()+N_/2, ola_.begin()+N_/2+hop_, out.begin());
    // slide down OLA buffer
    std::rotate(ola_.begin(), ola_.begin()+hop_, ola_.end());
    std::fill(ola_.end()-hop_, ola_.end(), T(0));

    // 6) manage stutter lifetime
    if (frozen_) {
      ++freezeIdx_;
      if (freezeIdx_ >= framesPerStutter_) {
        frozen_=false; // release freeze, resume live updates
      }
    }

    // save a small history if you want to audition different frames
    if (ring_.size()<ring_.capacity()) ring_.push_back(X);
    else { ring_[ringHead_]=X; ringHead_=(ringHead_+1)%ring_.capacity(); }

    ++frameCount_;
    return out;
  }

  // Runtime setters
  void setPitchRatio(double r){ cfg_.pitchRatio=std::max(0.01, r); }
  void setStutterBeats(double b) {
    cfg_.stutterBeats=std::max(0.0, b);
    if (fs_>0) prepare(fs_); // recompute framesPerStutter_
  }
  void setBPM(double bpm) {
    cfg_.bpm=std::max(1.0, bpm);
    if (fs_>0) prepare(fs_);
  }

  const Config& config() const { return cfg_; }

private:
  // Lightweight MDCT-bin resampler for pitch shifting.
  // Reindexes magnitudes in the real MDCT domain with linear interpolation.
  // (For higher quality, add phase-aware methods in STFT or PQMF banks.)
  std::vector<T> pitchShiftBins(const std::vector<T>& X, double ratio)
  {
    const size_t N=X.size();
    std::vector<T> Y(N, T(0));
    for (size_t k=0;k<N;++k) {
      double srcIdx=static_cast<double>(k)/ratio;
      if (srcIdx<0 || srcIdx>static_cast<double>(N-1)) continue;
      size_t i0=static_cast<size_t>(std::floor(srcIdx));
      size_t i1=std::min(N-1, i0+1);
      double t=srcIdx-static_cast<double>(i0);
      Y[k]=static_cast<T>((1.0-t)*X[i0]+t*X[i1]);
    }
    return Y;
  }

private:
  Config cfg_;
  size_t N_{0};
  size_t hop_{0};
  double fs_{0.0};

  Window<T> W_;
  std::vector<T> xFrame_;      // rolling 2N analysis buffer
  std::vector<T> ola_;         // 2N synthesis OLA buffer

  // stutter state
  size_t framesPerStutter_{0};
  size_t lastFreezeStart_{0};
  bool   frozen_{false};
  size_t freezeIdx_{0};
  std::vector<T> frozenX_;

  // tiny ring of recent MDCT frames (optional, handy for UI scrubbing)
  std::vector<std::vector<T>> ring_;
  size_t ringHead_{0};

  // counters
  size_t writePtr_{0};
  size_t frameCount_{0};
};
