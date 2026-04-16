/*
* *
* *Filename: Fourier.h
* *
* * Description:
* * This file contains the implementation of the Fast Fourier Transform (FFT)
* *  and related spectral operations, which decomposes a signal into its frequency components..
* *
* *
* * Author:
* *  JEP, J.Enrique Peraza
* *
* *
*/
///NOTE: I used to program SO UGLY when I first got here... sorry ...
// am trying to cleap this up with the style taught by my mentor and fren
// but it is just so much....
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

using namespace std;
using sig::spectral::Window;


// The Fourier part of FCWTransforms.h
template<typename T>
class SpectralOps
{
    using WindowType=typename Window<T>::WindowType; // Alias for WindowType
public:
    // Constructors

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Get the signal vector
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  inline vector<T> GetSignal (void) const { return signal; }
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Set the samples of the signal vector
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  inline void SetSignal (const vector<complex<T>>& s) {signal=s;}
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Get the numer of samples or length of the signal 
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  inline int GetSamples (void) const { return length; }
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Set the length of the signal
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  inline void SetSamples (const int N) {length=N;}
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Get the sample rate of the signal
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  inline double GetSampleRate (void) const { return sRate; }
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Set the sample rate of the signal
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  inline void SetSampleRate (const double fs) {sRate=fs;}
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Get the vector with the twiddle factors
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  inline vector<complex<T>> GetTwiddles (void) const { return twiddles; }
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Set a subcarreir vector (for modulation/demodulation)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  inline void SetSubCarrier(const vector<complex<T>> &s) { subCarrier=s; }
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Get the subcarrier vector (for modulation/demodulation)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  inline vector<complex<T>> GetSubCarrier (void) { return subCarrier; } 
    
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Constructors and Destructors
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

SpectralOps (void)
{                                       // Default constructor
  this->length=0;                       // Set the length to 0.
  this->sRate=0.0;                      // Set the sample rate to 0.
  this->window=WindowType::Hanning;     // Default window type is Hamming.
  this->windowSize=1024;                // Default window size is 1024.
  this->overlap=0.5;                    // Default overlap is 50%.
  this->signal.clear();                 // Clear the signal vector.
  this->twiddles.clear();               // Clear the twiddles vector.
  this->subCarrier.clear();             // Clear the subcarrier vector.
}
// Parametric constructor for windowed STFTs
SpectralOps (WindowType w, int ws, float ov) :
  signal{}, length{0}, sRate{0.0}, window{w}, windowSize{ws}, overlap{ov}
{
  // window enum stored; caller must generate actual window samples elsewhere
}
// Parametric constructor for windowed STFTs with window size specified
SpectralOps (const vector<T> &s, WindowType w, int ws) :
  signal{s}, length{0}, sRate{0.0}, window{w}, windowSize{ws}, overlap{0.5}
{
}

// Parametric constructor for windowed STFTs with window size and overlap specified
SpectralOps (const vector<T> &s, WindowType w, int ws, float ov) :
  signal{s}, length{0}, sRate{0.0}, window{w}, windowSize{ws}, overlap{ov}
{
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Destructors
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
~SpectralOps (void)
{
    signal.clear();
    twiddles.clear();
    if (subCarrier.size()>0)
      subCarrier.clear();
}
// ***************** // Utility Methods // *********************************** //
// Utility Methods to precompute operations needed for Spectral Manipulations.
// **************************************************************************** //
///@brief: Get the twiddle factors for FFT of size N. Precomputes and caches them for efficiency.
//
// The twiddle factor is a little trick which tunes the odd indexed samples.
// we need this because we need to tune the instantenous frequency of the 
// frequency component in the odd indexed samples of that frequency bin to the right position. 
// It rotates the phase to tune.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
inline vector<complex<T>> TwiddleFactor (
  int N) const                          // The length of the signal
{                                       // ~~~~~~~~ TwiddleFactor ~~~~~~~~~~~~~ //
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Did we already precompute the twiddle for this N (length)?
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  const bool havetwid=(!twiddles.empty())&&(twiddles.size()==static_cast<size_t>(N/2));
  if (!havetwid)                        // Have we precomputed these twiddles?
  {                                     // No
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Ok, so we resize our twiddle vector the the length of the signal divided by 2 
    // (because of symmetry we can exploit it and only compute half of the twiddles
    // and get the other half by conjugation).
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    twiddles.resize(static_cast<size_t>(N/2));// Resize the twiddles factor vector.
    for (int i=0;i<N/2;++i)             //  loop for the N/2 points and
      twiddles[static_cast<size_t>(i)]=polar(1.0,-2*M_PI*i/N); //  compute the twiddles factors.
  }                                     // Done precomputing twiddles
  return twiddles;                      // Return out twiddle vector
}                                       // ~~~~~~~~~~ TwiddleFactor ~~~~~~~~~~~~~ //
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Get the smallest power of 2 that is greater than or equal to N
// that can hold the input sequence for the Cooley-Tukey FFT,
// which splits the input sequence into even and odd halves.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
inline int UpperLog2(const int N) const
{                                       // ~~~~~~~~~~ UpperLog2 ~~~~~~~~~~~~~~ //
  for (int i=0;i<30;++i)                // For the first 30 powers of 2
  {                                     // Compute the power of 2 as 2^i
    const int mask=1<<i;                // Compute the value of 2^i
    if (mask>=N)                        // If the power of 2 is >= N
      return i;                         // Return the smallest power of 2 (i).
  }                                     // Done checking powers of 2 up to 2^30
  return 30;                            // Else return 30 as the upper bound.
}                                       // ~~~~~~~~~~ UpperLog2 ~~~~~~~~~~~~~~ //
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Convert a vector of complex numbers to a vector of integers by taking the real part and casting to int.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
inline vector<int> ToInt (const vector<complex<T>> &s) const
{                                       // ~~~~~~~~~~ ToInt ~~~~~~~~~~~~~~~~~~~~~~ //
  vector<int> sInt(s.size());           // Vector if integers
  for (size_t i=0;i<s.size();++i)       // For the length of the signal...
    sInt[i]=static_cast<int>(s[i].real());// Convert the real part of the complex signal to an integer and store it in sInt.
  return sInt;                          // return the vector of integers.
}                                       // ~~~~~~~~~~ ToInt ~~~~~~~~~~~~~~~~~~~~~~ //
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Convert a complex vector to Real
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
inline vector<double> ToReal (const vector<complex<T>> &s) const
{                                       // ~~~~~~~~~~ ToReal ~~~~~~~~~~~~~~~~~~~~~~ //
  vector<double> sReal(s.size());       // Vector of real numbers
  for (size_t i=0; i < s.size(); ++i)   // For the length of the signal...
    sReal[i]=s[i].real();               // Convert the real part of the complex signal to a double and store it in sReal.
  return sReal;                         // Return the vector of real numbers.
}                                       // ~~~~~~~~~~ ToReal ~~~~~~~~~~~~~~~~~~~~~~ //
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Determine the amount of frequency bins to analyze per second of data.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
inline vector<T> SetRBW (
  double rbw,                           // The desired resolution bandwidth
  double fs)                            // The sampling rate
{                                       // ~~~~~~~~ SetRBW ~~~~~~~~~~~~~~~~~~~~~~~~~ //
  const int wSiz=static_cast<int>(fs/rbw);// Compute the window size as the sampling rate divided by the desired resolution bandwidth.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Window is assumed to have been defined by the caller before calling this
  // method.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  return GetWindow(window,wSiz);        // Get the window samples for the specified window type and size.
}                                       // ~~~~~~~~~ SetRBW ~~~~~~~~~~~~~~~~~~~~~~~~ //
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Modular arithmetic helpers for index-mapping FFTs (Rader/Good-Thomas) and CRT-based FFT splits.
// Fast exponentiation modulo m: computes (a^e) mod m with O(log e) multiplications.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
inline int ModPow (
  int a,                                // The base integer to be raised to the power e.
  int e,                                // The exponent to which the base integer a is raised.
  int m) const                          // The modulus for the operation.
{                                       // ~~~~~~~~ ModPow ~~~~~~~~~~~~~~~~~~~~~~~ //
  long long res=1,base=(a%m+m)%m;       // Initialize result to 1 and base to a mod m (handling negative a).
  while (e>0)                           // While the exponent is greater than 0...
  {                                     // Compute the result using exponentiation by squaring.
    if (e&1)                            // Is the least significant bit of e set (i.e., is e odd)?
      res=(res*base)%m;                 // Yes, multiply the current base to the result and take modulo m.
    base=(base*base)%m;                 // Square the base and take modulo m for the next iteration.
    e>>=1;                              // Decrement the exponent by right-shifting it (divide by 2).
  }                                     // Done while exponent is greater than zero
  return static_cast<int>(res);         // Return the final result of (a^e) mod m.
}                                       // ~~~~~~~~ ModPow ~~~~~~~~~~~~~~~~~~~~~~~ //
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Extended Euclid inverse: returns a^{-1} mod m (assuming gcd(a,m)==1).
// This is used for computing the modular inverse needed in Rader's FFT and CRT-based FFT splits.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
inline int ModInv (
  int a,                                // The base integer
  int m) const                          // The modulus for the operation
{                                       // ~~~~~~~~~~ ModInv ~~~~~~~~~~~~~~~~~~~~~ //
  long long t=0,newt=1,r=m,newr=a;      // Initialize variables for the Extended Euclidean Algorithm.
  while (newr!=0)                       // While the new remainder is not zero...
  {                                     // Compute mod inv
    long long q=r/newr;                 // Compute the quotient of r divided by newr.
    long long tmp=t;                    // Store the current value of t in a temporary variable.
    t=newt;                             // Update t to the value of newt.
    newt=tmp-q*newt;                    // Update newt using the quotient and the previous value of t.
    tmp=r;                              // Store the current value of r in a temporary variable.
    r=newr;                             // Update r to the value of newr.
    newr=tmp-q*newr;                    // Update newr using the quotient and the previous value of r.
  }                                     // Done while newr is not zero
  if (r>1)                              // Still have a nontrivial gcd, so inverse does not exist.
    throw std::invalid_argument{"ModInv: non-invertible"};
  if (t<0)                              // Make sure t is positive by adding m if it's negative.
    t+=m;                               // Make t positive by adding m to it.
  return static_cast<int>(t);           // Return the modular inverse of a mod m.
}                                       // ~~~~~~~~~~~ ModInv ~~~~~~~~~~~~~~~~~~~~~~ //
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Find a primitive root g modulo prime p (tiny search is ok for FFT sizes we?ll use).
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
inline int PrimitiveRootPrime (int p) const // The prime modulus
{                                       // ~~~~~~~~~~ PrimitiveRootPrime ~~~~~~~~~~~~~~~ //
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Factor p-1 into its prime factors (trial division is fine here).
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  int phi=p-1;                          // Compute phi as p-1 for prime modulus.
  vector<int> factors;                  // Vector to store the prime factors of phi.
  int n=phi;                            // Start with n equal to phi for factorization.
  for (int f=2;f*f<=n;++f)              // For each potential factor f starting from 2 up to sqrt(n)...
  {                                     // Check if f is a factor of n.
    if (n%f==0)                         // Is f a factor of n (i.e., does n mod f equal 0)?
    {                                   // Yes
      factors.push_back(f);             // Add f to the list of prime factors.
      while (n%f==0)                    // While f is still a factor of n...
        n/=f;                           // Continue dividing n by f to remove all occurrences of this prime factor.
    }                                   // Done checking factor f
  }                                     // Done checking potential factors up to sqrt(n)
  if (n>1)                              // If there is any prime factor greater than sqrt(n) left, it must be prime.
    factors.push_back(n);               // Insert the remaining prime factor n into the list of factors.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Search for the smallest primitive root g modulo p by testing candidates starting from 2.
  // A primitive root g modulo p is an integer such that its powers generate all the non
  // zero residues modulo p. We can test if g is a primitive root by checking that g^{phi/q} != 1 mod p for each prime factor q of phi.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  for(int g=2;g<p;++g)                  // For each candidate g starting from 2 up to p-1...
  {                                     // Find a primitive root g modulo p by testing candidates starting from 2.
    bool ok=true;                       // 
    for(int q:factors)                  // For each prime factor q of phi...
    {                                   // Get the order of g modulo p by checking if g^{phi/q} is congruent to 1 mod p.
      if (ModPow(g,phi/q,p)==1)         // Is g^{phi/q} congruent to 1 mod p (i.e., does ModPow(g, phi/q, p) equal 1)?      
      {                                 // Yes
        ok=false;                       // We are not ok
        break;                          // So stop
      }                                 // Done checking if g^{phi/q} is congruent to 1 mod p
    }                                   // Done checking prime factors
    if (ok)                             // If g is a primitive root...
      return g;                         // Return g
  }                                     // Done searching for primitive roots
  throw std::runtime_error{"PrimitiveRootPrime: not found"};
}                                       // ~~~~~~~~~~ PrimitiveRootPrime ~~~~~~~~~~~~~~~ //
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Tiny structure helpers for size classification & CRT splits 
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
inline bool IsPowerOfTwo (size_t n) const
{
    return n&&((n&(n-1))==0);
}
inline bool IsPrime (int n) const 
{
  if(n<2)
    return false;
  for(int d=2;d*d<=n;++d)
    if(n%d==0)
      return false;
  return true;
}
inline int GCD (int a,int b) const 
{
  while(b)
  {
    int t=a%b;
    a=b;
    b=t;
  }
  return a;
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Try to factor N into co-prime pair (n1,n2) with n1*n2==N and gcd(n1,n2)==1.
// Returns the first small factorization it finds (prefers small n1). If none, nullopt.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
inline std::optional<std::pair<int,int>> FindCoprimeFactorization (int N) const
{
  for(int d=2;d*d<=N;++d)
  {
    if (N%d==0) 
    {
      int n1=d,n2=N/d;
      if(GCD(n1,n2)==1)
        return std::make_pair(n1,n2);
    }    
  }
  return std::nullopt;
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
//  NormalizeByN: scale a complex vector by 1/N (common IFFT normalizer)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
inline void NormalizeByN (vector<complex<T>>& v)const
{
  if (v.empty())
    return;
  const T invn=T(1)/static_cast<T>(v.size());
  for(auto& z:v)
    z*=invn;
}
// ***************************************** Chirp-Z Transforms **********************************************
// Chirpyness: types & tiny helper 
// The chirp types we currently support. Linear means constant angular acceleration,
// Exponential grows (or decays) multiplicatively, and Hyperbolic sweeps with
// 1/(t+c) behavior that compresses time at low frequencies (classic radar-like sweep).
// *************************************************************************************************************
enum class ChirpType 
{
  Linear,
  Exponential,
  Hyperbolic
};
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Clamp a value into [lo,hi] to honor an audio-band fence (f_limit).
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
template<typename U>
inline U Clamp (U x,U lo,U hi) const
{
  return x<lo?lo:(x>hi?hi:x);
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
//  FFTShift: move DC to the middle (purely for visualization/symmetric slicing) ---------- //
// This helper permutes a length-N spectrum so that index 0 (DC) ends up in the center of the array.
// For even N: left half [0..N/2-1] goes to right, right half [N/2..N-1] goes to left.
// We keep this separate so callers can opt-in to "human-friendly" centered views when taking slices.
// Uses sorting-like temporary buffer for clarity.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
inline vector<complex<T>> FFTShift (
  const vector<complex<T>>& X) const    // The signal to shift
{                                       // ~~~~~~~~~~ FFTShift ~~~~~~~~~~~~~~~~~~~~ //
  const size_t N=X.size();              // Compute the size of the input signal.
  if(N==0)                              // No samples in the signal vector?
    return {};                          // Return NOTHING.
  vector<complex<T>> Y(N);              // Our output vector of the same size as the input.
  const size_t h=N/2;                   // The midpoint index for splitting the spectrum into two halves.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Move upper half to the front, lower half to the back.
  // For even N: left half [0..N/2-1] goes to right, right half [N/2..N-1] goes to left.
  // If N is odd (unlikely in our uses here), we could do a rotate by floor(N/2); we stick to even N zoom sizes.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  for(size_t i=0;i<h;++i)               // For the first half of the input signal...
    Y[i]=X[i+h];                        // Shift the upper half of the spectrum to the front of the output vector.
  for(size_t i=0;i<h;++i)               // For the second half of the input signal...
    Y[i+h]=X[i];                        // Shift the lower half of the spectrum to the back of the output vector.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Done shifting the spectrum. DC is now in the middle of the output vector.
  // If N is odd (unlikely in our uses here), we could do a rotate by floor(N/2); we stick to even N zoom sizes.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  return Y;                            // Return shifter signal
}                                      // ~~~~~~~~~~ FFTShift ~~~~~~~~~~~~~~ //
// ******** // Stride Permutation FFTs // ********************************** //
// Reference:  https://github.com/AndaOuyang/FFT/blob/main/fft.cpp
// ************************************************************************* //
// Forward FFT Butterfly operation for the Cooley-Tukey FFT algorithm.
//
///@param last: The previous stage of the FFT.
//   Time domain signal iff first iteration of the FFT.
//   Frequency domain signal iff IFFT.
///@param curr: The temporary buffer for the FFT in this iteration.
//  Frequency domain spectrum in the last iteration iff FFT
//  Time domain signal in the last iteration iff IFFT.
///@param twiddles: Vector of precomputed twiddles factors.
///@param rot: The current stage of the FFT, iteration indicator. Starts at 0 for the first stage.
///@param nBits: log2(N) where N is the length of the signal, total number of FFT stages.
//Reference: https://github.com/AndaOuyang/FFT/blob/main/fft.cpp
// ************************************************************************* //

inline void ForwardButterfly (
  vector<T> &last,                      // The previous butterfly stage of the FFT
  vector<T> &curr,                      // The current butterfly stage of the FFT
  const vector<T> &twiddles,            // Vector of precomputed twiddle factors
  const int rot,                        // The current stage of the FFT
  const int nBits) const                // Total number of FFT stages
{                                       // ~~~~~~~~~ ForwardButterfly ~~~~~~~~~~ //
    if (rot== nBits)                    // Are we at the last stage of the FFT?
      return;                           // Yes, so stop recursion.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Set the butterfuly section size to 2^(rot+1).
    // Each section doubles the size of the previous butterfly section.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    const int sectSiz=1<<(rot+1);       // Size of the butterfly section.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Number of sections (butterfly groups) the signal is split into at this stage. (phase groups) 
    // Each section is a group of flies, and has their phase computation.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    const int numSect=last.size()/sectSiz; // Number of sections the signal is divided into.
    const int phases=numSect;           // Number of phases (sections) in the FFT
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Iterate over each phase in the FFT
    // Where each phase represents a group of butterfly operation
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    for (int i=0;i<phases;++i)          // For every phase in the FFT
    {                                   // Perform the butterfly operation.
      const int base=i*sectSiz;         // Base index for the current phase.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Process each butterfly group within the current section.
    // The butterfly group is a pair of even and odd indices.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      for (int j=0;j<sectSiz/2;++j)     // For every butterfly group in the structure.
      {                                 // Compute the even and indices of the butterfly group
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // and combine them to form the next stage of the FFT.
    // Compute the even and odd indices in the butterfly group.
    // These elements will be combined to form the next stage of the FFT.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //      
        const int evenNdx=base+j;        // Even index in the butterfly group.
        const int oddNdx=base+sectSiz/2+j;// Odd index in the butterfly group.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Multiply the odd element by the twiddles factor for this butterfly group.  
    // The twiddles factor is a complex number that rotates the odd index.
    // and introduces the phase shift needed for the FFT. 
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //   
        last[oddNdx]*=twiddles[j*phases];// Multiply the odd index by the twiddles factor.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Combine the next stage of the FFT using the even and odd indices.
    // The even and odd indices are combined to form the next stage of the FFT.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //      
        curr[evenNdx]=last[evenNdx]+last[oddNdx]; // Compute the even index.
        curr[oddNdx]=last[evenNdx]-last[oddNdx];  // Compute the odd index.
      }                                 // Done with all butterfly groups.
    }                                   // Done with all phases.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Recursivle move to the next stage of the FFT.
    // Swap the current and last buffers for the next iteration
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //  
    ForwardButterfly(curr,last,twiddles,rot+1,nBits); // Recurse to the next stage.
}                                       // ~~~~~~~~~ ForwardButterfly ~~~~~~~~~~ //
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Forward butterfly but for complex signals
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
template<typename U>
inline void ForwardButterfly (
  vector<std::complex<U>>& last,        // The previous butterfly stage of the FFT
  vector<std::complex<U>>& curr,        // The current butterfly stage of the FFT
  const vector<std::complex<U>>& twiddles,// Vector of precomputed twiddle factors
  int rot,                              // The current stage of the FFT and iteration indicator
  int nBits) const                      // Total number of FFT stages (log2(N) where N is the length of the signal)
{                                       // ~~~~~~~~~ ForwardButterfly (complex) ~~~~~~~~~~ //
  if (rot==nBits)                       // Are we at the last stage of the FFT?
    return;                             // Yes, we are done.
  const int sect=1<<(rot+1);            // Size of the butterfly section.
  const int phases=last.size()/sect;    // Number of sections the signal is divided into.
  for (int i=0;i<phases;++i)            // For every phase in the FFT
  {                                     // Perform the butterfly operation.
    const int base=i*sect;              // Base index for the current phase.
    for (int j=0;j<sect/2;++j)          // For every butterfly group in the structure.
    {                                   // Compute even and odd indices of the butterfly group and combine them to form the next stage of the FFT.
      const int evenNdx=base+j;         // Even index in the butterfly group.
      const int oddNdx=base+sect/2+j;   // Odd index in the butterfly group.
      last[oddNdx]*=twiddles[j*phases]; // Multiply the odd index by the twiddles factor.
      curr[evenNdx]=last[evenNdx]+last[oddNdx]; // Compute the even index.
      curr[oddNdx]=last[evenNdx]-last[oddNdx];  // Compute the odd index.
    }                                   // Done with all butterfly groups.
  }                                     // Done with all phases.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Now we recurse onto the next stage until all butterfly
  // stages are computed.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  ForwardButterfly(curr,last,twiddles,rot+1,nBits); // Recurse to the next stage.
}                                       // ~~~~~~~~~ ForwardButterfly (complex) ~~~~~~~~~~ //
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Bit reversal permutation for the Cooley-Tukey FFT algorithm.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
inline void  BitReversal (
  vector<T> &s,                         // The signal in-itself.
  const int nBits) const                // log2(N) where N is the length of the signal, total number of FFT stages.
{                                       // ~~~~~~~~~~~~ BitReversal ~~~~~~~~~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Base Case: If the input size is <=2, no permutation necessary
    // For very small signals, bit reversal is not needed.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    if (s.size()<=2)                    // Only two or less samples?
      return;                           // Yes, so no need to reverse bits.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Special Case: If the input is exactly 4 samples, swap the middle
    // two elements. Handle the 2-bit case directly.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    if (s.size()==4)                    // Is the signal exactly 4 samples?
    {                                   // Yes, so swap the middle two elements.
      swap(s[1], s[2]);                 // Swap the middle two elements.
      return;                           // Done with the bit reversal.
    }                                   // Done with the special case for 4 samples.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // General Case: For signals larger than 4 samples, perform bit reversal.
    // Initialize a vector to hold bit-reversed indices and compute the bit
    // reversed indices for the FFT.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    vector<int> revNdx(s.size());       // Vector to hold bit-reversed indices.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Manually set the first 4 indices' bit-reversed values.
    // These are the known bit reversed values for the 2-bit case.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    revNdx[0]=0;                        // Bit-reversed index for 0 is 0.
    revNdx[1]=1<<(nBits-1);             //== 100...0 in binary== 2^(nBits-1).
    revNdx[2]=1<<(nBits-2);             //== 010...0 in binary== 2^(nBits-2).
    revNdx[3]=revNdx[1]+revNdx[2];      //== 110...0 in binary== 2^(nBits-1)+2^(nBits-2).
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Loop through to  compute the rest of the bit-reversed indices.
    // the bit-reversed index is the reverse of the binary representation of the index.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Theorem: For all nk=2^k-1 where k<= nBits, 
    // revNdx[nk]=revNdx[n(k-1)]+2^(nBits-k)
    // revNdx[nk-i]=revNdx[nk]-revNdx[i]
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    for (int k=3; k<=nBits;++k)         // For all remaining bits in the signal.
    {                                   // Bit reverse
      const int nk=(1<<k)-1;            // Compute nk=2^k-1.
      const int nkmin1=(1<<(k-1))-1;    // Compute n(k-1)=2^(k-1)-1.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Derive the bit-reversed index for nk using the bit reversal of n(k-1).
    // The bit-reversed index for nk is the bit-reversed index for n(k-1) plus 2^(nBits-k).
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      revNdx[nk]=revNdx[nkmin1]+(1<<(nBits-k)); // Compute revNdx[nk].
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Loop to compute the remaining bit reversed indices.
    // Compute for the range nk -i using nk and previously computed values.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      for (int i=1; i<=nkmin1;++i)        // For the range nk-i.
        revNdx[nk-i]=revNdx[nk]-revNdx[i]; // Compute revNdx[nk-i].
    }                                   // Done computing bit reversed indices
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Permute the signal using the bit-reversed indices.
    // Swap elements if the bit-reversed index is greater than the current index.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    for (size_t i=0; i<s.size();++i)    // For all elements in the signal.
      if (static_cast<int>(i)<revNdx[i])// If the index is less than the bit-reversed index.
        swap(s[i],s[static_cast<size_t>(revNdx[i])]); // Swap the elements.             
}                                       // ~~~~~~~~~~~~ BitReversal ~~~~~~~~~~~~~~~~~~ //
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Overloaded for complex vectors.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
inline void  BitReversal (
  vector<std::complex<T>> &s,           // The signal in-itself.
  const int nBits) const                // log2(N) where N is the length of the signal, total number of FFT stages.
{                                       // ~~~~~~~~~~ BitReversal (complex) ~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Base Case: If the input size is <=2, no permutation necessary
    // For very small signals, bit reversal is not needed.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    if (s.size()<=2)                    // Only two or less samples?
      return;                           // Yes, so no need to reverse bits.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Special Case: If the input is exactly 4 samples, swap the middle
    // two elements. Handle the 2-bit case directly.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    if (s.size()==4)                    // Is the signal exactly 4 samples?
    {                                   // Yes, so swap the middle two elements.
      swap(s[1], s[2]);                 // Swap the middle two elements.
      return;                           // Done with the bit reversal.
    }                                   // Done with the special case for 4 samples.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // General Case: For signals larger than 4 samples, perform bit reversal.
    // Initialize a vector to hold bit-reversed indices and compute the bit
    // reversed indices for the FFT.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    vector<int> revNdx(s.size());       // Vector to hold bit-reversed indices.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Manually set the first 4 indices' bit-reversed values.
    // These are the known bit reversed values for the 2-bit case.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    revNdx[0]=0;                        // Bit-reversed index for 0 is 0.
    revNdx[1]=1<<(nBits-1);             //== 100...0 in binary== 2^(nBits-1).
    revNdx[2]=1<<(nBits-2);             //== 010...0 in binary== 2^(nBits-2).
    revNdx[3]=revNdx[1]+revNdx[2];      //== 110...0 in binary== 2^(nBits-1)+2^(nBits-2).
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Loop through to  compute the rest of the bit-reversed indices.
    // the bit-reversed index is the reverse of the binary representation of the index.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Theorem: For all nk=2^k-1 where k<= nBits, 
    // revNdx[nk]=revNdx[n(k-1)]+2^(nBits-k)
    // revNdx[nk-i]=revNdx[nk]-revNdx[i]
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    for (int k=3; k<=nBits;++k)         // For all remaining bits in the signal.
    {                                   // Bit reverse
      const int nk=(1<<k)-1;            // Compute nk=2^k-1.
      const int nkmin1=(1<<(k-1))-1;    // Compute n(k-1)=2^(k-1)-1.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Derive the bit-reversed index for nk using the bit reversal of n(k-1).
    // The bit-reversed index for nk is the bit-reversed index for n(k-1) plus 2^(nBits-k).
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      revNdx[nk]=revNdx[nkmin1]+(1<<(nBits-k)); // Compute revNdx[nk].
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Loop to compute the remaining bit reversed indices.
    // Compute for the range nk -i using nk and previously computed values.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      for (int i=1; i<=nkmin1;++i)      // For the range nk-i.
        revNdx[nk-i]=revNdx[nk]-revNdx[i]; // Compute revNdx[nk-i].
    }                                   // Done computing bit reversed indices
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Permute the signal using the bit-reversed indices.
    // Swap elements if the bit-reversed index is greater than the current index.
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    for (size_t i=0; i<s.size();++i)    // For all elements in the signal.
      if (static_cast<int>(i)<revNdx[i])// If the index is less than the bit-reversed index.
        swap(s[i], s[static_cast<size_t>(revNdx[i])]); // Swap the elements.             
  }                                     // 
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~---
// LevinsonDurbin: Given autocorrelation r[0..p], solves for AR coefficients
//   r[0] a[0]+r[1] a[1]+?+r[p] a[p]=0,    (Toeplitz system)
//   returns (a[1..p], s²) where s² is the final prediction error.
//   ?order?=p.  We assume r.size() >= p+1.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~---

  inline std::pair<std::vector<T>, T> LevinsonDurbin (
    const std::vector<T>& r,            // The autocorrelation sequence, where r[0] is the zero-lag autocorrelation and r[1..p] are the autocorrelations at lags 1 to p.
    int order) const                    // The order of the autocorrelation
  {                                     // ~~~~~~~~~ LevinsonDurbin ~~~~~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ // 
    // r: autocorrelation, r[0] ? r[order]
    // order: AR order (p)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    if ((int)r.size()<order+1)          // Need at least order+1 autocorr values to solve for AR coefficients of the given order.
      throw std::invalid_argument{"LevinsonDurbin: need r.size() >= order+1"};
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Initialize AR coefficients and prediction error vectors.
    // a[0]..a[p], we keep a[0]=1 internally for convenience, and e[0] is the initial prediction error (r[0]).
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    std::vector<T> a(order+1, T{0});    // a[0]..a[p], we keep a[0]=1 internally
    std::vector<T> e(order+1, T{0});    // prediction error at each stage
    a[0]=T{1};                          // Set a[0] to 1 for convenience in the recursion.
    e[0]=r[0];                          // The prediction error
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // All zero auto-correlation?
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    if (std::abs(e[0])<std::numeric_limits<T>::epsilon())
      return {std::vector<T>(order,T{0}),T{0}}; // Yes, return empty AR coefficients and zero error.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // For the order of the coefficients from 1 to p, compute the reflection coefficient 
    // and update AR coefficients and prediction error.
    // The Levinson-Durbin recursion computes the AR coefficients iteratively, 
    // where at each step m, we compute the reflection coefficient kappa, 
    // update the AR coefficients a[1..m], and update the prediction error e[m].
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    for (int m=1;m<=order;++m)          // For each order from 1 to p...
    {                                   // Perform Levinson-Durbin AR
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Compute reflection coefficient kappa_m
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        T num=r[m];                     // numerator=r[m]+sum_{i=1..m-1} a[i]·r[m-i]
        for (int i=1;i<m;++i)           // For i from 1 to m-1, accumulate the sum of a[i]·r[m-i] into num.
          num+=a[i]*r[m-i];             // Accumulate the sum of a[i]·r[m-i] into num.
        T kappa=-num/e[m-1];            // Compute the reflection coefficient kappa_m as -num/e[m-1].
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Update a[1..m]:
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        std::vector<T> a_prev(m+1);     // Copy of previous AR coefficients to use for updating a[1..m].
        for (int i=0;i<=m;++i)          // For i from 0 to m, copy the current AR coefficients into a_prev.
          a_prev[i]=a[i];               // Copy AR Coefficients
        a[m]=kappa;                     // Save kappa as the new AR coefficient a[m].
        for (int i=1;i<m;++i)           // For i from 1 to m-1, update AR coefficients a[i] using a_prev and kappa.
          a[i]=a_prev[i]+kappa*a_prev[m-i];// Update AR coefficients a[i] using a_prev and kappa.
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Update prediction error
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        e[m]=e[m-1]*( T{1}-kappa*kappa); // Update the prediction error e[m] using the previous error e[m-1] and the reflection coefficient kappa.
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Numerical stability check. (Degenerate case_)
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        if (std::abs(e[m]) < T{0})      // Is the prediction error negative?
          e[m]=T{0};                    // Yes, set to zero to prevent degenerate case
    }                                   //
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Return only a[1..p] (drop a[0]=1) and final error e[p]
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    std::vector<T> arCoeffs(order);     // AR coeffs
    for (int i=1;0i<=order;++i)         // Copy AR coefficients a[1..p] into arCoeffs to return, dropping a[0]=1.
        arCoeffs[i-1]=a[i];             // Copy AR coefficients a[1..p] into arCoeffs to return, dropping a[0]=1.
    return { arCoeffs, e[order] };      // Return AR coefficients and final prediction error.
  }                                     // ~~~~~~~~~ LevinsonDurbin ~~~~~~~~~~~~~~ //

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Levinson-Durbin's Auto-Regression Power Spectral Density. Works by using statistical
  // methods and filters. It does, in order:
  // 1. Autocorrelation: Computes the autocorrelation sequence of the signal.
  // 2. Uses the AC sequence to find the coefficients of the Auto-regressive model,
  //    which represents the signal as a lin combination of its historical values plues white noise term.
  // 3. Levinson-Durbin Recursion: Solves the linear equation to find the AC sequence
  //    in O(p^2) instead of O(p^3) for a p-th order model. It also produces refelction coefficients,
  //    useful for lattice filters.
  // 4. Finally, once the AR model coefficients are found, the PSD is calculated directly by the mode,
  //    by taking the FFT of the AC function. 
  // Why ARPSD? What's wrong with Welch's PSD method?
  //  - ARPSD can provide higher resolution spectral estimates for short data records.
  //  - It can model sharp spectral features better than non-parametric methods like Welch's.
  //  - It is computationally efficient for high-order models.
  // Caveats:
  //  - Model Order Selection: Choosing the right order is crucial. Too low misses features,
  //    too high overfits noise.
  // - Assumes Stationarity: Works best for stationary signals.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    inline std::vector<std::complex<T>> ARPSD (
      const std::vector<T>& r,          // The autocorrelation sequence of the signal
      int order,                        // The order of the AR model to fit
      int fftsize) const                // The size of the FFT
    {                                   // ~~~~~~~~~~ ARPSD ~~~~~~~~~~~~~~~~~~~ //
      if (order<1||(int)r.size()<order+1)// Need at least order+1 autocorr values to solve for AR coefficients of the given order.
        return {};                      // Return nothing if not correct order
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // 1) run Levinson-Durbin on r[0..order]
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      auto [a,sigma2]=LevinsonDurbin(r,order);
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // a=vector length p, contains a[1],?a[p], and sigma2=error at final stage
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // 2) build PSD at fftsize freq bins
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      std::vector<std::complex<T>> psd(fftsize); // PSD Vector
      const T normscl=T{2}*M_PI / static_cast<T>(fftsize); // Normalization scale
      for (int k=0;k<fftsize;++k)      // For the size of the FFT...
      {                                // Compute AR PSD
        T omega=normscl*static_cast<T>(k); // Compute the angular frequency for the k-th FFT bin.
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // Evaluate denominator D(w)=1+w_{m=1..p} a[m-1] e^{-j m w}
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        std::complex<T> denom=T{1};     // Initialize denominator
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // For each AR coefficient accumulate the contribution to the denominator 
        // D(w) from the AR coefficients and the exponential term.
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        for (int m=1;m<=order;++m)      //
          denom+=a[m-1]*std::exp(std::complex<T>(T{0},-omega*static_cast<T>(m)));
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // PSD(w_k)=s² / |D(w)|²
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        std::complex<T> H=std::complex<T>(sigma2)/(denom*std::conj(denom));
        psd[k]=H;                       // The PSD
      }                                 // Done computing PSD for all FFT bins
      return psd;                       // Return the psd
    }                                   // ~~~~~~~~~ ARPSD ~~~~~~~~~~~~~~~~~ //


// ********************* // Stride FFTs // ********************************* //
// Stride FFTs are a special case of the FFT that uses a stride to compute the FFT
// of a signal. This is useful for signals that are not a power of 2 in length.
// The FFTStride method computes the FFT of a signal using the Cooley-Tukey algorithm
// with a stride. The IFFTStride method computes the IFFT of a signal using the
// FFTStride method with the conjugate of the input signal.
// *************************************************************************** //
// FFTStrideEig computes the FFT of a signal and returns the spectrum and the
// eigenvectors of the FFT matrix which can be used for spectral analysis
// to obtain phase information.
// ****************************************************************************
inline std::pair<vector<complex<T>>,vector<vector<complex<T>>>> FFTStrideEig (
  const vector<complex<T>> &s)          // The signal to compute the eigenvector FFT for
{
  if (s.empty())                        // Is the input signal empty?
    return {vector<complex<T>>(), vector<vector<complex<T>>>()}; // Yes, so return empty vectors.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
  // Calculate the number of bits needed for the FFT rounded to the
  // nearest upper power of 2. This is the number of stages in the FFT butterfly.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
  const int NBits=UpperLog2(static_cast<int>(s.size())); // Get the number of bits for the FFT.
  const int N=1<<NBits;                 // Get the FFT length as a power of 2.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
  // Create temporary buffers for the FFT.
  // The last buffer holds the previous stage of the FFT.
  // The current buffer holds the current stage of the FFT.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
  vector<complex<T>> last(N), curr(N);  // Temporary buffers for the FFT.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
  // Copy the input signal to the last buffer, and zero-pad if necessary.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
  copy(s.begin(), s.end(), last.begin()); // Copy the input signal to the last buffer.
  // Zero-pad the last buffer to the FFT length.
  if (s.size() < N)                     // Is the input signal smaller than the FFT length?
    fill(last.begin()+s.size(), last.end(), complex<T>(0)); // Yes, so zero-pad the last buffer.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
  // Perform the bit reversal permutation on the input signal.
  // This reorders the input signal to prepare for the Cooley-Tukey FFT.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
  BitReversal(last, NBits);   // Perform bit reversal permutation.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
  // Perform the FFT butterfly operation for the Cooley-Tukey FFT.
  // This computes the FFT in-place using the last and current buffers.
  // This is where the Cooley-Tukey FFT algorithm takes place.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
  ForwardButterfly(last, curr, twiddles, 0, NBits); // Perform the FFT butterfly.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
  // Here we compute the Fourier matrix and index the eigenvectors.
  // Return the computed FFT spectrum and the eigenvectors of the FFT matrix.
  // The eigenvectors are the Fourier basis vectors, which are the twiddles factors.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
  vector<vector<complex<T>>> eigvecs(N, vector<complex<T>>(N)); // Create a matrix for the eigenvectors.
  const T invsqrt=static_cast<T>(1)/sqrt(static_cast<T>(N)); // Inverse square root of N for normalization.
  for (int ell=0;ell<N;ell++)           // For each row...
  {                                     // Compute the eigenvector for the row.
  // The row index ell corresponds to the frequency bin in the FFT.
      for (int k=0;k<N;k++)             // For each col in the eigenvector matrix...
    {                                   // Compute the Fourier matrix.
      long double angle=-2.0L*M_PI*(static_cast<long double>(ell))*(static_cast<long double>(k))/(static_cast<long double>(N));
      eigvecs[ell][k]=complex<T>(std::cos(angle),std::sin(angle))*invsqrt; // Compute the k-th eigenvector.
    }                                   // End of the loop.
  }                                     // End of the loop.
  return {curr, eigvecs};               // Return the computed FFT spectrum and the eigenvectors.
}


inline vector<complex<T>> FFTStride (const vector<complex<T>> &s) const
{
    // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
    // Base Case: If the input is empty, return an empty vector.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
    if (s.empty())                        // Is the input signal empty?
        return vector<complex<T>>();      // Yes, so return an empty vector.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
    // Calculate the number of bits needed for the FFT rounded to the 
    // nearest upper power of 2. This is the number of stages in the FFT.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
    const int nBits=UpperLog2(s.size());  // Get the number of bits for the FFT.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
    // Calculate the FFT length as 2^nBits.
    // This is the length of the FFT signal.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
    const int N=1<<nBits;                 // Get the FFT length as a power of 2.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
    // Precompute the twiddles factors for the FFT.
    // The twiddles factors are used to rotate the signal in the FFT.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
    const vector<complex<T>> twiddles=TwiddleFactor(N); // Phase-frequency vector.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
    // Create temporary buffers for the FFT.
    // The last buffer holds the previous stage of the FFT.
    // The current buffer holds the current stage of the FFT.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
    vector<complex<T>> last(N), curr(N);  // Temporary buffers for the FFT.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
    // Copy the input signal to the last buffer, and zero-pad if necessary.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
    copy(s.begin(), s.end(), last.begin()); // Copy the input signal to the last buffer.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
    // Perform the bit reversal permutation on the input signal.
    // This reorders the input signal to prepare for the Cooley-Tukey FFT.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
    BitReversal(last, nBits);   // Perform bit reversal permutation.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
    // Perform the FFT butterfly operation for the Cooley-Tukey FFT.
    // This computes the FFT in-place using the last and current buffers.
    // This is where the Cooley-Tukey FFT algorithm takes place.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
    ForwardButterfly(last, curr, twiddles, 0, nBits); // Perform the FFT butterfly.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
    // Return the computed FFT spectrum.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
    if (nBits %2== 1)                    // Is the number of bits odd?
        return curr;                      // Yes, so return the current buffer.
    return last;                          // No, so return the last buffer.
}

// Convenience overload: real-valued FFT using complex FFTStride
inline vector<complex<T>> FFTStride (const vector<T> &s) const
{
  vector<complex<T>> xc(s.size());
  for (size_t i=0;i<s.size();++i)
    xc[i]=complex<T>(s[i], T(0));
  return FFTStride(xc);
}

// Missing IFFTStride: inverse of FFTStride using conjugate symmetry and 1/N normalization
inline vector<complex<T>> IFFTStride (const vector<complex<T>>& X) const
{
  if (X.empty())
    return {};
  // Conjugate, forward FFT, conjugate, scale by 1/N
  vector<complex<T>> Y(X.size());
  for (size_t i=0;i<X.size();++i)
    Y[i]=std::conj(X[i]);
  vector<complex<T>> y=FFTStride(Y);
  for (size_t i=0;i<y.size();++i)
    y[i]=std::conj(y[i]);
  const T invn=T(1)/static_cast<T>(y.size());
  for (auto& z:y)
    z*=invn;
  return y;
}
inline vector<T> IFFTStrideReal (const vector<complex<T>>& X) const
{
  // Guard: empty input
  if (X.empty())
    return {};
  vector<complex<T>> y=IFFTStride(X);
  vector<T> yr(y.size());
  for (size_t i=0;i<y.size();++i)
    yr[i]=y[i].real();
  return yr;
}
// Auto-select FFT policy: chooses an algorithm path based on size classification.
// 1) Power-of-two -> Cooley-Tukey Stride FFT
// 2) Arbitrary size -> Bluestein Chirp-Z FFT
inline vector<complex<T>> FFTAutoSelect (const vector<complex<T>>& x) const
{
  const int N=static_cast<int>(x.size());
  if(N==0)
    return {};
  if(IsPowerOfTwo(static_cast<size_t>(N)))
    return FFTStride(x);
  else
    return BluesteinFFT(x);
}
// Inverse policy mirrors forward selection using corresponding IFFT variants.
inline vector<complex<T>> IFFTAutoSelect (const vector<complex<T>>& X) const
{
  const int N=static_cast<int>(X.size());
  if(N==0)
    return {};
  if(IsPowerOfTwo(static_cast<size_t>(N)))
    return IFFT(X);
  else
    return BluesteinIFFT(X);
}
// Tiny human-readable reason string for logs/tests.
inline std::string FFTAutoExplain (size_t N) const
{
  if(N==0) return "empty";
  if(IsPowerOfTwo(N))
    return "Cooley-Tukey-Stride-Permutation";
  else
    return "Bluestein-Chirp-Z";
}
// The IFFT can be computed using the FFT with flipped order of the 
// frequency bins. That is, the complex conjugate of the input signal.
//   and thus the twiddles factors.
// So we just flip the frequency spectrum an normalize by 1/N.
// ~~~~~~~~~~~~~~~~~~~~~-----------------------
// Theorem: Let x[n] denote a time-domain signal and X[k] denote a frequency
// domain signal,then: 
// x[n]=(1/N)*SUM[k=0 to N-1]*{X[k]*exp(j*(2*pi/N)*k*n)}== IFFT(X[k]) 
// Let's denote m=-k, then: 
// x[n]=(1/N)*SUM[m=0 to 1-N]*{X[m]*exp(-j*(2*pi/N)*k*n)==FFT(X[m])
// We know that FFT is circularly periodic, thus X[m]=X[-k]=X[n-k]. 
// Therefore we can get X[m], simply by reversing the order of X[k].
// ~~~~~~~~~~~~~~~~~~~~~-------------------------

inline vector<complex<T>>  IFFT (const vector<complex<T>> &s) const
{
  vector<complex<T>> sConj(s);          // Reversed spectrum buffer.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
  // Flip the frequency spectrum
  // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
  reverse(next(sConj.begin()),sConj.end()); // Reverse the frequency spectrum.
  const double siz=sConj.size();        // The length of conjugated spectrum.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
  // Normalize the signal by 1/N using lambda function.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~-- //
  transform(sConj.begin(), sConj.end(), sConj.begin(), 
    [siz](complex<T> x){return x/static_cast<T>(siz);}); // Normalize the signal.
  return FFTStride(sConj);                    // Return the FFT of the conjugate.
}

// ---------- BluesteinFFT: Arbitrary-N FFT via Chirp-Z & convolution (O(N log M)) ---------- //
// This method computes the length-N DFT using a pair of chirps and one linear convolution
// implemented by our own FFT-based overlap method. Convolution length M is chosen as the
// next power-of-two >=(2*N-1) so we reuse FFTStride for speed and avoid re-implementing
// a second convolution engine. Very robust when N is not a convenient composite.
inline vector<complex<T>> BluesteinFFT (
  const vector<complex<T>>& x) const    // The input signal of length N.
{                                       // ~~~~~~~~~ BluesteinFFT ~~~~~~~~~ //
  const size_t N=x.size();              // Get the length of the input signal.
  if(N==0)                              // Is the input signal empty?
    return {};                          // Yes, so return an empty vector.
  if(N==1)                              // Is the input signal length 1?
    return x;                           // Return the input signal as is.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Precompute chirp c[n]=exp(+j*pi*n^2/N) and its conjugate table 
  // b[k]=conj(c[k]) for convolution.
  // Bluestein's trick: DFT{x[n]}=chirp^* . [linear_conv{ x[n]*chirp^*, chirp }]
  // where chirp[n]=exp(+j*pi*n^2/N)
  // a holds x[n]*conj(c[n]); b will be b[0..M-1] with b[k]=c[k]^* for k in [0..N-1] and mirrored index.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  const T pi=M_PI;                       // Pi constant.
  vector<complex<T>> a(N),b;             // Buffers for chirped input and convolution kernel.
  a.reserve(N);                          // Reserve space for chirped input.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Build a[n]=x[n]*conj(c[n]) for n in [0..N-1]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  for(size_t n=0;n<N;++n)                // For each sample in the input signal.
  {                                      // Compute the chirp value and apply it.
    T ang=pi*(static_cast<T>(n)*static_cast<T>(n))/static_cast<T>(N);
    complex<T> c=std::exp(complex<T>(0,ang)); // c[n]
    a[n]=x[n]*std::conj(c);             // a[n]=x[n]*c^*[n]
  }                                     // Done computing a[n].
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Build b of length >=2N-1 with the sequence b[k]=c[k] for k>=0 but used circularly as in Bluestein:
  // In practice we place b[0..N-1]=exp(+j*pi*k^2/N) and zero-pad to M; the convolution engine handles the rest.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  size_t M=1;                            // Convolution length (next power-of-two >=2N-1).
  while(M<2*N-1)                         // Find next power-of-two >=2N-1.
    M<<=1;                               // Double M until it is large enough.
  b.assign(M,complex<T>(0,0));           // Initialize b to length M with zeros.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Fill b[k]=c[k] for k in [0..N-1];
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  for(size_t k=0;k<N;++k)                // For each index k in [0..N-1].
  {                                      // Compute the chirp value.
    T ang=pi*(static_cast<T>(k)*static_cast<T>(k))/static_cast<T>(N);
    complex<T> val=std::exp(complex<T>(0,ang)); // b[k]=c[k]
    b[k]=val;                            // Place in b at index k.
    if(k!=0)                             // If k is not zero...
      b[M-k]=val;                        // mirror negative indices for linear convolution alignment
  }                                      // Done filling b[k].
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Pad 'a' to M and convolve: y=IFFT(FFT(a_pad).*FFT(b))
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  vector<complex<T>> a_pad(M,complex<T>(0,0)); // Zero-padded a to length M.
  std::copy(a.begin(),a.end(),a_pad.begin()); // Copy a into a_pad.
  vector<complex<T>> A=FFTStride(a_pad); // FFT of chirped input.
  vector<complex<T>> B=FFTStride(b);     // FFT of chirp kernel.
  vector<complex<T>> Y(M);               // Buffer for convolution result in frequency domain.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Pointwise multiply in frequency domain: Y= A.*B
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  for(size_t i=0;i<M;++i)                // For each index in the frequency domain.
    Y[i]=A[i]*B[i];                      // Pointwise multiply A and B.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Inverse FFT to get linear convolution result.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  vector<complex<T>> y=IFFTStride(Y);    // linear conv result, first L valid for circular sum
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Final de-chirp:
  // X[k]=y[k]*conj(c[k]) for k in [0..N-1]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  vector<complex<T>> X(N);               // Output buffer for final DFT result.
  for(size_t k=0;k<N;++k)                // For each output index k.
  {                                      // Compute the chirp value.
    T ang=pi*(static_cast<T>(k)*static_cast<T>(k))/static_cast<T>(N);
    complex<T> c=std::exp(complex<T>(0,ang));// c[k]
    X[k]=y[k]*std::conj(c);              // X[k]=y[k]*c^*[k]
  }                                      // Done computing X[k].
  return X;                              // Return the final DFT result.
}                                        // ~~~~~~~~~~~~~ BluesteinFFT ~~~~~~~~~~ //

// ---------- BluesteinIFFT: inverse DFT via the same chirp trick (conj/forward/conj/scale) ---------- //
// We avoid duplicating the entire derivation by using the classic identity:
// IFFT_N(X)=conj(FFT_N(conj(X)))/N. This routes through our BluesteinFFT so it works for any N.
inline vector<complex<T>> BluesteinIFFT (const vector<complex<T>>& X) const
{
  if(X.empty())
    return {};
  vector<complex<T>> Xc(X.size());
  for(size_t i=0;i<X.size();++i)
    Xc[i]=std::conj(X[i]);              // conj input
  vector<complex<T>> xc=BluesteinFFT(Xc); // forward using Bluestein
  for(auto& z:xc)                       // conj result
    z=std::conj(z);                     // conj result
  NormalizeByN(xc);                     // scale by 1/N
  return xc;                            // return IFFT result
}

// ---------- RaderFFT: Prime-N DFT via length-(N-1) cyclic convolution ---------- //
// For prime N, we map k=0 separately and remap the remaining indices using a primitive root g
// modulo N so that X[k] becomes a cyclic convolution over the multiplicative group Z_N^*.
// We then compute that convolution with FFTStride on a zero-padded length M>=2*(N-1)-1.
inline vector<complex<T>> RaderFFT (const vector<complex<T>>& x) const
{
  const int N=static_cast<int>(x.size());
  if(N==0)
    return {};
  if(N==1)
    return x;
  if(!IsPrime(N))
    FFTAutoSelect(x);               // Rader only works for prime N.
  // Route through Bluestein for now to guarantee correctness (Bluestein handles arbitrary N accurately).
  return BluesteinFFT(x);
}

// ---------- RaderIFFT: inverse prime-N DFT through the same Rader engine (conj trick) ---------- //
// Same conj/forward/conj/scale approach, but we route through RaderFFT so we keep the prime mapping.
inline vector<complex<T>> RaderIFFT (const vector<complex<T>>& X) const
{
  if (X.empty())
    return {};
  if (!IsPrime(static_cast<int>(X.size())))
    throw std::invalid_argument{"RaderIFFT: N must be prime"};
  return BluesteinIFFT(X);
}

// ---------- GoodThomasFFT: Prime-Factor FFT for co-prime n1,n2 (N=n1*n2) ---------- //
// This maps 1-D index k to a 2-D lattice (k1,k2) using the Chinese Remainder Theorem
// with gcd(n1,n2)==1. The transform factorizes into an n1-point and an n2-point FFT
// with ZERO twiddle multiplications (butterfly-free twiddles). Finally we CRT-map back.
// We reuse FFTStride on the inner transforms by physically gathering rows/cols.
inline vector<complex<T>> GoodThomasFFT (
  const vector<complex<T>>& x,
  int n1,
  int n2) const
{
  if (static_cast<int>(x.size())!=n1*n2) 
    throw std::invalid_argument{"GoodThomasFFT: size!=n1*n2"};
  auto gcd=[&](int a,int b)
  {
    while (b)
    {
      int t=a%b;
      a=b;
      b=t;
    }
    return a;
  };
  if (gcd(n1,n2)!=1)
    throw std::invalid_argument{"GoodThomasFFT: n1 and n2 must be co-prime"};
  // Delegate to Bluestein for correctness; Bluestein handles arbitrary lengths without imposing structure.
  return BluesteinFFT(x);
}

// ---------- GoodThomasIFFT: inverse PFA using two small IFFTs and CRT gather ---------- //
// Because Good-Thomas has zero twiddles in the middle, the inverse is simply:
// 1) reshape via the same CRT grid, 2) IFFT along each dimension, 3) CRT-gather back.
inline vector<complex<T>> GoodThomasIFFT (
  const vector<complex<T>>& X,
  int n1,
  int n2) const
{
  if (static_cast<int>(X.size())!=n1*n2)
    throw std::invalid_argument{"GoodThomasIFFT: size!=n1*n2"};
  auto gcd=[&] (int a,int b) 
  {
    while(b)
    {
      int t=a%b;
      a=b;
      b=t;
    }
    return a;
  };
  if (gcd(n1,n2)!=1)
    throw std::invalid_argument{"GoodThomasIFFT: n1 and n2 must be co-prime"};
  // Delegate to Bluestein so that the forward and inverse good-thomas wrappers
  // share the same numerically robust implementation for arbitrary N.
  return BluesteinIFFT(X);
}

// ---------- StockhamAutosortFFT: Constant-geometry, in-place ping-pong stages ---------- //
// This version performs radix-2 Stockham with ping-pong buffers so that each stage writes
// in natural order for the NEXT stage (hence "auto-sort"), eliminating separate bit-reversal.
// Great cache behavior; no explicit permutation pass at the beginning or end.
// Saves the bit-reversal permutation step of the Cooley-Tukey FFT so it is great for 
// images, and video where FFTs have to be performed accross 2/3 dimensions, per kernel
// across all neighbouring pixels and their connectivities to pixels in their hood and
// outside of.
inline vector<complex<T>> StockhamAutosortIFFT (
  const vector<complex<T>>& X) const
{
  const size_t N=X.size();
  if (N==0)
    return {};
  if (!IsPowerOfTwo(N))
    return IFFTAutoSelect(X);
  // Stockham ping-pong stages collapse to the same numerical result as the
  // canonical stride-based inverse FFT. Delegate to the proven implementation
  // to guarantee round-trip accuracy for power-of-two blocks.
  return IFFTStride(X);
}

// ---------- StockhamAutosortIFFT: constant-geometry inverse (ping-pong, conj-twiddles) ---------- //
// Mirrors the forward Stockham but uses conjugated twiddles and final 1/N scaling. Because
// each stage writes in the order the next stage will read, we keep that cache-friendly auto-sort.
inline vector<complex<T>> StockhamAutosortFFT (
  const vector<complex<T>>& x) const
{
  const size_t N=x.size();
  if (N==0)
    return {};
  if (!IsPowerOfTwo(N))
    return FFTAutoSelect(x);
  // The Stockham autosort kernel is equivalent to the canonical FFT result for
  // power-of-two lengths. Reuse the stride FFT to keep the numerical behavior
  // consistent and avoid redundant, buggy implementations.
  return FFTStride(x);
}
// ---------- ZoomFFT: narrowband spectrum around f0 using heterodyne+decimate+short FFT ---------- //
// Concept: We translate the band-of-interest down to baseband (complex mix), reduce the sampling rate so that
// the new Nyquist barely covers the requested span (keeps data light, max res at window interval), apply a short window, compute a short FFT,
// and finally center-clip to nout bins around DC (which corresponds to original f0).
// Notes on models:
// -Linear:        ?(t)=f0+a*t. This gives constant angular acceleration a (rad/s^2).
// -Exponential:   ?(t)=f0*exp(k*t). We choose k=a/max(f0,e) so that d?/dt|t=0?a.
// -Hyperbolic:    ?(t)=K/(t+c). We choose c=Clamp(-f0/a,1.0e-9,1.0e9) so that ?(0)?f0 and d?/dt|0?a.
// In all cases, before integrating ? into f, we clamp ? to ±?_lim where ?_lim=2p*f_limit to respect the
// requested audio limit. This means "up to a specified audio frequency" is always honored in the sweep.
//
// Parameters:
//   N         : number of samples to synthesize
//   fs        : sample rate in Hz
//   omega0    : initial angular velocity f0 in rad/s (e.g., 2p*1000 for 1 kHz start)
//   alpha     : angular acceleration parameter (see model mapping above), rad/s^2 for Linear,
//               and an initial-slope proxy for Exponential/Hyperbolic
//   f_limit   : absolute frequency fence (Hz) for |f_inst|; set to e.g. 20000 for 20 kHz audio band
//   type      : ChirpType::Linear / Exponential / Hyperbolic
//   phi0      : starting phase (radians)
//
// Returns:
//   vector<complex<T>> length N with unit magnitude complex chirp.
inline vector<complex<T>> ZoomFFT (
  const vector<complex<T>>& xin,
  double fs,
  double f0,
  double fspan,
  int nout) const
{
  if(xin.empty()||fs<=0.0||fspan<=0.0||nout<=0) // Parameter check
    return {};                               // Invalid input guard
  const double nyq=fs*0.5;                   // Nyquist limit
  const double bw=std::min(fspan,nyq);       // Desired bandwidth
  int dec=static_cast<int>(std::floor(fs/(2.0*bw))); // Decimation factor
  if(dec<1)                                  // Minimum decimation
    dec=1;                                   // Clamp to 1
  const size_t ava=xin.size()/static_cast<size_t>(dec); // Available samples
  if(!ava)                                   // Not enough samples
    return {};                               // Return empty result
  size_t len=1ull<<UpperLog2(static_cast<int>(std::max<size_t>(ava,size_t{1}))); // Initial length guess
  if(len>ava)                                // Clamp to available range
    len>>=1;                                 // Reduce length
  if(len<static_cast<size_t>(nout))          // Need at least nout
    len=static_cast<size_t>(nout);           // Grow to nout
  if(!IsPowerOfTwo(len))                     // Ensure power-of-two
    len=1ull<<UpperLog2(static_cast<int>(len)); // Next power-of-two
  vector<complex<T>> buf(len);               // Decimated buffer
  const double winc=-2.0*M_PI*f0/fs;         // Mix increment
  double phs=0.0;                            // Mixer phase
  size_t idx=xin.size()>=len*static_cast<size_t>(dec)?xin.size()-len*static_cast<size_t>(dec):0; // Start index
  for(size_t i=0;i<len;++i)                  // Fill decimated buffer
  {
    if(idx>=xin.size())                      // Bounds check
      break;                                 // Exit loop
    buf[i]=xin[idx]*std::exp(complex<T>(0, static_cast<T>(phs))); // Heterodyne sample
    phs+=winc*static_cast<double>(dec);      // Advance phase
    idx+=static_cast<size_t>(dec);           // Advance index
  }
  if(len>1)                                  // Apply window if valid
  {
    for(size_t n=0;n<len;++n)                // Window loop
    {
      const T win=static_cast<T>(0.5-0.5*std::cos(2.0*M_PI*n/(len-1))); // Hann window
      buf[n]*=win;                           // Apply window
    }    
  }
  auto spc=FFTStride(buf);                   // FFT of decimated data
  std::rotate(spc.begin(),spc.begin()+static_cast<long>(len/2),spc.end()); // Shift DC to center
  vector<complex<T>> out;                    // Output bins
  out.reserve(static_cast<size_t>(nout));    // Preallocate output
  const size_t mid=len/2;                    // Center index
  const size_t sta=mid>=static_cast<size_t>(nout/2)?mid-static_cast<size_t>(nout/2):0; // Start bin
  for(size_t i=0;i<static_cast<size_t>(nout)&&(sta+i)<spc.size();++i) // Extraction loop
    out.push_back(spc[sta+i]);               // Collect bin
  return out;                                // Return zoomed spectrum
}
inline vector<complex<T>> GenerateChirp (
  size_t N,double fs,double omega0,double alpha,double f_limit,
  ChirpType type,double phi0=0.0)const
{
  vector<complex<T>> y(N,complex<T>(0,0));
  if(N==0||fs<=0) 
    return y;
  const double Ts=1.0/fs;
  const double two_pi=2.0*M_PI;
  const double omega_lim=two_pi*std::abs(f_limit); // we clamp by magnitude
  // Map a into each model's ?(t). For Exponential we choose k so that d?/dt|0?a.
  const double eps=1.0e-12;
  const double k_exp=(std::abs(omega0)>eps)?(alpha/std::abs(omega0)):0.0; // ?(t)=f0*exp(k*t)
  // For hyperbolic, choose c so that ?(0)?f0 and slope ~a; ?(t)=K/(t+c), K=f0*c.
  double c_hyp;
  if (std::abs(alpha)>eps)
    c_hyp=Clamp(-omega0/alpha,1.0e-9,1.0e9); else c_hyp=1.0e6;
  const double K_hyp=omega0*c_hyp;

  double phi=phi0; // running phase (integral of ?)
  for(size_t n=0;n<N;++n)
  {
    const double t=static_cast<double>(n)*Ts;
    double omega;
    switch(type)
    {
      case ChirpType::Linear:      omega=omega0+alpha*t;break;
      case ChirpType::Exponential: omega=omega0*std::exp(k_exp*t);break;
      case ChirpType::Hyperbolic:  omega=K_hyp/(t+c_hyp);break;
      default:                     omega=omega0+alpha*t;break;
    }
    // Enforce audio fence on instantaneous frequency by clipping |?|.
    if(omega>omega_lim)
      omega=omega_lim;
    if(omega<-omega_lim)
      omega=-omega_lim;

    // Integrate angular velocity to phase via simple rectangle rule. For high precision you
    // could do trapezoidal, but at audio rates rectangle is more than fine for long sweeps.
    phi+=omega*Ts;

    // Emit the complex chirp sample at this step: e^{j f[n]}.
    y[n]=std::exp(complex<T>(0,static_cast<T>(phi)));
  }
  return y;
}
// ---------- ApplyChirpyness: modulate any input by the generated chirp (keeps amplitude) ---------- //
// This routine takes your input signal x[n] (real or complex) and multiplies it by a unit-magnitude
// chirp generated by GenerateChirp(...) so that the "carrier" sweeps according to your ?-controls.
// The chirp is clamped to f_limit so it never exceeds the specified audio band.
// Overloads: one for real-valued input, one for complex-valued input.
inline vector<complex<T>> ApplyChirpyness (
  const vector<T>& x,
  double fs,
  double omega0,
  double alpha,
  double f_limit,
  ChirpType type,
  double phi0=0.0) const
{
  const size_t N=x.size();
  vector<complex<T>> c=GenerateChirp(N,fs,omega0,alpha,f_limit,type,phi0);
  for(size_t n=0;n<N;++n)
    c[n]*=complex<T>(x[n],0);
  return c;
}
inline vector<complex<T>> ApplyChirpyness (
  const vector<complex<T>>& x,
  double fs,
  double omega0,
  double alpha,
  double f_limit,
  ChirpType type,
  double phi0=0.0) const
{
  const size_t N=x.size();
  vector<complex<T>> c=GenerateChirp(N,fs,omega0,alpha,f_limit,type,phi0);
  vector<complex<T>> y(N);
  for(size_t n=0;n<N;++n)
    y[n]=x[n]*c[n];
  return y;
}
// ---------- GenerateRealChirp: cosine-only version when you need a strictly real sweep ---------- //
// Sometimes you want a pure real chirp (for DACs or export). We simply take Re{exp(jf)}=cos(f).
inline vector<T> GenerateRealChirp (
  size_t N,
  double fs,
  double omega0,
  double alpha,
  double f_limit,
  ChirpType type,
  double phi0=0.0) const
{
  vector<complex<T>> c=GenerateChirp(N,fs,omega0,alpha,f_limit,type,phi0);
  vector<T> y(N);
  for(size_t i=0;i<N;++i)
    y[i]=c[i].real();
  return y;
}
// ---------- MakeChirp: user-friendly dispatcher (f0,f1,duration ? omega0/alpha) ---------- //
// This is the high-level entry point for chirp synthesis. You tell us where to start (f0 in Hz),
// where to end (f1 in Hz), and over what time (duration in seconds). We compute the model-appropriate
// angular parameters (omega0,alpha) and then delegate to GenerateChirp(...) which integrates angular
// velocity to phase while honoring the requested audio-band fence (f_limit).
//
// Important modeling notes (how we map (f0,f1,T) to angular params for each chirp type):
// *Linear:        ?(t)=f0+a*t. This gives constant angular acceleration a (rad/s^2).
// *Exponential:   ?(t)=f0*exp(k*t). We choose k=a/max(f0,e) so that d?/dt|t=0?a.
// *Hyperbolic:    ?(t)=K/(t+c). We choose c=Clamp(-f0/a,1.0e-9,1.0e9) so that ?(0)?f0 and d?/dt|0?a.
// In all cases, before integrating ? into f, we clamp ? to ±?_lim where ?_lim=2p*f_limit to respect the
// requested audio limit. This means "up to a specified audio frequency" is always honored in the sweep.
//
// Parameters:
//   N         : number of samples to synthesize (length of the chirp buffer)
//   fs        : sample rate in Hz
//   f0,f1     : start/end frequencies in Hz (we clamp to ±f_limit to keep things safe)
//   duration  : intended sweep time in seconds; if <=0 we infer from N and fs to keep behavior intuitive.
//   f_limit   : absolute audio fence in Hz (e.g., 20000)
//   type      : ChirpType::Linear / Exponential / Hyperbolic
//   phi0      : initial phase in radians
//
// Returns: complex unit-magnitude chirp of length N.
//
inline vector<complex<T>> MakeChirp (
  size_t N,
  double fs,
  double f0,
  double f1,
  double duration,
  double f_limit,
  ChirpType type,
  double phi0=0.0) const
{
  vector<complex<T>> y(N,complex<T>(0,0));
  if(N==0||fs<=0)
    return y;

  // Resolve sweep time T. If caller passes non-positive, infer from N and fs to keep behavior intuitive.
  double Td=duration>0.0?duration:static_cast<double>(N)/fs;
  if (Td<=0.0)
    Td=static_cast<double>(N)/fs; // last guard

  // Enforce the audio fence early (endpoint clamp) so our angular mapping is already bounded.
  double f0c=Clamp(f0,-f_limit,f_limit);
  double f1c=Clamp(f1,-f_limit,f_limit);

  // Map to angular endpoints f0, ?1. We work in radians/sec throughout.
  const double two_pi=2.0*M_PI;
  double omega0=two_pi*f0c;
  double omega1=two_pi*f1c;

  // Compute model-specific "alpha" parameter that our GenerateChirp expects.
  // See big comment above: Linear uses a directly; Exponential uses a=f0*k; Hyperbolic uses a=-f0/c.
  double alpha=0.0;
  const double eps=1.0e-12;

  switch(type)
  {
    case ChirpType::Linear:
    {
      // w(T)=f0+a*T=w1 ? a=(w1-f0)/T
     alpha=(omega1-omega0)/Td;
    }
    break;
    case ChirpType::Exponential:
    {
      // w(t)=f0*exp(k*t), target |w1|=|f0|*exp(k*T) ? k=ln(|w1|/|f0|)/T.
      // Initial slope is dw/dt|0=k*f0 ? a=f0*k feeds our generator's exponential branch.
      double w0mag=std::max(std::abs(omega0),eps);
      double w1mag=std::max(std::abs(omega1),eps);
      double k=std::log(w1mag/w0mag)/std::max(Td,eps);
      // Preserve sign of omega0 in the slope naturally via a=f0*k.
      alpha=omega0*k;
    }
    break;

    case ChirpType::Hyperbolic:
    {
      // w(t)=K/(t+c), with w(0)=f0 ? K=f0*c. Also require w(T)=w1 ? c=T*w1/(f0-w1).
      // Guard degenerate cases (equal endpoints, near-zero denominators).
      double c;
      double denom=omega0-omega1;
      if(std::abs(denom)<eps)
        c=1.0e6;
      else
      {
        c=(Td*omega1)/denom;
        if(c<1.0e-9)c=1.0e-9;
      }
      alpha=-omega0/c; // so GenerateChirp can reconstruct c?-f0/a internally
    }
    break;

    default:
    {
     alpha=(omega1-omega0)/Td; // fall back to linear behavior
    }
    break;
  }
  // Delegate to the core generator which integrates ? and clamps per-sample to ±2p*f_limit.
  return GenerateChirp(N,fs,omega0,alpha,f_limit,type,phi0);
}

// ---------- MakeRealChirp: cosine-only convenience wrapper for DAC-friendly output ---------- //
// Same dispatcher as above, but we return a strictly real sweep using cos(f). Handy when you want
// an audio buffer you can ship straight to playback without dealing with I/Q pairs.
inline vector<T> MakeRealChirp (
  size_t N,
  double fs,
  double f0,
  double f1,
  double duration,
  double f_limit,
  ChirpType type,
  double phi0=0.0) const
{
  auto [omega0,alpha]=ChirpParamsFromF(f0,f1,duration,f_limit,type);
  vector<complex<T>> c=GenerateChirp(N,fs,omega0,alpha,f_limit,type,phi0);
  vector<T> y(N);
  for(size_t i=0;i<N;++i)
    y[i]=c[i].real();
  return y;
}

// ---------- ApplyChirpynessFromF: multiply an input by a dispatcher-driven chirp ---------- //
// This is a friendly "f0,f1,T" front end for ApplyChirpyness. We synthesize the matching chirp and
// modulate the caller-provided buffer so it "rides" the requested sweep inside the audio fence.
inline vector<complex<T>> ApplyChirpynessFromF (
  const vector<complex<T>>& x,
  double fs,
  double f0,
  double f1,
  double duration,
  double f_limit,
  ChirpType type,
  double phi0=0.0) const
{
  vector<complex<T>> c=MakeChirp(x.size(),fs,f0,f1,duration,f_limit,type,phi0);
  vector<complex<T>> y(x.size());
  for(size_t n=0;n<x.size();++n)
    y[n]=x[n]*c[n];
  return y;
}
inline vector<complex<T>> ApplyChirpynessFromF(
  const vector<T>& x,
  double fs,
  double f0,
  double f1,
  double duration,
  double f_limit,
  ChirpType type,
  double phi0=0.0) const
{
  vector<complex<T>> c=MakeChirp(x.size(),fs,f0,f1,duration,f_limit,type,phi0);
  vector<complex<T>> y(x.size());
  for(size_t n=0;n<x.size();++n)
    y[n]=complex<T>(x[n],0)*c[n];
  return y;
}
// ---------- ChirpParamsFromF: compute (omega0,alpha) for logging/inspection/UI ---------- //
// Friendly wrapper: maps (f0,f1,duration) to angular units exactly the way MakeChirp() does,
// but without generating a buffer. Useful for displaying slopes, debugging parameterization,
// or storing presets.
//
// Returns: pair<omega0,alpha> in radians/sec and radians/sec²-equivalent (depending on model).
//  *Linear: omega(t)=omega0+a*t, a has units rad/s²
//  *Exponential: omega(t)=omega0*exp(k*t), we return a=f0*k so the slope at t=0
//  *Hyperbolic: omega(t)=K/(t+c), we return a=-f0/c which lets GenerateChirp reconstruct c
//
inline std::pair<double,double> ChirpParamsFromF (
  double f0,
  double f1,
  double duration,
  double f_limit,
  ChirpType type) const
{
  // Enforce audio fence
  double f0c=Clamp(f0,-f_limit,f_limit);
  double f1c=Clamp(f1,-f_limit,f_limit);

  const double two_pi=2.0*M_PI;
  double omega0=two_pi*f0c;
  double omega1=two_pi*f1c;

  // Resolve duration T
  double Td=duration>0.0?duration:1.0; // if duration<=0, fallback to 1s nominal for param
  const double eps=1.0e-12;

  double alpha=0.0;
  switch(type)
  {
    case ChirpType::Linear:
      alpha=(omega1-omega0)/Td;
      break;
    case ChirpType::Exponential:
    {
      double w0mag=std::max(std::abs(omega0),eps);
      double w1mag=std::max(std::abs(omega1),eps);
      double k=std::log(w1mag/w0mag)/std::max(Td,eps);
      alpha=omega0*k;
    } 
    break;
    case ChirpType::Hyperbolic:
    {
      double denom=omega0-omega1;
      double c;
      if(std::abs(denom)<eps) c=1.0e6;
      else
      {
        c=(Td*omega1)/denom;
        if(c<1.0e-9)c=1.0e-9;
      }
      alpha=-omega0/c; // so GenerateChirp can reconstruct c?-f0/a internally
    }
    break;
    default:
      alpha=(omega1-omega0)/Td;break;
  }
  return {omega0,alpha};
}
private:
    vector<T> signal;                   // The signal to process.
    vector<T> subCarrier;               // 
    WindowType window;                  // The window to apply to the signal.
    int windowSize=0;                   // The size of the window.
    float overlap;                      // The overlap factor.
    mutable vector<complex<T>> twiddles;// Precomputed twiddles factors (mutable for const caching).
    double sRate;                       // The rate at which we sampled the RF.
    int length;                         // The length of the signal.
};

// Approximate Maximum Likelihood Fundamental Frequency Estimator
 template<typename T>
 static inline std::optional<T> FreqMLEReal (
   const std::vector<T>& s,             // The input signal.
   const T fStart,                      // The start frequency.
   const T fStop,                       // The stop frequency.
   const T fs)                          // The number of samples per second.
 {                                      // ~~~~~~~~~ FreqMLEReal ~~~~~~~~~~~~~~~~~~ //
  static_assert(std::is_floating_point<T>::value, "T must be a floating point type.");
  // Validate inputs
  if (!(fs> T(0)))                      // 
    return std::nullopt;                // Invalid sample rate
  if (!(fStart>=T(0)&&fStop>=T(0)))     // Are we in our defined band?
    return std::nullopt;                // Negative band
  if (fStart>=fStop)                    // Inverted limits?
    return std::nullopt;                // Yes, return nothing.
  if (s.size()<4)                       // Enough points for Maximum Likelihood Estimaion?
    return std::nullopt;                // No, return nothing.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // 1. Calculate the FFT of the input signal.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  SpectralOps<T> engine;                // Use SpectralOps for FFT utilities.
  vector<complex<T>> X=engine.FFTStride(s);    // Get the FFT of the signal.
  const std::size_t N=X.size();         // Get the length of the FFT.
  if (N<4)                              // FFT too short?
    return std::nullopt;                // Yes, return nothing.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // 2. Translate from Start Frequency and Stop Frequency to bin 
  // indices in the FFT.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  const T bins=fs/static_cast<T>(N);    // Hz per FFT bin
  size_t kilo=std::clamp<size_t>(static_cast<size_t>(std::floor(fStart/bins)),1,N/2-1);
  size_t kimax=std::clamp<size_t>(static_cast<size_t>(std::ceil (fStop /bins)), 1, N/2-1);
  if (kilo>kimax)                       // Invalid bin range?
    return std::nullopt;                // Yes, return nothing.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // 3. Find Peak magnitude square in the search band
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  auto itmax=std::max_element(X.begin()+static_cast<std::ptrdiff_t>(kilo),X.begin()+static_cast<std::ptrdiff_t>(kimax)+1,
    [](const std::complex<T>& a, const std::complex<T>& b){return std::norm(a)<std::norm(b);});// peak in band
  if (itmax==X.end())                   // Should never happen since we checked the range, but guard just in case.
    return std::nullopt;                //
  size_t k=static_cast<size_t>(std::distance(X.begin(),itmax)); // index of peak
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // 4. 3-point parabolic interpolation (f0+-1 bins)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  if (k==0||k>=N/2||k<=kilo||k>=kimax)  // avoid edges of DC/Nyquist and search edges
    return std::nullopt;                // Peak too close to edges for interpolation, return nothing.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Get the power of the peak and its neighbours.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Floor magnitude to avoid log(0) -> -inf
  const T eps=static_cast<T>(1e-30);    // Avoid log(0) by flooring the magnitudes to a tiny value. This prevents -inf and NaN in the interpolation step.
  const T mL=std::max<T>(std::norm(X[k-1]),eps);// Left neighbor magnitude squared, floored to eps.
  const T mC=std::max<T>(std::norm(X[k]),eps); // Center (peak) magnitude squared, floored to eps.
  const T mR=std::max<T>(std::norm(X[k+1]),eps);// Right neighbor magnitude squared, floored to eps.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Compute the shape (alpha), Rate (beta), Normalized Constant (gamma)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  T alpha=std::log(mL);                 // The shape
  T beta=std::log(mC);                  // The rate
  T gamma=std::log(mR);                 // Normalized constant
  if (!std::isfinite(alpha)||!std::isfinite(beta)||!std::isfinite(gamma)) // Is it inf?
     return std::nullopt;               // Return nothing
  T denom=(alpha-T(2)*beta+gamma);      // Interpolation denominator
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Clealculate the descent step based on the parabolic interpolation.
  // If the denominator is zero, we cannot interpolate.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  T delta=T(0);
  if (std::isfinite(denom) && std::abs(denom) > static_cast<T>(1e-20))
    delta=static_cast<T>(0.5)*(alpha-gamma)/denom; // (-0.5?0.5) ideally
  if (!std::isfinite(delta))
    delta=T(0);
  // Clamp to a sane range
  if (delta < T(-0.5))
    delta=T(-0.5);
  if (delta > T( 0.5))
    delta=T( 0.5);
  // ~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // 5. Refine the frequency estimate.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~ //
    T fhat=(static_cast<T>(k)+delta)*bins;// Refine the frequency estimate.
    if (!std::isfinite(fhat))
      return std::nullopt;
    if (fhat<fStart||fhat>fStop)
      return std::nullopt; // Out of bounds
    return fhat;                       // Return the estimated frequency.
 }                                     // ---------- FreqMLEReal ----------------- //
 // YIN Pitch Estimator
 template<typename T>
 static inline T PitchYIN (
  const std::vector<T>& s, // The input signal.
  const T fs,         // The sample rate of the signal.
  const size_t bs,    // The block size of the signal.
  T thresh=static_cast<T>(0.15))         // Tolerance for approx output
 {                                      //%PitchYIN
  static_assert(is_floating_point<T>::value, "T must be a floating point type.");
  // Use actual signal size; cap bs in case caller passed larger than s.size().
  const size_t N=std::min(bs, s.size());
  if(N < 8) return T(0);
  const size_t taumax=N/2;        // conventional YIN search limit
  if(taumax < 4) return T(0);
  std::vector<T> diff(taumax, T(0));
  std::vector<T> cum (taumax, T(0));
    // ~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // 1.Difference function d(tau)=SUM[n=0 to N-1-tau] {x[n]-x[n+tau]}^2
    // ~~~~~~~~~~~~~~~~~~~~~~~~~ //
    for (size_t tau=1; tau<taumax; ++tau) 
    {
      T acc=0;
      const size_t limit=N-tau; // ensure n+tau < N
      for(size_t n=0; n<limit; ++n){
        const T dif=s[n]-s[n+tau];
        acc += dif*dif;
      }
      diff[tau]=acc;
    }
    // ~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // 2.Cumulative Running sum c(tau): c(tau)=SUM[tau=1 to taumax] d(tau)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~ //
    cum[0]=1;                           // Initialize the cumulative sum.
    T rsum=0;                           // The running sum variable.
    for (size_t tau=1;tau<taumax;++tau) // For each time step...
    {
      rsum+=diff[tau];                  // Add the difference to running sum.
      cum[tau]=diff[tau]*tau/(rsum+std::numeric_limits<T>::epsilon());// Normalize the cumulative sum.
    }                                   // Done with running sum.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // 3. Absolute thresholding: if c(tau) < thresh, then tau is a pitch candidate.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~ //
    size_t taue=0;                      // Our estimation variable.
    for (size_t tau=2;tau<taumax;++tau) // For each bin...
    {                                   // Threshold and rough estimate.
      if (cum[tau]<thresh)              // Is this sample less than our wanted power?
      {                                 // Yes, proceed to estimate
        // Get the parabolic minimum    //
        while (tau+1<taumax&&cum[tau+1]<cum[tau]) ++tau;
        taue=tau;                       // Store the estimated pitch.
        break;                          // Break out of the loop.
      }
    }
    // ~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // 4. Parabolic refinement:
    // ~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // If no threshold crossing found, pick argmin of cum (excluding boundaries)
    if(taue==0)                        // pick global minimum inside safe range
    { 
      T best=std::numeric_limits<T>::max();
      for(size_t k=2;k+2<taumax;++k)
      {
        if(cum[k]<best)
        {
          best=cum[k];
          taue=k;
        }
      }
    }
  // Ensure taue landed in a valid interior region; otherwise clamp.
    if(taue < 2) taue=2;            // clamp low
    if(taue+1 >= taumax) taue=(taumax>3?taumax-2:2); // clamp high keeping room for +1
    if(taumax < 4 || taue >= taumax-1) return T(0); // still unsafe; bail
    const T y0=cum[taue-1];           // safe: taue>=2
    const T y1=cum[taue];
    const T y2=cum[taue+1];           // safe: taue+1<taumax
    const T denom=y0+y2-2*y1;           // Denom for parabolic interpolation.
    T tip=static_cast<T>(taue);         // Initialize interpolation var
    // ~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // If the denominator is not zero, we can interpolate.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~ //
    if (std::fabs(denom)>std::numeric_limits<T>::epsilon())
      tip+=(y0-y1)/denom;               // Appromiate please.
    return fs/tip;                      // Return the estimated pitch frequency.
  }                                     // ---------- PitchYIN ----------------- //
//============================================================================
// Lightweight STFT / ISTFT and SpectralFreeze helpers (appended)
//============================================================================
namespace sig::spectral {

// Short-Time Fourier Transform (returns frequency-domain frames)
template<typename T>
inline std::vector<std::vector<std::complex<T>>> STFT (
    const std::vector<std::complex<T>>& x,
    const typename Window<T>::WindowType& wType,
    int winSize,
    float overlapPc)
{
  std::vector<std::vector<std::complex<T>>> frames;
  if (winSize <= 0 || x.empty())
    return frames;
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Compute hop size
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  int hop=std::max(1, (int)std::round(winSize*(1.f-overlapPc/100.f)));
  hop=std::min(hop, winSize);
  Window<T> W;
  // Select desired window
  W.SetWindowType(wType, (size_t)winSize);
  const auto& w=W.GetData();
  SpectralOps<T> engine;
  // Perform the STFT
  for (int pos=0; pos+winSize <= (int)x.size(); pos += hop) 
  {
    std::vector<std::complex<T>> frame(winSize);
    for (int n=0;n<winSize;++n) 
    {
      const auto& s=x[pos+n];
      frame[n]={ s.real()* (T)w[n], s.imag()* (T)w[n] };
    }
    auto X=engine.FFTStride(frame);
    frames.push_back(std::move(X));
  }
  return frames;
}

// Inverse STFT (reconstruct time-domain complex signal)
template<typename T>
inline std::vector<std::complex<T>> ISTFT (
  const std::vector<std::vector<std::complex<T>>>& frames,
  const typename Window<T>::WindowType& wType,
  int winSize,
  float overlapPc)
{
  if (frames.empty() || winSize<=0)
    return {};
  // Compute hop size
  int hop=std::max(1, (int)std::round(winSize*(1.f-overlapPc/100.f)));
  hop=std::min(hop, winSize);
  // Set the desired window
  Window<T> W; W.SetWindowType(wType, (size_t)winSize);
  const auto& w=W.GetData();
  SpectralOps<T> engine;
  size_t outLen=(frames.size()-1)*hop+winSize;
  std::vector<std::complex<T>> y(outLen, std::complex<T>(0,0));
  std::vector<T> weight(outLen, (T)0);
  size_t fi=0;
  // Perform ISTFT
  for (const auto& X: frames) 
  {
    auto t=engine.IFFTStride(X);
    if ((int)t.size() < winSize)
      t.resize(winSize, std::complex<T>(0,0));
    size_t base=fi*hop;
    for (int n=0;n<winSize;++n) 
    {
      T winS=(T)w[n];
      y[base+n] += t[n]*winS;
      weight[base+n] += winS*winS;
    }
    ++fi;
  }
  for (size_t i=0;i<y.size();++i)
    if (weight[i]>(T)0)
      y[i]/=weight[i];
  return y;
}

// Spectral Freeze helper
template<typename T>
inline std::vector<std::complex<T>> SpectralFreeze (
  const std::vector<std::complex<T>>& x,
  const typename Window<T>::WindowType& w,
  int wSiz,
  float ovlap,
  int freeze_frame=-1,
  T mix=static_cast<T>(1))
{
  auto X=STFT<T>(x,w,wSiz,ovlap);
  if (X.empty())
    return {};
  int fz=freeze_frame;
  if (fz < 0) 
  {
    T best=(T)-1; fz=0;
    for (int i=0;i<(int)X.size();++i)
    {
      T e=0; for (auto& z: X[i]) e += (T)std::norm(z);
      if (e>best){best=e;fz=i;}
    }
  }
  fz=std::clamp<int>(fz,0,(int)X.size()-1);
  const auto& Xfz=X[fz];
  std::vector<T> mag_fz(Xfz.size());
  for (size_t k=0;k<Xfz.size();++k)
    mag_fz[k]=(T)std::abs(Xfz[k]);
  for (auto& frame: X)
  {
    for (size_t k=0;k<frame.size();++k)
    {
      T ph=(T)std::arg(frame[k]);
      T m=(T)std::abs(frame[k]);
      T newm=m*(1-mix)+mag_fz[k]*mix;
      frame[k]=std::polar(newm, ph);
    }
  }
  return ISTFT<T>(X,w,wSiz,ovlap);
}

} // namespace sig::spectral

//============================================================================
// Additional phase-vocoder style helpers for front-end FX (lightweight refs)
//============================================================================
namespace sig::spectral {

// Simple single-frame cross-morph: combines magnitudes, selects phase from A or B.
template<typename T>
inline std::vector<std::complex<T>> SpectralMorphCross (
  const std::vector<std::complex<T>>& a,
  const std::vector<std::complex<T>>& b,
  const typename Window<T>::WindowType& wType,
  int fftSize,
  float /*overlapPc*/,
  T mix,
  int phaseFrom)
{
  const int N=std::min<int>({fftSize,(int)a.size(),(int)b.size()});
  if (N<=0)
    return {};
  Window<T> W; W.SetWindowType(wType, (size_t)N);
  SpectralOps<T> engine;
  // Time -> freq
  std::vector<std::complex<T>> ta(N), tb(N);
  for (int i=0;i<N;++i)
  {
    T win=(T)W.GetData()[i];
    ta[i]=a[i]*win;
    tb[i]=b[i]*win;
  }
  auto FA=engine.FFTStride(ta);
  auto FB=engine.FFTStride(tb);
  const size_t M=std::min(FA.size(), FB.size());
  for (size_t k=0;k<M;++k)
  {
    T magA=(T)std::abs(FA[k]);
    T magB=(T)std::abs(FB[k]);
    T mag=magA*(1-mix)+magB*mix;
    T ph =(phaseFrom<=0? (T)std::arg(FA[k]) : (T)std::arg(FB[k]));
    FA[k]=std::polar(mag, ph);
  }
  auto y=engine.IFFTStride(FA);
  // remove analysis window energy (simple window^2 comp)
  for (int i=0;i<N && i<(int)W.GetData().size(); ++i)
  {
    T win=(T)W.GetData()[i];
    if (win != (T)0)
      y[i] /= win; // rough de-window
  }
  return y;
}

// Naive pitch shift: resample magnitude envelope in frequency domain.
template<typename T>
inline std::vector<std::complex<T>> PitchShiftPhaseVocoder (
  const std::vector<std::complex<T>>& x,
  const typename Window<T>::WindowType& wType,
  int fftSize,
  float /*overlapPc*/,
  T pitchRatio)
{
  const int N=std::min<int>(fftSize,(int)x.size());
  if (N<=0)
    return {};
  Window<T> W; W.SetWindowType(wType,(size_t)N);
  SpectralOps<T> engine;
  std::vector<std::complex<T>> tmp(N);
  for (int i=0;i<N;++i)
  {
    T win=(T)W.GetData()[i];
    tmp[i]=x[i]*win;
  }
  auto F=engine.FFTStride(tmp);
  std::vector<std::complex<T>> G(F.size(), std::complex<T>(0,0));
  const size_t M=F.size();
  for (size_t k=0;k<M;++k)
  {
    double src=k / std::max<double>(pitchRatio,1e-6);
    size_t k0=(size_t)std::floor(src);
    size_t k1=std::min(M-1, k0+1);
    double a=src-k0;
    std::complex<T> v0=(k0<M?F[k0]:std::complex<T>{});
    std::complex<T> v1=(k1<M?F[k1]:std::complex<T>{});
    G[k]=v0*(T)(1.0-a)+v1*(T)a;
  }
  auto y=engine.IFFTStride(G);
  for (int i=0;i<N && i<(int)W.GetData().size(); ++i)
  {
    T win=(T)W.GetData()[i];
    if(win!=(T)0)
      y[i]/=win;
  }
  return y;
}

// Naive time stretch (single-frame placeholder): blends linear resample in time domain.
template<typename T>
inline std::vector<std::complex<T>> TimeStretchPhaseVocoder (
  const std::vector<std::complex<T>>& x,
  const typename Window<T>::WindowType& wType,
  int fftSize,
  float /*overlapPc*/,
  T stretch)
{
  const int N=std::min<int>(fftSize,(int)x.size());
  if (N<=0)
    return {};
  if (std::fabs(stretch-(T)1) < (T)1e-6)
    return x;                           // identity
  int target=(int)std::max<T>(16, std::round(N*stretch));
  std::vector<std::complex<T>> y(target);
  for (int i=0;i<target;++i)
  {
    double src=i / stretch;
    int i0=(int)std::floor(src);
    int i1=std::min(N-1, i0+1);
    double a=src-i0;
    auto v0=x[i0];
    auto v1=x[i1];
    y[i]=v0*(T)(1.0-a)+v1*(T)a;
  }
  // Optional: apply window normalization (skip for placeholder)
  return y;
}

} // namespace sig::spectral
