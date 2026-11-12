// Copyright 2019 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ABSL_PROFILING_INTERNAL_EXPONENTIAL_BIASED_H_
#define ABSL_PROFILING_INTERNAL_EXPONENTIAL_BIASED_H_

#include <stdint.h>

#include "absl/base/config.h"
#include "absl/base/macros.h"
#include "absl/base/optimization.h"

namespace absl {
ABSL_NAMESPACE_BEGIN
namespace profiling_internal {

// Per-thread state for stride-based periodic sampling with unbiased unsampling
// weights.
//
// `SamplingState` is a trivial, zero-initializable POD so callers can place it
// in `ABSL_PER_THREAD_TLS_KEYWORD` (`__thread`) or `ABSL_CONST_INIT
// thread_local` variables inside hot headers without dynamic TLS initialization
// guards.
//
// Why `sample_period` stores the configured period rather than the random
// `GetStride()` draw:
//   1. Because the geometric/exponential countdown is memoryless, every item
//      has the same `1 / period` inclusion probability regardless of the random
//      gap that preceded it. Weighting a sampled item by the realized random
//      draw doubles per-callsite estimator variance and produces ~18x
//      single-sample weight outliers.
//   2. On a newly started thread, any random stride finishing at the t-th item
//      satisfies `stride <= t` (for example, if item #1 is sampled, its stride
//      was necessarily 1), systematically undercounting items created early in
//      a thread's lifetime (`1 - (1 - 1/period)^t` in expectation instead of
//      1).
//   3. Snapshotting `sample_period` when `next_sample` is drawn preserves
//      unbiased weights across runtime changes to the configured period.
struct SamplingState {
  // Remaining distance (events or bytes) until the next sample. Zero on a
  // newly initialized thread so the first fast-path decrement enters
  // `ConfirmSampleSlow()` to draw an initial stride.
  int64_t next_sample = 0;

  // Sampling period active when `next_sample` was drawn, or 0 when
  // uninitialized or after a disabled check.
  int64_t sample_period = 0;
};

// ExponentialBiased provides a small and fast random number generator for a
// memoryless geometric distribution (the discrete analog of an exponential
// distribution). This generator manages very little state, and imposes no
// synchronization overhead. This makes it useful in specialized scenarios
// requiring minimum overhead, such as stride-based periodic sampling.
//
// ExponentialBiased provides two closely related functions, GetSkipCount() and
// GetStride(), returning the number of events to skip (`>= 0`) or the number
// of events required (`>= 1`) before an event with a given mean occurs.
//
// The distribution is useful to generate a random wait time or some periodic
// event with a given mean probability. For example, if an action is supposed to
// happen on average once every 'N' events, then we can get a random 'stride'
// counting down how long before the event to happen. For example, if we'd want
// to sample one in every 1000 'Frobber' calls, our code could look like this:
//
//   Frobber::Frobber() {
//     stride_ = exponential_biased_.GetStride(1000);
//   }
//
//   void Frobber::Frob(int arg) {
//     if (--stride_ == 0) {
//       SampleFrob(arg);
//       stride_ = exponential_biased_.GetStride(1000);
//     }
//     ...
//   }
//
// This class is thread-compatible.
class ExponentialBiased {
 public:
  // The number of bits set by NextRandom.
  static constexpr int kPrngNumBits = 48;

  // `GetSkipCount()` returns the number of events to skip before some chosen
  // event happens. For example, randomly tossing a coin, we will on average
  // throw heads once before we get tails. We can simulate random coin tosses
  // using GetSkipCount() as:
  //
  //   ExponentialBiased eb;
  //   for (...) {
  //     int number_of_heads_before_tail = eb.GetSkipCount(1);
  //     for (int flips = 0; flips < number_of_heads_before_tail; ++flips) {
  //       printf("head...");
  //     }
  //     printf("tail\n");
  //   }
  //
  int64_t GetSkipCount(int64_t mean);

  // GetStride() returns the number of events required for a specific event to
  // happen. See the class comments for a usage example. `GetStride()` is
  // equivalent to `GetSkipCount(mean - 1) + 1`. When to use `GetStride()` or
  // `GetSkipCount()` depends mostly on what best fits the use case.
  int64_t GetStride(int64_t mean);

  // Slow-path companion to `ShouldSampleNext()`. Must be called whenever
  // `ShouldSampleNext(state, step)` returns true (or directly in tests with
  // `state.next_sample <= 0`).
  //
  // Returns 0 if the current event should NOT be sampled, or a positive
  // unsampling weight if it SHOULD be sampled:
  //   * Unit-step item sampling (`step == 1`, e.g. hashtablez, cordz): returns
  //     `sample_period` (the period active when the completed stride started),
  //     including when the very first item on a thread is sampled.
  //   * Weighted/byte sampling (`step == size + 1`, e.g. heap profilers):
  //     returns `sample_period - state.next_sample` (`sample_period` plus any
  //     within-item overshoot past zero), matching `size / (1 - exp(-size /
  //     sample_period))` in expectation without computing `exp()`.
  //
  // If `period <= 0`, sampling is treated as disabled: `state` is set to skip
  // `disabled_stride` units before re-checking `period`, and 0 is returned.
  int64_t ConfirmSampleSlow(SamplingState& state, int64_t period,
                            int64_t step = 1,
                            int64_t disabled_stride = 1 << 16);

  // Computes a random number in the range [0, 1<<(kPrngNumBits+1) - 1]
  //
  // This is public to enable testing.
  static uint64_t NextRandom(uint64_t rnd);

 private:
  void Initialize();

  uint64_t rng_{0};
  bool initialized_{false};
};

// Returns the next prng value.
// pRNG is: aX+b mod c with a = 0x5DEECE66D, b =  0xB, c = 1<<48
// This is the lrand64 generator.
inline uint64_t ExponentialBiased::NextRandom(uint64_t rnd) {
  const uint64_t prng_mult = uint64_t{0x5DEECE66D};
  const uint64_t prng_add = 0xB;
  const uint64_t prng_mod_power = 48;
  const uint64_t prng_mod_mask =
      ~((~static_cast<uint64_t>(0)) << prng_mod_power);
  return (prng_mult * rnd + prng_add) & prng_mod_mask;
}

// Fast-path decrement for always-on samplers (`hashtablez`, `cordz`, heap
// samplers). Touches only `state` in TLS without loading the global period on
// the fast path. When this returns true, the caller must invoke
// `ExponentialBiased::ConfirmSampleSlow()`.
inline bool ShouldSampleNext(SamplingState& state, int64_t step = 1) {
  state.next_sample -= step;
  return ABSL_PREDICT_FALSE(state.next_sample <= 0);
}

// Combined fast/slow helper for windowed on-demand samplers whose global
// `period` is 0 when idle and may change dynamically between collection
// windows. Resets any pending stride immediately when `period` changes, and
// returns 0 (skip) or the positive unsampling weight (`period`).
inline int64_t SampleWithLivePeriod(SamplingState& state, int64_t period,
                                    ExponentialBiased& rng) {
  if (ABSL_PREDICT_TRUE(period <= 0)) return 0;
  if (ABSL_PREDICT_FALSE(state.sample_period != period)) {
    state = {0, 0};
  }
  if (ABSL_PREDICT_TRUE(!ShouldSampleNext(state))) return 0;
  return rng.ConfirmSampleSlow(state, period);
}

}  // namespace profiling_internal
ABSL_NAMESPACE_END
}  // namespace absl

#endif  // ABSL_PROFILING_INTERNAL_EXPONENTIAL_BIASED_H_
