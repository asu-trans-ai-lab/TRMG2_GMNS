// demandlite.h — API contract for the C++ auto-calibration demand kernel.
// Fused forward+backward of the nested-DC demand layers (TCG_MATH eqs 1-9,
// adjoints 12-16). All arrays row-major, caller-allocated. OpenMP inside.
#pragma once
#include <cstdint>

extern "C" {

// Forward: theta -> OD (and caches for backward). Returns 0 on success.
// Z zones, C clusters, P purposes; member[Z] in [0,C); skims t[P?][Z*Z] shared.
int dl_forward(int Z, int C, int P,
               const int32_t* member,          // [Z]
               const double* theta_c,          // [C] nest scales (frozen)
               const double* U_frozen,         // [P*Z*Z] frozen zone utilities
               const double* pop_rate,         // [P*Z] pop_i * rate_p
               const double* g,                // [P]
               const double* dASC,             // [P*C]
               const double* dIC,              // [P*C]
               double* T_out,                  // [P*Z*Z] trips
               double* cache);                 // [P*Z*(C+1)] Pc + rowsums

// Backward: grad wrt T -> grads wrt (g, dASC, dIC). Uses cache from forward.
int dl_backward(int Z, int C, int P,
                const int32_t* member,
                const double* T,                // [P*Z*Z]
                const double* cache,
                const double* G_T,              // [P*Z*Z] upstream adjoint
                const double* pop_rate,
                const double* g,
                double* g_g,                    // [P]
                double* g_dASC,                 // [P*C]
                double* g_dIC);                 // [P*C]

}  // extern "C"
