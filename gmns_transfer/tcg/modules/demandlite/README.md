# 04_demandlite_cpp — C++ auto-calibration kernel

Goal: fused OpenMP forward+backward of the demand layers for production
cadence (PERFORMANCE.md P2), IF GPU/torch.compile numbers (05_gpu_training)
leave a CPU niche. API contract in demandlite.h. Pairs with the kernel-side
sparse-pi/select-link export (TAPLite.cpp; the TAPLITE_ROUTE_VOL_MIN env
patch was step 0 — writing pi directly as binary instead of 1.4GB route CSV
is step 1 and benefits ALL users, so it proceeds regardless of GPU results).
