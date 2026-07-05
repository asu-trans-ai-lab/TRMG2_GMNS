# 02_lora_adapters — low-rank utility adapters

Idea (NN_EXPORT_DESIGN sec 3.3): U_p <- U_p + a_p b_p^T, r<=4, frozen TRMG2
coefficients as pretrained weights, Frobenius trust region. Generalizes the
cluster dASC knobs (r=1 with b = cluster indicator). Deliverable: %RMSE vs
drift curve over r in {1,2,4} on Chicago sketch, then TRMG2; gauge discipline
(column-centering, pinning) per TCG_MATH sec 5. Also here: superzone
encoder-decoder as frozen bottleneck layer (bridges the od-compression work).
