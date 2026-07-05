# params/ — TRMG2 binary parameters converted to CSV (pipeline-usable)

Converted from TransCAD .bin to plain CSV (0-mismatch round-trip) so the demand
build needs NO TransCAD binaries.

| file | source | rows x cols | used by |
|---|---|---|---|
| shadow_prices.csv | resident/dc/shadow_prices.bin | 3247 x 2 (TAZ, hbw) | HBW destination-choice double constraint |
| init_cong_time_2020.csv | networks/init_cong_time_2020.bin | 40274 x 9 (ID + AB/BA times x AM/MD/PM/NT) | warm-start period skims for destination choice |

Loaded via `se_loader.load_table(csv_path, bin_fallback)`. Regenerate with
`python -m dtalite_qa.transcad_bin <file>.bin --out <file>.csv`.
