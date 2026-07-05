# se_data/ — socioeconomic data as CSV (GMNS-pipeline usable)

TRMG2's socioeconomic/land-use tables converted from TransCAD binary to plain
CSV (verified 0-mismatch round-trip). One row per TAZ, portable, no TransCAD.

| file | year | rows x cols | key fields |
|---|---|---|---|
| se_2020.csv | 2020 base | 3247 x 47 | HH, HH_POP, Median_Inc, Industry/Office/Retail/Service, PctHighPay, K12, College, Hospital, ParkCost, Enplanements, AWDT |
| se_2035.csv | 2035 adopted | 3247 x 47 | same schema |
| se_2045.csv | 2045 adopted | 3247 x 47 | same schema |
| se_2055.csv | 2055 adopted | 3247 x 55 | extended schema |

## Use in the pipeline
```python
from se_loader import load_se
se_rows = load_se(REPO, year=2020)   # list of dicts, numeric types coerced
```
`se_loader.py` reads the CSV (preferred) or falls back to the .bin. Regenerate
with: `python -m dtalite_qa.transcad_bin <sedata>/se_2020.bin --out se_2020.csv`.

Regional totals (2020): 812,747 households, 2,031,531 persons, 1,062,962 jobs.
