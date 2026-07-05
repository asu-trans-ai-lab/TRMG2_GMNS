import os, sys, time, csv
sys.path.insert(0, '.')
os.environ["TAPLITE_ROUTE_VOL_MIN"] = "0.04"
EXE = "C:/source_codes/0_source_code_new/dtalite_with_taplite_Cpp_kernel/bin/DTALite.exe"

# 1. regional column/ADMM demo
from column_tools import run_demo
run_demo(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_chicago_regional"),
         "chicago_regional")

# 2. TRMG2 MD/PM/NT route reruns + operator extraction
import pytaplite
from matrix_ops import build_period, link_index
lidx = link_index()
for per in ("MD", "PM", "NT"):
    sdir = f"scenario_{per}"
    s = open(os.path.join(sdir, "settings.csv")).read()
    s = s.replace(",0,0,0,0\n", ",1,0,0,0\n") if ",1,0,0,0" not in s else s
    open(os.path.join(sdir, "settings.csv"), "w").write(s)
    t0 = time.time()
    pytaplite.assign(sdir, exe=EXE, prefer_inproc=False)
    print(f"[{per}] kernel {time.time()-t0:.0f}s", flush=True)
    build_period(per, lidx)
print("CHAIN DONE", flush=True)
