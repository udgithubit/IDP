"""Summarise the pieces tables written by tools/run_pipeline.py --out into one table, a volume calibration
and the tumour trend.

    python summarize_results.py PIECES_DIR [OUT_CSV]

Known-volume recordings are recognised by their leading number (1 .. 9, see Info.on.measurements.odt),
tumour recordings by names like high_2gr / high_2gr_resuc. Nothing is written into the repository.
"""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd

KNOWN = {1: ("HIGH", 4, 3), 2: ("HIGH", 4, 2), 3: ("HIGH", 2, 2), 4: ("HIGH", 1, 2), 5: ("HIGH", 1, 0),
         6: ("MIDDLE", 3, 0), 7: ("MIDDLE", 2, 1), 8: ("MIDDLE", 1, 0), 9: ("MIDDLE", 1, 0)}   # conc, ml, re-suctions

d = Path(sys.argv[1]); rows = []
for f in sorted(d.glob("*.pieces.csv")):
    p = pd.read_csv(f); name = f.name[:-len(".pieces.csv")]; dy = p[p.label == "dyed"] if len(p) else p
    r = {"recording": name, "pieces": len(p), "dyed": len(dy),
         "solid_no_dye": int((p.label == "solid_no_dye").sum()) if len(p) else 0,
         "water": int((p.label == "water").sum()) if len(p) else 0,
         "undecided": int((p.label == "undecided").sum()) if len(p) else 0,
         "dyed_area": float(dy.area.sum()) if len(dy) else 0.0}
    m = re.match(r"^(\d)[.\s]", name)
    if m and int(m.group(1)) in KNOWN:
        c, ml, rs = KNOWN[int(m.group(1))]; r.update(conc=c, ml=ml, resuction=rs, units=ml * (1 + rs))
    g = re.search(r"(\d)gr", name)
    if g and name.startswith("high_"): r.update(tumour_gr=int(g.group(1)), resuc="resuc" in name)
    rows.append(r)
t = pd.DataFrame(rows)
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 48)
print(t[["recording", "pieces", "dyed", "solid_no_dye", "water", "undecided", "dyed_area"]].round(0).to_string(index=False))

k = t[t.units.notna()] if "units" in t else t.iloc[0:0]
for conc in ("HIGH", "MIDDLE"):
    s = k[k.conc == conc]
    if len(s) < 3: continue
    r2 = np.corrcoef(s.units, s.dyed_area)[0, 1] ** 2
    print(f"\n{conc}: dyed area vs volume x (1 + re-suctions): R2 = {r2:.3f}  ({len(s)} recordings)")
    if conc == "HIGH":
        u, a = s.units.values.astype(float), s.dyed_area.values
        for i in range(len(u)):
            mk = np.arange(len(u)) != i; sl = (u[mk] * a[mk]).sum() / (u[mk] ** 2).sum()
            print(f"  leave-one-out: {u[i]:>4.0f} units -> predicted {a[i]/sl:5.1f} ({100*(a[i]/sl-u[i])/u[i]:+.0f}%)")
if "tumour_gr" in t:
    tt = t[t.tumour_gr.notna()]
    if len(tt):
        print("\nTumour recordings, mean dyed area by size:")
        print(tt.groupby(["resuc", "tumour_gr"])[["dyed", "dyed_area"]].mean().round(0).to_string())
if len(sys.argv) > 2:
    t.to_csv(sys.argv[2], index=False); print("\nsaved", sys.argv[2])
