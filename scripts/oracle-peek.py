import json, sys
import numpy as np
R = [json.loads(l) for l in open(sys.argv[1])]
names = ("coarse", "tight", "touch", "loose", "rand_k5", "abs_k5", "label_k5", "rand_k24", "abs_k24", "label_k24", "full2x")
print("window group      " + " ".join(f"{n:>9s}" for n in names) + "   effect range")
for r in R[:12]:
    e = np.array(r["effect"])
    print(f"{r['group']:8s} y={r['label']}    " + " ".join(f"{r['margin'][n]:+9.2f}" for n in names) + f"   {e.min():+.2f}..{e.max():+.2f}")
print("tokens:", {n: int(np.mean([r["tokens"][n] for r in R])) for n in names})
