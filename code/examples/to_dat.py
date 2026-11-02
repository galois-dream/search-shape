# to_dat.py -- written on day 2 at 00:10. Turn ring_examples.py's JSON into pgfplots tables for the appendix
# figure: per query, the 50 neighbours' 2-D layout (classical MDS of the unit offsets; display only) with a sense
# class, all MST edges, and the longest edge. Usage: python to_dat.py ring_examples.json OUT_DIR "query 1" "query 2" ...
import json, sys
from collections import defaultdict
from pathlib import Path

d = json.load(open(sys.argv[1]))
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
Q = {q["query"]: q for q in d["queries"]}
for k, name in enumerate(sys.argv[3:]):
    q = Q[name]
    order = [s for s, _ in q["senses_in_top50"] if s != "other pages"]   # classes 0, 1, 2, ... by frequency
    cls = [order.index(s) if s in order else 9 for s in q["sense"]]        # 9 = other pages
    xy = q["xy"]
    i, j = q["longest_edge"]
    adj = defaultdict(list)                       # MST without its longest edge -> the two sides
    for a, b, _ in q["mst"]:
        if {a, b} != {i, j}:
            adj[a].append(b); adj[b].append(a)
    side, todo = {i}, [i]
    while todo:
        for n in adj[todo.pop()]:
            if n not in side:
                side.add(n); todo.append(n)
    small = side if len(side) <= len(xy) - len(side) else set(range(len(xy))) - side
    stray = small if len(small) <= 3 else set()     # circle only isolated chunks (a side of 3 or fewer)
    with open(out / f"ex{k}_pts.dat", "w") as f:
        f.write("x y c h s\n" + "".join(f"{x:.4f} {y:.4f} {c} {h} {int(n in stray)}\n" for n, ((x, y), c, h) in enumerate(zip(xy, cls, q["hdbscan"]))))
    with open(out / f"ex{k}_mst.dat", "w") as f:   # one segment per block, blank line between (pgfplots)
        f.write("x y\n" + "".join(f"{xy[a][0]:.4f} {xy[a][1]:.4f}\n{xy[b][0]:.4f} {xy[b][1]:.4f}\n\n" for a, b, _ in q["mst"]))
    with open(out / f"ex{k}_long.dat", "w") as f:
        f.write(f"x y\n{xy[i][0]:.4f} {xy[i][1]:.4f}\n{xy[j][0]:.4f} {xy[j][1]:.4f}\n")
    print(name, "classes:", order[:4], "+ other")
