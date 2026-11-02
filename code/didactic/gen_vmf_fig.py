# gen_vmf_fig.py -- Data for the vMF appendix figure: three toy clouds of 50 unit
# directions on the sphere in 3-D (one meaning; two meanings; one elongated meaning), and the same directions after the
# registered step of the two-meaning statistic (centre, project onto the leading principal directions, renormalise;
# here 2 directions so that the result can be drawn as points on a circle). vmf2 = BIC(1) - BIC(2) from the paper's own
# code (code/realtext/vmf.py: bic_gap) on the projected points; the share of 200 replicate clouds with vmf2 > 0 is reported
# per case (seed 0 is the cloud drawn). A 3-D sketch of the paper's 5-of-D projection, not a re-analysis.
# Usage (imports code/realtext/vmf.py; scipy): python gen_vmf_fig.py OUT_DIR
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realtext"))
from vmf import bic_gap  # noqa: E402

N = 50
CASES = {  # name: list of (centre angle along x in degrees, share of points, s.d. along x, s.d. along y)
    "one": [(0, 1.0, 0.16, 0.16)],
    "two": [(-28, 0.5, 0.07, 0.07), (28, 0.5, 0.07, 0.07)],
    "long": [(0, 1.0, 0.30, 0.05)]}


def cloud(spec, rng):
    pts = []
    for ang, share, sx, sy in spec:
        m = int(round(share * N)); a = np.radians(ang)
        c, ex = np.array([np.sin(a), 0, np.cos(a)]), np.array([np.cos(a), 0, -np.sin(a)])
        t = rng.standard_normal((m, 2)) * [sx, sy]
        pts.append(c + t[:, :1] * ex + t[:, 1:] * np.array([0, 1, 0]))
    X = np.vstack(pts)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def project(X, r=2):
    _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    Y = (X - X.mean(0)) @ Vt[:r].T          # centred, so the elongated cloud splits at its two ends
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    meta = []
    for name, spec in CASES.items():
        X = cloud(spec, np.random.default_rng(0)); Y = project(X)
        g0 = bic_gap(Y, 0)
        hits = int(sum(bic_gap(project(cloud(spec, np.random.default_rng(s))), s) > 0 for s in range(200)))
        np.savetxt(out / f"vmf_{name}_sphere.dat", X[:, :2], fmt="%.4f", header="x y", comments="")
        np.savetxt(out / f"vmf_{name}_proj.dat", Y, fmt="%.4f", header="x y", comments="")
        meta.append(f"\\def\\vmfGap{name.capitalize()}{{{g0:+.0f}}}\\def\\vmfHits{name.capitalize()}{{{hits}}}")
        print(name, f"vmf2 (seed 0) {g0:+.1f}; vmf2 > 0 in {hits} of 200 clouds", flush=True)
    (out / "vmf_meta.tex").write_text("% gen_vmf_fig.py output\n" + "\n".join(meta) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
