from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "thermal_history.npz"
ROOT = HERE.parents[1]
FIG_DIR = ROOT / "figures" / "hw07"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_A_INDEX = 4     # change here
EPOCH_B_INDEX = 13    # change here

T_MIN, T_MAX = 17.0, 170.0
CMAP = "viridis"


def load_data():
    assert DATA_PATH.exists(), f"Missing {DATA_PATH}"
    with np.load(DATA_PATH, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def check_data(d):
    x_m = d["x_m"]; z_m = d["z_m"]; ages = d["ages_ma"]
    T = d["temperature_C"]; active = d["active"]

    assert x_m.shape == (46,)
    assert z_m.shape == (112,)
    assert ages.shape == (18,)
    assert T.shape == (18, 112, 46)
    assert active.shape == (18, 112, 46)

    assert np.all(np.diff(x_m) > 0)
    assert np.all(np.diff(z_m) > 0)
    assert np.all(np.diff(ages) < 0)
    assert set(np.unique(active)).issubset({0, 1})

    for k in range(T.shape[0]):
        inside = active[k] == 1
        outside = ~inside
        assert np.all(np.isfinite(T[k][inside]))
        assert np.all(np.isnan(T[k][outside]))


def frame(k, d):
    T = d["temperature_C"][k]
    active = d["active"][k]
    return np.where(active == 1, T, np.nan)


def epoch_stats(k, d):
    T = d["temperature_C"][k]
    inside = d["active"][k] == 1
    vals = T[inside]
    print(f"[epoch {k}] age={d['ages_ma'][k]:.1f} Ma  "
          f"active={inside.sum()}  T_min={vals.min():.2f}  T_max={vals.max():.2f}")
    return inside, vals


def control_points(k, d):
    active = d["active"][k] == 1
    z_m = d["z_m"]; x_m = d["x_m"]; T = d["temperature_C"][k]

    rows = np.where(active.any(axis=1))[0]
    j_low, j_mid, j_high = rows.min(), rows[len(rows) // 2], rows.max()

    def pick(j):
        i = int(np.argmax(active[j]))
        return j, i, x_m[i], z_m[j], T[j, i]

    pts = {"A": pick(j_low), "B": pick(j_mid), "C": pick(j_high)}
    for name, (j, i, x, z, t) in pts.items():
        print(f"[epoch {k}] {name}: j={j} i={i}  x={x:.1f} m  z={z:.1f} m  T={t:.2f} °C")
    return pts


def make_grid(d):
    X, Z = np.meshgrid(d["x_m"], d["z_m"], indexing="xy")
    assert X.shape == Z.shape == (112, 46)
    j, i = 10, 20
    assert X[j, i] == d["x_m"][i]
    assert Z[j, i] == d["z_m"][j]
    return X, Z


def scatter_control(ax3d, pts):
    for name, (j, i, x, z, t) in pts.items():
        ax3d.scatter([x], [z], [t], color="red", s=40, depthshade=False)
        ax3d.text(x, z, t, f" {name}", color="black", fontsize=8)


def style_3d(ax):
    ax.set_xlabel("x, m")
    ax.set_ylabel("z, m")
    ax.set_zlabel("Temperature, °C")
    ax.set_zlim(T_MIN, T_MAX)


def plot_comparison(d, X, Z, kA, kB):
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)

    for row, k in enumerate((kA, kB)):
        T = frame(k, d)
        age = d["ages_ma"][k]
        pts = control_points(k, d)

        ax_map = fig.add_subplot(3, 3, row * 3 + 1)
        mesh = ax_map.pcolormesh(X, Z, T, shading="nearest",
                                 cmap=CMAP, vmin=T_MIN, vmax=T_MAX)
        ax_map.invert_yaxis()
        ax_map.set_aspect("equal")
        ax_map.set_xlabel("x, m"); ax_map.set_ylabel("z, m")
        ax_map.set_title(f"Map, {age:.1f} Ma")
        for name, (j, i, x, z, t) in pts.items():
            ax_map.plot(x, z, "o", color="red", markersize=5)
            ax_map.annotate(name, (x, z), color="black", fontsize=8)

        ax_surf = fig.add_subplot(3, 3, row * 3 + 2, projection="3d")
        surf = ax_surf.plot_surface(X, Z, T, cmap=CMAP,
                                    vmin=T_MIN, vmax=T_MAX,
                                    rcount=len(d["z_m"]), ccount=len(d["x_m"]))
        style_3d(ax_surf)
        ax_surf.set_title(f"Surface, {age:.1f} Ma")
        ax_surf.view_init(elev=25, azim=-60)
        scatter_control(ax_surf, pts)

        ax_wire = fig.add_subplot(3, 3, row * 3 + 3, projection="3d")
        ax_wire.plot_wireframe(X, Z, T, rstride=1, cstride=1,
                               color="C0", linewidth=0.3)
        style_3d(ax_wire)
        ax_wire.set_title(f"Wireframe, {age:.1f} Ma")
        ax_wire.view_init(elev=25, azim=-60)

    fig.colorbar(mesh, ax=fig.axes[:3], shrink=0.6, label="Temperature, °C")
    fig.savefig(FIG_DIR / "comparison.png", dpi=150)
    fig.savefig(FIG_DIR / "comparison.svg")
    plt.close(fig)


def plot_camera(d, X, Z, k):
    T = frame(k, d)
    age = d["ages_ma"][k]
    pts = control_points(k, d)
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)

    for col, (elev, azim, label) in enumerate(
        [(25, -60, "view 1 (elev=25, azim=-60)"),
         (10, 120, "view 2 (elev=10, azim=120)")]
    ):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        ax.plot_surface(X, Z, T, cmap=CMAP, vmin=T_MIN, vmax=T_MAX,
                        rcount=len(d["z_m"]), ccount=len(d["x_m"]))
        style_3d(ax)
        ax.set_title(f"{label}, {age:.1f} Ma")
        ax.view_init(elev=elev, azim=azim)
        scatter_control(ax, pts)

    fig.savefig(FIG_DIR / "camera.png", dpi=150)
    plt.close(fig)


def plot_mask_error(d, X, Z, k):
    T_ok = frame(k, d)
    active = d["active"][k] == 1
    T_bad = np.where(active, d["temperature_C"][k], 0.0)
    age = d["ages_ma"][k]

    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    for col, (T, title) in enumerate(
        [(T_ok, "Correct: NaN outside"),
         (T_bad, "Ошибочный вариант: вне области T = 0")]
    ):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        ax.plot_surface(X, Z, T, cmap=CMAP, vmin=0.0, vmax=170.0,
                        rcount=len(d["z_m"]), ccount=len(d["x_m"]))
        style_3d(ax)
        ax.set_zlim(0.0, 170.0)
        ax.set_title(f"{title}, {age:.1f} Ma")
        ax.view_init(elev=25, azim=-60)

    outside = np.argwhere(~active)
    j, i = map(int, outside[0])
    print(f"[mask] inactive cell j={j}, i={i}  "
          f"x={d['x_m'][i]:.1f} m  z={d['z_m'][j]:.1f} m  "
          f"T_orig=NaN  T_bad=0")

    fig.savefig(FIG_DIR / "mask_error.png", dpi=150)
    plt.close(fig)


def plot_animation(d, X, Z):
    ages = d["ages_ma"]
    n = len(ages)
    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    def update(k):
        ax.clear()
        T = frame(k, d)
        ax.plot_surface(X, Z, T, cmap=CMAP, vmin=T_MIN, vmax=T_MAX,
                        rcount=len(d["z_m"]), ccount=len(d["x_m"]))
        style_3d(ax)
        ax.set_zlim(T_MIN, T_MAX)
        ax.view_init(elev=25, azim=-60)
        ax.set_title(f"Age: {ages[k]:.1f} Ma")
        return ax,

    anim = FuncAnimation(fig, update, frames=n, interval=500)
    anim.save(FIG_DIR / "thermal_history.gif", writer=PillowWriter(fps=2))
    plt.close(fig)


def main():
    d = load_data()
    check_data(d)

    kA, kB = EPOCH_A_INDEX, EPOCH_B_INDEX
    print(f"Epochs: A k={kA} age={d['ages_ma'][kA]:.1f} Ma  "
          f"B k={kB} age={d['ages_ma'][kB]:.1f} Ma")
    epoch_stats(kA, d)
    epoch_stats(kB, d)

    X, Z = make_grid(d)

    plot_comparison(d, X, Z, kA, kB)
    plot_camera(d, X, Z, kA)
    plot_mask_error(d, X, Z, kA)
    plot_animation(d, X, Z)


if __name__ == "__main__":
    main()