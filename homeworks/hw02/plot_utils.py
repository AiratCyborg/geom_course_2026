import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pathlib import Path


def find_min_pressure(table: pd.DataFrame) -> tuple:
    """Return (min_time, min_pressure) for the minimum well pressure."""
    min_index = table["well_pressure_mpa"].idxmin()

    # min time and pressure
    min_time = table.loc[min_index, "time_h"]
    min_pressure = table.loc[min_index, "well_pressure_mpa"]

    return min_time, min_pressure


def plot_pressure_over_time(table: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot pressure over time (two lines on a single Axes)
    and save both PNG and SVG.
    """
    # figure and axes — one Axes with two lines, as required
    fig, ax = plt.subplots(figsize=(8, 5))

    t = table["time_h"]
    b = table["boundary_pressure_mpa"]
    w = table["well_pressure_mpa"]

    ax.plot(t, b, color="tab:blue", marker="o", markersize=5,
            linewidth=1.5, label="Boundary pressure")
    ax.plot(t, w, color="tab:red", marker="s", markersize=5,
            linewidth=1.5, label="Well pressure")

    # axis labels with units
    ax.set_xlabel("Time, h")
    ax.set_ylabel("Pressure, MPa")

    # title, grid, legend
    ax.set_title("Pressure change over time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # annotation for min value — point on the well_pressure_mpa line
    min_time, min_pressure = find_min_pressure(table)
    ax.annotate(
        f"{min_pressure:.2f} MPa",
        xy=(min_time, min_pressure),
        xytext=(20, 20),
        textcoords="offset points",
        fontsize=9,
        color="black",
        ha="left",
        va="bottom",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )

    fig.tight_layout()

    # save both formats with bbox_inches="tight"
    fig.savefig(output_dir / "pressure_over_time.png",
                dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "pressure_over_time.svg",
                bbox_inches="tight")

    # close figure to free memory
    plt.close(fig)


def plot_pressure_by_radius(table: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot pressure_mpa(radius_m) as scatter with a log X scale
    and save both PNG and SVG.
    """
    # log scale requires strictly positive values
    assert (table["radius_m"] > 0).all()

    fig, ax = plt.subplots(figsize=(8, 5))

    x = table["radius_m"]
    y = table["pressure_mpa"]

    ax.plot(x, y, linestyle="none", marker="o", markersize=7,
            color="tab:green")

    # logarithmic X scale
    ax.set_xscale("log")

    ax.set_xlabel("Distance, m")
    ax.set_ylabel("Pressure, MPa")
    ax.set_title("Pressure in observation wells")

    # grid with alpha=0.3, both major and minor ticks for log scale
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()

    # save both formats with bbox_inches="tight"
    fig.savefig(output_dir / "pressure_by_radius.png",
                dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "pressure_by_radius.svg",
                bbox_inches="tight")

    plt.close(fig)