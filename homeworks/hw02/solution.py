import pandas as pd

from pathlib import Path

from plot_utils import plot_pressure_over_time, plot_pressure_by_radius


class Relative_Path:
    '''
    Creates a relative path object
    '''
    ROOT = Path(__file__).resolve().parents[2]

    def __init__(self, relative_path: str) -> None:
        self.path = self.ROOT / relative_path.lstrip('/')

    def __str__(self) -> str:
        return str(self.path)


def main() -> None:
    # relative paths
    pumping_path = Relative_Path("/data/raw/hw01/pumping_test.txt")
    wells_path = Relative_Path("/data/processed/hw01/wells_clean.csv")

    figures_dir = Relative_Path("/figures/hw02")
    figures_dir.path.mkdir(parents=True, exist_ok=True)

    # read pumping test
    pumping_test = pd.read_csv(str(pumping_path), sep="\t")

    # read cleaned wells
    wells_clean = pd.read_csv(str(wells_path))

    # sort by distance before plotting
    wells_sorted = wells_clean.sort_values("radius_m")

    # log scale requires positive radius values
    assert (wells_sorted["radius_m"] > 0).all()

    # build both plots
    plot_pressure_over_time(pumping_test, figures_dir.path)
    plot_pressure_by_radius(wells_sorted, figures_dir.path)

    # check that all four output files exist
    assert (figures_dir.path / "pressure_over_time.png").exists()
    assert (figures_dir.path / "pressure_over_time.svg").exists()
    assert (figures_dir.path / "pressure_by_radius.png").exists()
    assert (figures_dir.path / "pressure_by_radius.svg").exists()

    print("Figures saved to", figures_dir.path)


if __name__ == '__main__':
    # A main block of code
    main()