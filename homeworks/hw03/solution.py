from pathlib import Path
import numpy as np

from plot_utils import Relative_Path, Plot_1, Plot_2


def check_results(path: str) -> None:
    for file_name in ["/pressure_by_radius.svg", "/pressure_over_time.svg"]:
        p = Relative_Path(path + file_name).path
        assert p.exists()
        assert p.stat().st_size > 0


def save_npz(out_path: Path, plot: Plot_1) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        time_h=plot.time_h,
        boundary_pressure_mpa=plot.boundary_pressure_mpa,
        well_pressure_mpa=plot.well_pressure_mpa,
        pressure_drop_mpa=plot.pressure_drop_mpa,
        well_id=plot.well_id,
        radius_m=plot.radius_m,
        observation_pressure_mpa=plot.observation_pressure_mpa,
        observation_drop_mpa=plot.observation_drop_mpa,
    )
    with np.load(out_path) as d:
        assert set(d.files) == {
            'time_h', 'boundary_pressure_mpa', 'well_pressure_mpa',
            'pressure_drop_mpa', 'well_id', 'radius_m',
            'observation_pressure_mpa', 'observation_drop_mpa',
        }
        assert d['time_h'].shape == (13,)
        assert d['well_id'].shape == (7,)
        assert np.allclose(d['pressure_drop_mpa'],
                           d['boundary_pressure_mpa'] - d['well_pressure_mpa'])


def main():
    check_results("figures/hw02")

    fig_dir = Relative_Path('/figures/hw03').path
    p1 = Plot_1(
        2, 2, (12, 8), fig_dir,
        ['pressure_overview.png', 'pressure_overview.svg'],
    )
    p1.build()

    p2 = Plot_2(
        1, 2, (12, 8), fig_dir,
        ['distributions.png', 'distributions.svg'],
    )
    p2.build()

    save_npz(
        Relative_Path('/homeworks/hw03/viz_data.npz').path,
        p1,
    )


if __name__ == '__main__':
    main()