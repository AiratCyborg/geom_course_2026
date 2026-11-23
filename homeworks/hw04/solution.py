from pathlib import Path
import numpy as np

from plot_utils import Relative_Path, FieldData, PressureMaps, Isolines


def save_field_npz(out_path: Path, field: FieldData) -> None:
    d = field.data
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vmin = d['pressure_field'].min()
    vmax = d['pressure_field'].max()
    levels = np.linspace(vmin, vmax, 9)

    np.savez(
        out_path,
        time_h=d['time_h'],
        radius_m=d['radius_m'],
        well_id=d['well_id'],
        pressure_field=d['pressure_field'],
        contour_levels=levels,
    )

    with np.load(out_path, allow_pickle=False) as data:
        assert set(data.files) == {
            'time_h', 'radius_m', 'well_id',
            'pressure_field', 'contour_levels',
        }
        assert data['pressure_field'].shape == (13, 7)
        assert data['contour_levels'].shape == (9,)
        assert np.all(np.diff(data['radius_m']) > 0)


def main():
    field = FieldData(str(Relative_Path('/homeworks/hw03/viz_data.npz')))

    fig_dir = Relative_Path('/figures/hw04').path

    maps = PressureMaps(
        1, 2, (14, 6), field, fig_dir,
        ['pressure_maps.png', 'pressure_maps.svg'],
    )
    maps.build()

    iso = Isolines(
        1, 1, (8, 6), field, fig_dir,
        ['pressure_isolines.png', 'pressure_isolines.svg'],
    )
    iso.build()

    save_field_npz(
        Relative_Path('/homeworks/hw04/field_data.npz').path,
        field,
    )


if __name__ == '__main__':
    main()