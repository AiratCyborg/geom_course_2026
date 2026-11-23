import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


class Relative_Path:
    ROOT = Path(__file__).resolve().parents[2]

    def __init__(self, relative_path: str) -> None:
        self.path = self.ROOT / relative_path.lstrip('/')

    def __str__(self) -> str:
        return str(self.path)


class FieldData:
    def __init__(self, data_source: str):
        self._data_source = data_source
        self._read_data()
        self._build_field()
        self._check_data()

    def _read_data(self):
        keys = [
            'time_h', 'boundary_pressure_mpa',
            'well_pressure_mpa', 'pressure_drop_mpa',
            'well_id', 'radius_m',
            'observation_pressure_mpa', 'observation_drop_mpa',
        ]
        self.data = {}
        with np.load(self._data_source, allow_pickle=True) as data:
            for key in keys:
                self.data[key] = data[key]

    def _build_field(self):
        time_h = self.data['time_h']
        well_pressure_mpa = self.data['well_pressure_mpa']
        radius_m = self.data['radius_m']
        observation_pressure_mpa = self.data['observation_pressure_mpa']

        time_component = well_pressure_mpa - well_pressure_mpa[0]
        radius_component = observation_pressure_mpa - observation_pressure_mpa[0]

        self.data['time_component'] = time_component
        self.data['radius_component'] = radius_component
        self.data['pressure_field'] = (
            well_pressure_mpa[0]
            + time_component[:, np.newaxis]
            + radius_component[np.newaxis, :]
        )

    def _check_data(self):
        assert self.data['time_h'].shape == (13,)
        assert self.data['well_pressure_mpa'].shape == (13,)
        assert self.data['well_id'].shape == (7,)
        assert self.data['radius_m'].shape == (7,)
        assert self.data['observation_pressure_mpa'].shape == (7,)
        assert np.all(np.diff(self.data['time_h']) > 0)
        assert np.all(np.diff(self.data['radius_m']) > 0)
        assert self.data['pressure_field'].shape == (13, 7)


class Plot:
    def __init__(self, rows: int, cols: int, fig_size: tuple,
                 field: FieldData, output_dir: Path, file_names: list[str]):
        self.field = field
        self.fig, self.axes = plt.subplots(
            nrows=rows, ncols=cols, figsize=fig_size,
            constrained_layout=True, squeeze=False,
        )
        self.output_dir = output_dir
        self.file_names = file_names
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_files(self):
        for file_name in self.file_names:
            self.fig.savefig(self.output_dir / file_name, bbox_inches="tight")

    @staticmethod
    def _mark(ax, letter: str):
        ax.set_title(letter, loc='left', fontweight='bold')


class PressureMaps(Plot):
    def build(self):
        d = self.field.data
        time_h = d['time_h']
        radius_m = d['radius_m']
        field_for_plot = d['pressure_field'].T
        assert field_for_plot.shape == (7, 13)

        vmin = d['pressure_field'].min()
        vmax = d['pressure_field'].max()
        levels = np.linspace(vmin, vmax, 9)

        ax_a, ax_b = self.axes[0, 0], self.axes[0, 1]

        mesh = ax_a.pcolormesh(
            time_h, radius_m, field_for_plot,
            shading="auto", cmap="viridis",
            vmin=vmin, vmax=vmax, edgecolors='k', linewidth=0.2,
        )
        ax_a.set_yscale("log")
        ax_a.set_xlabel('Время, ч')
        ax_a.set_ylabel('Радиус, м')
        ax_a.set_title('Тепловая карта (pcolormesh)')
        self._mark(ax_a, 'A')

        filled = ax_b.contourf(
            time_h, radius_m, field_for_plot,
            levels=levels, cmap="viridis",
            vmin=vmin, vmax=vmax,
        )
        lines = ax_b.contour(
            time_h, radius_m, field_for_plot,
            levels=levels, colors="black", linewidths=0.6,
        )
        ax_b.clabel(lines, fmt="%.2f", fontsize=7)
        ax_b.set_yscale("log")
        ax_b.set_xlabel('Время, ч')
        ax_b.set_ylabel('Радиус, м')
        ax_b.set_title('Заливка по уровням + изолинии')
        self._mark(ax_b, 'B')

        cbar = self.fig.colorbar(mesh, ax=[ax_a, ax_b], shrink=0.9)
        cbar.set_label('Давление, МПа')

        self._save_files()


class Isolines(Plot):
    def build(self):
        d = self.field.data
        time_h = d['time_h']
        radius_m = d['radius_m']
        field_for_plot = d['pressure_field'].T

        vmin = d['pressure_field'].min()
        vmax = d['pressure_field'].max()
        levels = np.linspace(vmin, vmax, 9)

        ax = self.axes[0, 0]
        lines = ax.contour(
            time_h, radius_m, field_for_plot,
            levels=levels, cmap="viridis", linewidths=1.2,
        )
        ax.clabel(lines, fmt="%.2f", fontsize=8)
        ax.set_yscale("log")
        ax.set_xlabel('Время, ч')
        ax.set_ylabel('Радиус, м')
        ax.set_title('Изолинии давления')

        self._save_files()