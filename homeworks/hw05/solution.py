import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

class Relative_Path:
    ROOT = Path(__file__).resolve().parents[2]

    def __init__(self, relative_path: str):
        self.path = self.ROOT / relative_path.lstrip('/')

    def __str__(self):
        return str(self.path)

class FieldData:
    def __init__(self, data_source: str):
        self._data_source = data_source
        self._read_data()
        self._check_data()

    def _read_data(self):
        keys = [
            'time_h', 'radius_m', 'well_id',
            'pressure_field', 'contour_levels'
        ]
        self.data = {}
        with np.load(self._data_source, allow_pickle=True) as data:
            for key in keys:
                self.data[key] = data[key]

    def _check_data(self):
        assert self.data['time_h'].shape == (13,); assert self.data['radius_m'].shape == (7,)
        assert self.data['well_id'].shape == (7,); assert self.data['pressure_field'].shape == (13, 7)
        assert self.data['contour_levels'].shape == (9,); assert np.all(np.diff(self.data['time_h']) > 0)
        assert np.all(np.diff(self.data['radius_m']) > 0)

class Table:

    def __init__(self, save_file : str):
        self.save_file = save_file
        self.data = FieldData(
            str(Relative_Path('homeworks/hw04/field_data.npz'))
        ).data
        
        self.build_table(); self._check_table()
        self._save_table()

    def build_table(self):
        time_grid, radius_grid = np.meshgrid(
            self.data['time_h'],
            self.data['radius_m'],
            indexing="ij",
        )

        _, radius_index = np.indices(
            self.data['pressure_field'].shape
        )

        well_names_long = self.data['well_id'][
            radius_index.reshape(-1)
        ]

        self.long_table = pd.DataFrame({
            'time_h':       time_grid.reshape(-1),
            'radius_m':     radius_grid.reshape(-1),
            'well_id':      well_names_long,
            'pressure_mpa': self.data['pressure_field'].reshape(-1),
        })

    def _check_table(self):
        assert self.long_table.shape == (91, 4); assert self.long_table["time_h"].nunique() == 13
        assert self.long_table["radius_m"].nunique() == 7; assert not self.long_table.isna().any().any()

    def _save_table(self):
        self.long_table.to_csv(self.save_file, index = False)

class PlotAnimation:

    def __init__(self, field: FieldData, save_file: str):
        self.field = field
        self.save_file = Path(save_file)
        self.save_file.parent.mkdir(parents=True, exist_ok=True)

        self.fig, self.ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    def build(self):
        d = self.field.data
        time_h = d['time_h']
        radius_m = d['radius_m']
        pressure_field = d['pressure_field']

        self.ax.set_xscale('log')
        margin = 0.1; y_min = pressure_field.min() - margin
        y_max = pressure_field.max() + margin
        
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_xlabel('Радиус, м')
        self.ax.set_ylabel('Давление, МПа')

        line, = self.ax.plot(radius_m, pressure_field[0], marker = 'o')
        points, = self.ax.plot(radius_m, pressure_field[0], 'o', color = 'C0')
        title = self.ax.set_title(f'Время: {time_h[0]:.2f} ч')

        def update(frame_index):
            y = pressure_field[frame_index]
            line.set_ydata(y)
            points.set_ydata(y)
            title.set_text(f'Время: {time_h[frame_index]:.2f} ч')
            return line, points, title

        self.anim = FuncAnimation(
            self.fig, update,
            frames=len(time_h),
            interval=300,
            blit=False,
        )
        self.anim.save(self.save_file, writer = PillowWriter(fps = 13))
        return self.anim

def check_hw04_figure() -> None:
    svg = Relative_Path('/figures/hw04/pressure_maps.svg').path
    assert svg.exists()
    assert svg.stat().st_size > 0

def main():
    check_hw04_figure()

    field = FieldData(
        str(Relative_Path('/homeworks/hw04/field_data.npz'))
    )

    Table(
        str(Relative_Path('/homeworks/hw05/pressure_long.csv')),
    )

    PlotAnimation(
        field,
        str(Relative_Path('/figures/hw05/pressure_animation.gif')),
    ).build()

if __name__ == '__main__':
    main()