from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Relative_Path:
    ROOT = Path(__file__).resolve().parents[2]

    def __init__(self, relative_path: str) -> None:
        self.path = self.ROOT / relative_path.lstrip('/')

    def __str__(self) -> str:
        return str(self.path)


class Plot:
    def __init__(self, rows: int, cols: int, fig_size: tuple,
                 output_dir: Path, file_names: list[str]):
        self.fig, self.axes = plt.subplots(
            nrows=rows, ncols=cols, figsize=fig_size,
            constrained_layout=True, squeeze=False,
        )
        self.output_dir = output_dir
        self.file_names = file_names
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._read_data()
        self._check_tables()
        self._read_parameters()

    def _read_data(self):
        self.pumping_test = pd.read_csv(
            str(Relative_Path('/data/raw/hw01/pumping_test.txt')), sep='\t'
        )
        self.wells = pd.read_csv(
            str(Relative_Path('/data/processed/hw01/wells_clean.csv'))
        )

    def _check_tables(self):
        assert self.pumping_test.shape == (13, 4)
        assert self.wells.shape[0] == 7
        assert not self.pumping_test.isna().any().any()
        assert not self.wells.isna().any().any()

    @staticmethod
    def _to_numpy(table: pd.DataFrame, columns: list) -> list:
        return [table[col].to_numpy() for col in columns]

    def _read_parameters(self):
        (self.time_h,
         self.boundary_pressure_mpa,
         self.well_pressure_mpa) = self._to_numpy(
            self.pumping_test,
            ['time_h', 'boundary_pressure_mpa', 'well_pressure_mpa']
        )
        (self.well_id,
         self.radius_m,
         self.observation_pressure_mpa) = self._to_numpy(
            self.wells, ['well_id', 'radius_m', 'pressure_mpa']
        )
        order = np.argsort(self.radius_m)
        self.well_id = self.well_id[order]
        self.radius_m = self.radius_m[order]
        self.observation_pressure_mpa = self.observation_pressure_mpa[order]

        self.pressure_drop_mpa = self.boundary_pressure_mpa - self.well_pressure_mpa
        self.observation_drop_mpa = 12.0 - self.observation_pressure_mpa

    def _save_files(self):
        for file_name in self.file_names:
            self.fig.savefig(
                self.output_dir / file_name,
                bbox_inches="tight",
            )

    @staticmethod
    def _mark(ax, letter: str):
        ax.set_title(letter, loc='left', fontweight='bold')


class Plot_1(Plot):
    def build(self):
        ax_a, ax_b = self.axes[0, 0], self.axes[0, 1]
        ax_c, ax_d = self.axes[1, 0], self.axes[1, 1]

        ax_a.plot(self.time_h, self.boundary_pressure_mpa, label='Давление на границе')
        ax_a.plot(self.time_h, self.well_pressure_mpa, label='Давление в скважине')
        ax_a.set_xlabel('Время, ч')
        ax_a.set_ylabel('Давление, МПа')
        ax_a.legend()
        self._mark(ax_a, 'A')

        sc = ax_b.scatter(
            self.radius_m, self.observation_pressure_mpa,
            c=self.observation_pressure_mpa, cmap='viridis',
        )
        cbar = self.fig.colorbar(sc, ax=ax_b)
        cbar.set_label('Давление, МПа')
        ax_b.set_xlabel('Радиус, м')
        ax_b.set_ylabel('Давление, МПа')
        self._mark(ax_b, 'B')

        ax_c.bar(self.well_id, self.observation_drop_mpa)
        ax_c.set_xlabel('Скважина (well_id)')
        ax_c.set_ylabel('Падение давления, МПа')
        self._mark(ax_c, 'C')

        ax_d.boxplot(
            [self.boundary_pressure_mpa, self.well_pressure_mpa],
            tick_labels=['Граница', 'Скважина'],
        )
        ax_d.set_xlabel('Набор')
        ax_d.set_ylabel('Давление, МПа')
        self._mark(ax_d, 'D')

        self._save_files()


class Plot_2(Plot):
    def build(self):
        data = [self.boundary_pressure_mpa, self.well_pressure_mpa]
        names = ['Граница', 'Скважина']

        self.axes[0, 0].boxplot(data, tick_labels=names)
        self.axes[0, 0].set_xlabel('Набор')
        self.axes[0, 0].set_ylabel('Давление, МПа')
        self.axes[0, 0].set_title('Boxplot', fontweight='bold')

        self.axes[0, 1].violinplot(data, showmeans=True)
        self.axes[0, 1].set_xticks([1, 2], names)
        self.axes[0, 1].set_xlabel('Набор')
        self.axes[0, 1].set_ylabel('Давление, МПа')
        self.axes[0, 1].set_title('Violin plot', fontweight='bold')

        self._save_files()