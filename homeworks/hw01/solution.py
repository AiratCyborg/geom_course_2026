import numpy as np
import pandas as pd

from pathlib import Path

from data_utils import (
    Relative_Path,
    DatasetInfo,
    read_wells,
    read_layers,
    read_pumping_test,
    print_chars,
    choose_elems,
)

if __name__ == '__main__':
    # A main block of code
    
    def main() -> None:
        # Files created in generate_data.py
        hw01_raw = '/data/raw/hw01/'
        hw01_processed = Relative_Path('/data/processed/hw01/')
        hw01_exports = Relative_Path('/exports/hw01/')

        # 1. Read the three files
        wells_path = Relative_Path(hw01_raw + 'wells.csv')
        layers_path = Relative_Path(hw01_raw + 'layers.xlsx')
        pumping_path = Relative_Path(hw01_raw + 'pumping_test.txt')

        wells_clean = read_wells(str(wells_path))
        layers = read_layers(str(layers_path))
        pumping_test = read_pumping_test(str(pumping_path))

        # 2. Three DatasetInfo objects
        info_wells = DatasetInfo("wells.csv", wells_clean)
        info_layers = DatasetInfo("layers.xlsx", layers)
        info_pumping = DatasetInfo("pumping_test.txt", pumping_test)
        for info in (info_wells, info_layers, info_pumping):
            print(info.describe())

        # 3. Min / max / mean
        print("--- pressure_mpa ---"); print_chars(wells_clean["pressure_mpa"])
        print("--- radius_m ---");     print_chars(wells_clean["radius_m"])
        print("--- thickness_m ---");  print_chars(layers["thickness_m"])
        print("--- porosity_fraction ---"); print_chars(layers["porosity_fraction"])

        # 4. Indexing, slicing, boolean mask
        choose_elems(wells_clean["pressure_mpa"])
        print("Rows with radius > 100 m:")
        print(wells_clean[wells_clean["radius_m"] > 100])

        # 5. pressure_matrix and pressure_cube
        pressure_matrix = pumping_test[
            ["boundary_pressure_mpa", "well_pressure_mpa"]
        ].to_numpy()
        experiment_1 = pressure_matrix
        experiment_2 = pressure_matrix + 0.05
        pressure_cube = np.stack([experiment_1, experiment_2], axis=0)

        assert pressure_matrix.shape == (13, 2)
        assert pressure_cube.shape == (2, 13, 2)

        # 6. Save outputs
        processed_dir = Path(str(hw01_processed))
        exports_dir = Path(str(hw01_exports))
        processed_dir.mkdir(parents=True, exist_ok=True)
        exports_dir.mkdir(parents=True, exist_ok=True)

        # CSV
        wells_clean.to_csv(processed_dir / "wells_clean.csv", index=False)

        # Excel: min/max/mean for four columns
        summary = pd.DataFrame({
            "value": ["min", "max", "mean"],
            "pressure_mpa": [
                wells_clean["pressure_mpa"].min(),
                wells_clean["pressure_mpa"].max(),
                wells_clean["pressure_mpa"].mean(),
            ],
            "radius_m": [
                wells_clean["radius_m"].min(),
                wells_clean["radius_m"].max(),
                wells_clean["radius_m"].mean(),
            ],
            "thickness_m": [
                layers["thickness_m"].min(),
                layers["thickness_m"].max(),
                layers["thickness_m"].mean(),
            ],
            "porosity_fraction": [
                layers["porosity_fraction"].min(),
                layers["porosity_fraction"].max(),
                layers["porosity_fraction"].mean(),
            ],
        })
        summary.to_excel(processed_dir / "table_summary.xlsx", index=False)

        # NPY
        np.save(processed_dir / "pressure_matrix.npy", pressure_matrix)
        # NPZ
        np.savez(processed_dir / "pressure_cube.npz", pressure_cube=pressure_cube)

        # TXT report
        with open(exports_dir / "results.txt", "w", encoding="utf-8") as f:
            f.write(f"pressure_matrix.shape = {pressure_matrix.shape}\n")
            f.write(f"pressure_cube.shape   = {pressure_cube.shape}\n")
            for name, series in [
                ("pressure_mpa", wells_clean["pressure_mpa"]),
                ("radius_m", wells_clean["radius_m"]),
                ("thickness_m", layers["thickness_m"]),
                ("porosity_fraction", layers["porosity_fraction"]),
            ]:
                f.write(f"{name}: min={series.min()}, max={series.max()}, mean={series.mean()}\n")

        # Reload and verify
        matrix_loaded = np.load(processed_dir / "pressure_matrix.npy")
        cube_loaded = np.load(processed_dir / "pressure_cube.npz")["pressure_cube"]
        assert np.allclose(matrix_loaded, pressure_matrix)
        assert np.allclose(cube_loaded, pressure_cube)
        print("Reloaded arrays match originals.")

    main()