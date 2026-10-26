import pandas as pd
import numpy as np

from pathlib import Path

class Relative_Path:
    '''
    Creates a relative path object
    '''
    ROOT = Path(__file__).resolve().parents[2]

    def __init__(self, relative_path: str) -> None:    
        self.path = self.ROOT / relative_path.lstrip('/')

    def __str__(self) -> str:
        return str(self.path)

class DatasetInfo:
    def __init__(self, name, table):
        self.name = name
        self.rows, self.columns = table.shape

    def describe(self):
        return f"{self.name}: {self.rows} строк, {self.columns} столбцов"

def check_x_y(df : pd.DataFrame) -> tuple[bool]:
    theta_rad = df['azimuth_deg'] * np.pi / 180
    x_check = df['radius_m'] * np.cos(theta_rad)
    y_check = df['radius_m'] * np.sin(theta_rad)

    return np.allclose(df['x_m'], x_check, atol = 0.02), \
        np.allclose(df['y_m'], y_check, atol = 0.02)

def testing_sequence(df : pd.DataFrame) -> None:
    decay = np.exp(-df['time_h'] / 36)

    # check sequence get down
    print(list(decay))

def print_chars(col : pd.Series) -> None:
    INFO = f"{col.min()}\n{col.max()}\n{col.mean()}"
    
    print(INFO)

def choose_elems(col : pd.Series) -> None:
    np_col = col.to_numpy()

    INFO = f"First: {np_col[0]}\n\
        Last: {np_col[-1]}\n\
        Even piece: {np_col[::2]}\n\
        Less than average: {list(np_col[np_col < np_col.mean()])}\n"

    print(INFO)

def print_numpy_array_info(arr : np.array) -> None:
    print(arr.ndim, arr.shape, arr.size, arr.dtype, sep = "\n")

def transpose(arr1 : np.array, arr2 : np.array) -> None:
    # transpose arr
    arr_t = arr1.T; print(arr_t.shape)

    # reshaping
    flat = arr2.reshape(-1)
    restored = flat.reshape(arr2.shape)

    assert np.allclose(restored, arr2)

def print_table_info(name : str, table : pd.DataFrame) -> None:
    print(f"A table {name} information: ")

    INFO = f"Dimension: {table.shape}\
        \nColumns: {table.columns}\
        \nTypes: {table.dtypes}\
        \nHead information: {table.head()}"

    print(INFO)

# wells.csv
def read_wells(path : str) -> pd.DataFrame:
    wells = pd.read_csv(path)

    print_table_info("wells.csv", wells)
    print(wells.isna().sum())

    #cleaned wells.csv
    wells_clean = wells.dropna(subset = ["pressure_mpa"]).copy()
    assert (wells_clean["radius_m"] > 0).all()

    # add a new columns
    wells_clean["pressure_pa"] = wells_clean["pressure_mpa"] * 1e6
    wells_clean["pressure_difference_mpa"] = 12 - wells_clean["pressure_mpa"]
    wells_clean["relative_change_percent"] = wells_clean["pressure_difference_mpa"] / 12 * 100

    print([['No', 'Yes'][i] for i in check_x_y(wells_clean)])

    # I don't know for what but: 
    log_radius = np.log(wells_clean["radius_m"])
    wells_clean["log_radius"] = log_radius

    return wells_clean

# layers.xlsx
def read_layers(path : str) -> pd.DataFrame:
    layers = pd.read_excel(path)

    # printing data info
    print_table_info("layers.xlsx", layers)

    # checking conditions
    assert (layers["thickness_m"] > 0).all()
    assert layers["porosity_fraction"].between(0, 1).all()

    return layers

# pumping_test.txt
def read_pumping_test(path : str) -> pd.DataFrame:
    pumping_test = pd.read_csv(path, sep = '\t')

    # printing data info
    print_table_info("pumping_test.txt", pumping_test)

    # creates exp sequence
    testing_sequence(pumping_test)

    # 2 dimensional numpy array
    pressure_matrix = pumping_test[
        ["boundary_pressure_mpa", "well_pressure_mpa"]
    ].to_numpy()

    print_numpy_array_info(pressure_matrix)

    # some experiments (idk for what)
    experiment_1 = pressure_matrix
    experiment_2 = pressure_matrix + 0.05

    pressure_cube = np.stack([experiment_1, experiment_2], axis=0)
    print(pressure_cube[0], pressure_cube[1], pressure_cube[0, 0], \
          pressure_cube[:, :, 1], sep = '\n')

    # transpose
    transpose(pressure_matrix, pressure_cube)

    return pumping_test