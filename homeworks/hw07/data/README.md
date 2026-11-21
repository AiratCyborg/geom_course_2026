# Температурное поле для Matplotlib

`thermal_history.npz` содержит без изменений массивы `x_m`, `z_m`, `ages_ma`, `temperature_C`, `active` из `homeworks/hw06/data/result.h5`. Для каждой эпохи сохранены исходные центры ячеек, маска и значения температуры, включая NaN. Порядок строк соответствует исходному HDF5; разворот строк, применявшийся при экспорте VTK, здесь не выполняется.

Исходный формат: basin-thermal-history, версия схемы 1. Происхождение расчёта описано в [ДЗ 6](../../hw06/data/README.md). В этой работе используется двумерное поле со временем, а не объёмная сетка. Третья ось графика поверхности — температура.

SHA-256 исходного HDF5:
`7b58d31534a65a1785aa0f8da6b7cfad9665587048acd75b193c0b275ebeb979`

SHA-256 NPZ:
`823caad87732d36b2ebacb600d94ccc659ff7486a85ec986f5767939c7a4c0fb`

Для чтения достаточно NumPy:

```python
from pathlib import Path
import numpy as np

path = Path(__file__).resolve().parent / "data" / "thermal_history.npz"
with np.load(path, allow_pickle=False) as data:
    x_m = data["x_m"].copy()
    z_m = data["z_m"].copy()
    ages_ma = data["ages_ma"].copy()
    temperature_C = data["temperature_C"].copy()
    active = data["active"].copy()
```
