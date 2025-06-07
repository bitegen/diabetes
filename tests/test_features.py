import random

import pytest
import pandas as pd
from pathlib import Path

from diabetes_model.processing.data_manager import load_dataset
from diabetes_model.config.core import config, DATASET_DIR


def test_three_random_rows_are_numeric():
    """
    Берём 3 случайные строки из датасета и проверяем,
    что во всех ячейках числовые значения (приводятся к float без ошибок).
    """
    df: pd.DataFrame = load_dataset(file_name=config.app_config.training_data_file)
    assert not df.empty, "Датасет пустой"

    n_rows = df.shape[0]
    n_samples = min(3, n_rows)  # либо 3, либо меньше, если строк < 3

    random_indices = random.sample(list(df.index), k=n_samples)

    for idx in random_indices:
        row = df.loc[idx]
        for col, val in row.items():
            if pd.isna(val):
                continue

            try:
                float(val)
            except Exception as e:
                pytest.fail(
                    f"Строка {idx}, колонка '{col}': "
                    f"значение {val!r} не приводится к float ({e})"
                )
