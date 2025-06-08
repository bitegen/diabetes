import numpy as np
import pandas as pd
import pytest
from diabetes_model.predict import make_prediction
from diabetes_model.config.core import config


@pytest.fixture
def sample_input_data():
    """Создает тестовый набор данных с одним образцом."""
    return pd.DataFrame(
        {
            "Pregnancies": [6],
            "Glucose": [148],
            "BloodPressure": [72],
            "Insulin": [0],
            "BMI": [33.6],
            "Age": [50],
        }
    )


@pytest.fixture
def multiple_input_data():
    """Создает тестовый набор данных с несколькими образцами."""
    return pd.DataFrame(
        {
            "Pregnancies": [6, 1, 8, 1],
            "Glucose": [148, 85, 183, 89],
            "BloodPressure": [72, 66, 64, 66],
            "Insulin": [0, 0, 0, 94],
            "BMI": [33.6, 26.6, 23.3, 28.1],
            "Age": [50, 31, 32, 21],
        }
    )


def test_make_prediction_on_single_sample(sample_input_data):
    """
    Тест прогнозирования для одного образца.
    Проверяет структуру и типы данных результата.
    """
    # Делаем прогноз
    result = make_prediction(input_data=sample_input_data)

    # Проверяем структуру результата
    assert isinstance(result, dict)
    assert "preds" in result
    assert "probs" in result
    assert "version" in result

    # Проверяем типы данных
    assert isinstance(result["preds"], list)
    assert isinstance(result["probs"], list)
    assert isinstance(result["version"], str)

    # Проверяем размерности
    assert len(result["preds"]) == 1
    assert len(result["probs"]) == 1

    # Проверяем значения
    assert result["preds"][0] in [0, 1]  # Бинарная классификация
    assert 0 <= result["probs"][0] <= 1  # Вероятность от 0 до 1
    assert result["version"] == config.app_config.version


def test_make_prediction_on_multiple_samples(multiple_input_data):
    """
    Тест прогнозирования для нескольких образцов.
    Проверяет корректность обработки множества образцов.
    """
    # Делаем прогноз
    result = make_prediction(input_data=multiple_input_data)

    # Проверяем размерности
    assert len(result["preds"]) == 4
    assert len(result["probs"]) == 4

    # Проверяем значения
    assert all(pred in [0, 1] for pred in result["preds"])
    assert all(0 <= prob <= 1 for prob in result["probs"])


def test_make_prediction_with_missing_values():
    """
    Тест прогнозирования с пропущенными значениями.
    Проверяет обработку пропущенных значений в данных.
    """
    data_with_na = pd.DataFrame(
        {
            "Pregnancies": [6],
            "Glucose": [np.nan],  # пропущенное значение
            "BloodPressure": [72],
            "Insulin": [0],
            "BMI": [33.6],
            "Age": [50],
        }
    )

    # Модель должна обработать пропущенные значения
    result = make_prediction(input_data=data_with_na)
    assert "preds" in result
    assert len(result["preds"]) == 1


def test_make_prediction_with_edge_cases():
    """
    Тест прогнозирования с граничными случаями.
    Проверяет работу модели с экстремальными значениями.
    """
    edge_cases = pd.DataFrame(
        {
            "Pregnancies": [0, 17],  # мин и макс
            "Glucose": [0, 199],
            "BloodPressure": [0, 122],
            "Insulin": [0, 846],
            "BMI": [0, 67.1],
            "Age": [21, 81],
        }
    )

    result = make_prediction(input_data=edge_cases)
    assert len(result["preds"]) == 2
    assert all(pred in [0, 1] for pred in result["preds"])
    assert all(0 <= prob <= 1 for prob in result["probs"])


def test_make_prediction_wrong_feature_order():
    """
    Тест прогнозирования с измененным порядком признаков.
    Проверяет независимость от порядка столбцов.
    """
    data = pd.DataFrame(
        {
            "Age": [50],
            "BMI": [33.6],
            "Insulin": [0],
            "BloodPressure": [72],
            "Glucose": [148],
            "Pregnancies": [6],
        }
    )

    result = make_prediction(input_data=data)
    assert len(result["preds"]) == 1
    assert result["preds"][0] in [0, 1]


def test_make_prediction_input_validation():
    """
    Тест валидации входных данных.
    Проверяет обработку некорректных входных данных.
    """
    invalid_data = pd.DataFrame(
        {
            "Pregnancies": ["invalid"],  # неверный тип данных
            "Glucose": [148],
            "BloodPressure": [72],
            "Insulin": [0],
            "BMI": [33.6],
            "Age": [50],
        }
    )

    with pytest.raises(Exception):
        make_prediction(input_data=invalid_data)


def test_make_prediction_missing_features():
    """
    Тест на отсутствующие признаки.
    Проверяет обработку отсутствующих столбцов.
    """
    incomplete_data = pd.DataFrame(
        {
            "Pregnancies": [6],
            "Glucose": [148],
            # отсутствуют остальные признаки
        }
    )

    with pytest.raises(Exception):
        make_prediction(input_data=incomplete_data)


def test_make_prediction_empty_input():
    """
    Тест на пустые входные данные.
    Проверяет обработку пустого DataFrame.
    """
    empty_data = pd.DataFrame()

    with pytest.raises(Exception):
        make_prediction(input_data=empty_data)
