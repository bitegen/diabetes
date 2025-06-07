import json

import pytest
from fastapi.testclient import TestClient

from app.app import app

client = TestClient(app)


def test_serve_frontend():
    """
    GET / должен вернуть index.html
    """
    resp = client.get("/")
    assert resp.status_code == 200
    text = resp.text.strip()
    assert text.lower().startswith("<!doctype html>")
    assert "Diabetes Prediction" in text


@pytest.mark.parametrize(
    "payload",
    [
        # нормальный случай
        {
            "Pregnancies": 2.0,
            "Glucose": 120.0,
            "BloodPressure": 70.0,
            "Insulin": 80.0,
            "BMI": 25.0,
            "Age": 40.0,
        },
        # граничные нули
        {
            "Pregnancies": 0.0,
            "Glucose": 0.0,
            "BloodPressure": 0.0,
            "Insulin": 0.0,
            "BMI": 0.0,
            "Age": 21.0,
        },
        # высокий возраст
        {
            "Pregnancies": 5.0,
            "Glucose": 190.0,
            "BloodPressure": 90.0,
            "Insulin": 200.0,
            "BMI": 45.0,
            "Age": 81.0,
        },
    ],
)
def test_predict_valid(payload):
    """
    POST /predict с корректным JSON (все обязательные поля) должен вернуть 200 и поле prediction
    """
    resp = client.post(
        "/predict",
        data=json.dumps(payload),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # проверяем, что модель возвращает 0 или 1
    assert "prediction" in body
    assert isinstance(body["prediction"], int)
    assert body["prediction"] in (0, 1)


def test_predict_missing_field():
    """
    Если не передать одно из обязательных полей, должно быть 422 Unprocessable Entity
    """
    bad = {
        # убрали Glucose
        "Pregnancies": 1.0,
        "BloodPressure": 70.0,
        "Insulin": 80.0,
        "BMI": 25.0,
        "Age": 40.0,
    }
    resp = client.post(
        "/predict",
        data=json.dumps(bad),
    )
    assert resp.status_code == 422
    err = resp.json()
    assert "detail" in err


def test_predict_wrong_type():
    """
    Если тип поля неверен (строка вместо числа), тоже 422
    """
    bad = {
        "Pregnancies": "two",
        "Glucose": 120.0,
        "BloodPressure": 70.0,
        "Insulin": 80.0,
        "BMI": 25.0,
        "Age": 40.0,
    }
    resp = client.post(
        "/predict",
        data=json.dumps(bad),
    )
    assert resp.status_code == 422
    err = resp.json()
    assert "detail" in err
