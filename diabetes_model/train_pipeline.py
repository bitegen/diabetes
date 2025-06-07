import logging
from pathlib import Path

from config.core import LOG_DIR, config
from pipeline import diabetes_pipe
from processing.data_manager import load_dataset, save_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


def run_training() -> None:
    """Train the model on Diabetes data."""
    # Настройка логирования
    log_path = Path(f"{LOG_DIR}/log_{config.app_config.version}.log")
    if log_path.exists():
        log_path.unlink()
    logging.basicConfig(filename=log_path, level=logging.INFO)

    # Загружаем датасет
    data = load_dataset(file_name=config.app_config.training_data_file)

    print(
        "Available columns:",
        config.app_config.training_data_file,
        data.columns.tolist(),
    )
    print("Config wants:   ", config.model_config_params.features)

    # Делим на X и y
    X = data[config.model_config_params.features]
    y = data[config.model_config_params.target]

    # Сплит на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.model_config_params.test_size,
        random_state=config.model_config_params.random_state,
    )

    # Обучаем пайплайн
    diabetes_pipe.fit(X_train, y_train)

    # Оценка на тренировочном
    y_pred_train = diabetes_pipe.predict(X_train)
    y_pred_proba_train = diabetes_pipe.predict_proba(X_train)[:, 1]

    train_acc = accuracy_score(y_train, y_pred_train)
    train_roc_auc = roc_auc_score(y_train, y_pred_proba_train)

    logging.info(f"Train ACC: {train_acc: .3f}, ROC-AUC: {train_roc_auc: .3f}")
    print(f"Train ACC: {train_acc: .3f}, ROC-AUC: {train_roc_auc: .3f}")

    # Оценка на тестовом
    y_pred_test = diabetes_pipe.predict(X_test)
    y_pred_proba_test = diabetes_pipe.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, y_pred_test)
    test_roc_auc = roc_auc_score(y_test, y_pred_proba_test)

    logging.info(f"Test  ACC: {test_acc: .3f}, ROC-AUC: {test_roc_auc: .3f}")
    print(f"Test  ACC: {test_acc: .3f}, ROC-AUC: {test_roc_auc: .3f}")

    # Сохраняем модель
    save_pipeline(pipeline_to_persist=diabetes_pipe)


if __name__ == "__main__":
    run_training()
