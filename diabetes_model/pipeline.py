from feature_engine.imputation import AddMissingIndicator, MeanMedianImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.core import config

diabetes_pipe = Pipeline(
    [
        # 1) добавляем индикаторы пропусков там, где нужно
        (
            "missing_indicator",
            AddMissingIndicator(
                variables=config.model_config_params.numerical_vars_with_na
            ),
        ),

        # 2) заполняем численные признаки медианой
        (
            "median_imputation",
            MeanMedianImputer(
                imputation_method="median",
                variables=config.model_config_params.numerical_vars_with_na,
            ),
        ),

        # 3) стандартизируем все числовые признаки
        ("scaler", StandardScaler()),

        # 4) классификатор
        (
            "classifier",
            RandomForestClassifier(random_state=config.model_config_params.random_state),
        ),
    ]
)
