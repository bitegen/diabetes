from pathlib import Path
from typing import List, Sequence, Optional

from pydantic import BaseModel
from strictyaml import YAML, load

MODULE_FOLDER = Path(__file__).resolve().parent
LOG_FOLDER = Path(__file__).resolve().parent.parent

CONFIG_FILE_PATH  = MODULE_FOLDER / "config.yml"

DATASET_DIR       = LOG_FOLDER / "datasets"
TRAINED_MODEL_DIR = LOG_FOLDER / "trained_models"
LOG_DIR           = LOG_FOLDER / "logs"


class AppConfig(BaseModel):
    """
    Application-level config.
    """
    package_name:      str
    training_data_file: str
    pipeline_save_file: str
    version:           str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model training
    and feature engineering.
    """
    target:                str
    features:              List[str]
    numerical_vars:        Sequence[str]
    numerical_vars_with_na: Sequence[str]
    test_size:             float
    random_state:          int


class Config(BaseModel):
    """Master config object."""
    app_config:         AppConfig
    model_config_params: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise FileNotFoundError(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    """Parse YAML containing the package configuration."""
    if cfg_path is None:
        cfg_path = find_config_file()

    with open(cfg_path, "r") as conf_file:
        parsed_config = load(conf_file.read())
    return parsed_config


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # Предполагается, что в parsed_config.data лежат все поля для обоих моделей
    _config = Config(
        app_config=         AppConfig(**parsed_config.data),
        model_config_params= ModelConfig(**parsed_config.data),
    )
    return _config


# Глобальная переменная, готовая к импорту
config = create_and_validate_config()
