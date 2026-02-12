import json
from pathlib import Path

from pydantic import TypeAdapter

from src.config.experiments import ExperimentConfig

config_adapter: TypeAdapter[ExperimentConfig] = TypeAdapter(ExperimentConfig)


def load_config(path: Path | str) -> ExperimentConfig:
    path = Path(path)
    with path.open() as f:
        data = json.load(f)
    return config_adapter.validate_python(data)
