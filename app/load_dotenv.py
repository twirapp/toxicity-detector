from os import environ
from pathlib import Path

__all__ = ["load_env_to_environ"]


def _format(key_or_value: str) -> str:
    return key_or_value.strip(" \"'")


def _get_env_data_as_dict(path: Path | str) -> dict[str, str]:
    with open(path, "r") as f:
        return dict(
            list(map(_format, line.replace("\n", "").split("=", 1)))
            for line in f.readlines()
            if not line.startswith("#") and line.count("=") >= 1
        )


def load_env_to_environ(path: Path | str = ".", quiet=True):
    path = Path(path)
    if path.is_dir():
        path = path / ".env"

    try:
        vars_dict = _get_env_data_as_dict(path)
        environ.update(vars_dict)
    except Exception:
        if not quiet:
            raise
