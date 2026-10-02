"""Load, validate and hash experiment configs. Fails closed on anything missing or ambiguous."""

from pathlib import Path

import yaml
from pydantic import ValidationError

from configs.schema import ExperimentConfig
from storage.hashing import hash_obj, sha256_bytes
from storage.versioning import Frozen


class ConfigError(Exception):
    pass


class LoadedConfig(Frozen):
    config: ExperimentConfig
    config_hash: str  # hash of the validated config (paths as written)
    file_hashes: dict[str, str]  # logical name -> sha256 of referenced file contents
    source_path: str
    base_dir: str

    def resolve(self, relative: str) -> Path:
        return (Path(self.base_dir) / relative).resolve()


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of silently overriding."""


def _construct_mapping(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ConfigError(f"duplicate key {key!r} at line {key_node.start_mark.line + 1}")
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load_yaml(path: Path) -> object:
    with open(path, encoding="utf-8") as fh:
        return yaml.load(fh, Loader=_StrictLoader)  # noqa: S506 - SafeLoader subclass


def load_config(path: str | Path) -> LoadedConfig:
    path = Path(path).resolve()
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    raw = load_yaml(path)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path.name}: top level must be a mapping")
    try:
        config = ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{path.name}: invalid config\n{exc}") from exc

    base = path.parent
    file_hashes: dict[str, str] = {}
    missing: list[str] = []
    for name, rel in config.referenced_paths().items():
        target = (base / rel).resolve()
        if not target.is_file():
            missing.append(f"{name} -> {rel}")
            continue
        file_hashes[name] = sha256_bytes(target.read_bytes())
    if missing:
        raise ConfigError("missing referenced files: " + "; ".join(missing))

    return LoadedConfig(
        config=config,
        config_hash=hash_obj(config),
        file_hashes=file_hashes,
        source_path=str(path),
        base_dir=str(base),
    )
