import pytest
from pydantic import ValidationError

from configs.loader import ConfigError, load_config
from configs.schema import ExperimentConfig
from models.types import ModelSpec, SamplingParams
from storage.hashing import canonical_json, hash_obj, sha256_text
from tests.conftest import base_config_dict


def test_valid_config_loads_and_hashes(config_dir):
    loaded = load_config(config_dir())
    assert loaded.config.experiment_id == "mvp-mock"
    assert len(loaded.config_hash) == 64
    assert set(loaded.file_hashes) == {
        "dataset.manifest",
        "pricing",
        "policy.fixed_escalation@0.1.ladder",
    }
    assert loaded.file_hashes["pricing"] == sha256_text("placeholder: true\n")


def test_config_hash_is_deterministic_and_key_order_independent(config_dir):
    a = load_config(config_dir())
    reordered = dict(reversed(list(base_config_dict().items())))
    b = load_config(config_dir(reordered, name="other.yaml"))
    assert a.config_hash == b.config_hash


def test_config_hash_changes_with_content(config_dir):
    cfg = base_config_dict()
    cfg["seeds"] = [0, 1, 2]
    assert load_config(config_dir()).config_hash != load_config(config_dir(cfg, "b.yaml")).config_hash


def test_schema_version_is_recorded_and_enforced(config_dir):
    assert load_config(config_dir()).config.schema_version == "1.0"
    cfg = base_config_dict() | {"schema_version": "0.9"}
    with pytest.raises(ConfigError):
        load_config(config_dir(cfg))


def test_unknown_fields_rejected(config_dir):
    with pytest.raises(ConfigError):
        load_config(config_dir(base_config_dict() | {"surprise": 1}))


def test_missing_required_field_rejected(config_dir):
    cfg = base_config_dict()
    del cfg["objective_weights"]
    with pytest.raises(ConfigError, match="objective_weights"):
        load_config(config_dir(cfg))


def test_missing_referenced_file_fails_closed(config_dir, tmp_path):
    (tmp_path / "pricing.yaml").unlink()
    with pytest.raises(ConfigError, match="pricing"):
        load_config(config_dir())


def test_duplicate_yaml_keys_rejected(tmp_path):
    p = tmp_path / "dup.yaml"
    p.write_text("experiment_id: a\nexperiment_id: b\n")
    with pytest.raises(ConfigError, match="duplicate key"):
        load_config(p)


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda c: c.update(attacker="nope"), "unknown model"),
        (lambda c: c.update(targets=["mock-attacker"]), "separate"),
        (lambda c: c.update(targets=["mock-target", "mock-target"]), "duplicate target"),
        (lambda c: c["dataset"].update(split="test"), "dev mode"),
        (lambda c: c.update(mode="final"), "final mode"),
        (lambda c: c.update(seeds=[1, 1]), "duplicate seeds"),
        (lambda c: c["evaluators"].update(success_threshold=0), "success_threshold"),
        (lambda c: c["budgets"][0].update(turns=0), "turns"),
    ],
)
def test_cross_field_validation(mutate, msg):
    cfg = base_config_dict()
    mutate(cfg)
    with pytest.raises(ValidationError, match=msg):
        ExperimentConfig.model_validate(cfg)


def test_schemas_are_frozen():
    spec = ModelSpec(provider="mock", model_id="m", version="1", params=SamplingParams(max_tokens=8))
    with pytest.raises(ValidationError):
        spec.model_id = "other"  # type: ignore[misc]
    assert spec.pricing_key == "mock:m:1"


def test_canonical_json_rejects_nan_and_sorts_keys():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})
    assert hash_obj({"b": 1, "a": 2}) == hash_obj({"a": 2, "b": 1})
