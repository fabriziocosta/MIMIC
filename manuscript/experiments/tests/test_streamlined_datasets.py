import pandas as pd

from streamlined import datasets
from streamlined.datasets import dataset_registry, feature_roles, load_dataset


def test_dataset_registry_contains_selected_keys():
    registry = dataset_registry()

    assert {"adult", "bank_marketing", "default_credit"}.issubset(set(registry["key"]))
    assert {"key", "name", "openml_id", "target", "n_rows", "status"}.issubset(registry.columns)
    assert registry.set_index("key").loc["adult", "n_rows"] == 48842


def test_load_dataset_normalizes_adult_target(monkeypatch):
    frame = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "workclass": ["Private", "Private", "Gov", "Gov"],
            "class": ["<=50K", "<=50K", "<=50K", ">50K"],
        }
    )

    monkeypatch.setattr(datasets, "fetch_openml", lambda **kwargs: type("Bunch", (), {"frame": frame})())

    loaded = load_dataset("adult")

    assert loaded["label"].tolist() == ["majority", "majority", "majority", "minority"]


def test_default_credit_roles_treat_coded_fields_as_categorical():
    frame = pd.DataFrame(
        {
            "LIMIT_BAL": [1, 2],
            "AGE": [30, 40],
            "SEX": [1, 2],
            "EDUCATION": [2, 1],
            "MARRIAGE": [1, 2],
            "PAY_0": [0, -1],
            "label": ["majority", "minority"],
        }
    )

    roles = feature_roles(frame, "default_credit")

    assert "LIMIT_BAL" in roles["numeric"]
    assert "AGE" in roles["numeric"]
    assert {"SEX", "EDUCATION", "MARRIAGE", "PAY_0"}.issubset(set(roles["categorical"]))
