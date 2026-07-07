import pandas as pd
import pytest

from mimic import load_paper_dataset, paper_dataset_columns, paper_dataset_names, paper_dataset_registry
from mimic import datasets


def test_paper_dataset_registry_contains_three_manuscript_datasets():
    registry = paper_dataset_registry()

    assert set(paper_dataset_names()) == {"adult", "bank_marketing", "default_credit"}
    assert {"key", "name", "openml_id", "target", "n_rows"}.issubset(registry.columns)


def test_load_paper_dataset_returns_frame_and_mimic_columns(monkeypatch):
    raw = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "workclass": ["Private", "Private", "Gov", "Gov"],
            "class": ["<=50K", "<=50K", ">50K", "<=50K"],
        }
    )

    monkeypatch.setattr(datasets, "fetch_openml", lambda **_kwargs: type("Bunch", (), {"frame": raw})())

    frame, columns = load_paper_dataset("adult", n_rows=3, random_state=0)

    assert len(frame) == 3
    assert frame["label"].isin(["majority", "minority"]).all()
    assert columns["ignore"] == []
    assert "age" in columns["regression"]
    assert {"workclass", "label"}.issubset(columns["classification"])


def test_paper_dataset_columns_default_credit_keeps_coded_fields_categorical():
    frame = pd.DataFrame(
        {
            "ID": [1, 2],
            "LIMIT_BAL": [10000, 20000],
            "AGE": [30, 40],
            "SEX": [1, 2],
            "EDUCATION": [2, 1],
            "MARRIAGE": [1, 2],
            "PAY_0": [0, -1],
            "label": ["majority", "minority"],
        }
    )

    columns = paper_dataset_columns(frame, "credit")

    assert columns["ignore"] == ["ID"]
    assert {"LIMIT_BAL", "AGE"}.issubset(columns["regression"])
    assert {"SEX", "EDUCATION", "MARRIAGE", "PAY_0", "label"}.issubset(columns["classification"])


def test_load_paper_dataset_rejects_unknown_key():
    with pytest.raises(ValueError, match="Unknown paper dataset"):
        load_paper_dataset("missing")
