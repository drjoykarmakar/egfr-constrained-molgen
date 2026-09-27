from pathlib import Path

import pandas as pd

from src.compare import (
    ComparisonConfig,
    cross_set_summary,
    prepare_random_library_sample,
    summarize_group,
)
from src.novelty import FingerprintConfig


def test_random_library_sample_is_deterministic_and_excludes_benchmark(tmp_path: Path):
    path = tmp_path / "lib.smi"
    path.write_text(
        "# canonical_smiles id\nCCO A\nCCN B\nCCC C\nCCCl D\nCCBr E\n",
        encoding="utf-8",
    )
    kwargs = dict(benchmark_smiles={"CCO"}, sample_size=3, seed=7)
    a = prepare_random_library_sample(path, **kwargs)
    b = prepare_random_library_sample(path, **kwargs)
    assert a["standardized_smiles"].tolist() == b["standardized_smiles"].tolist()
    assert "CCO" not in set(a["standardized_smiles"])


def test_group_summary_reports_strict_novelty_fraction():
    frame = pd.DataFrame(
        {
            "standardized_smiles": ["CCO", "CCN"],
            "mw": [46.0, 45.0],
            "clogp": [-0.1, -0.2],
            "qed": [0.4, 0.5],
            "nearest_training_active_tanimoto": [0.49, 0.50],
        }
    )
    out = summarize_group(
        "x", frame, seed=1, fingerprint=FingerprintConfig(radius=2, n_bits=128)
    )
    assert out["fraction_training_tanimoto_lt_0_5"] == 0.5


def test_cross_set_summary_detects_exact_overlap():
    q = pd.DataFrame({"standardized_smiles": ["CCO", "CCN"]})
    r = pd.DataFrame({"standardized_smiles": ["CCO", "CCC"]})
    out = cross_set_summary(
        "q", q, "r", r, fingerprint=FingerprintConfig(radius=2, n_bits=128)
    )
    assert out["exact_smiles_overlap"] == 1
    assert out["max_cross_set_tanimoto"] == 1.0
