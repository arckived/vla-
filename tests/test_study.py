"""Run with:  python -m pytest tests/ -q     (no GPU or simulator needed)"""
import json
import math
import os
import tempfile

import pytest

from vla_study.paraphrase import restruct, synonym, verbose
from vla_study.patched_eval import TaskRewritingPreprocessor, normalize
from vla_study.results import collect, wilson_interval

T = "pick up the alphabet soup and place it in the basket"


def test_paraphrases_change_wording_but_keep_objects():
    for fn in (synonym, restruct, verbose):
        out = fn(T)
        assert out != T
        assert "alphabet soup" in out and "basket" in out
    assert synonym(T) == "grab the alphabet soup and put it in the basket"  # no put<->place ping-pong


def test_task_rewriter_handles_str_and_batch_and_reports_misses():
    seen = {}
    rw = TaskRewritingPreprocessor(lambda d: seen.update(d) or d, {T + ".": "PARA"})
    rw({"task": "Pick up the alphabet soup and place it in the basket"})
    assert seen["task"] == "PARA"                      # normalization: case + trailing period
    rw({"task": [T, "something else"]})
    assert seen["task"] == ["PARA", "something else"]
    assert rw.hits == 2 and rw.misses == {"something else"}


def test_wilson_interval():
    lo, hi = wilson_interval(80, 100)
    assert 0.70 < lo < 0.72 and 0.86 < hi < 0.88
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_collect_reads_eval_info():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(f"{d}/run1")
        json.dump({"overall": {"pc_success": 75.0, "n_episodes": 40}}, open(f"{d}/run1/eval_info.json", "w"))
        rows = collect(d)
        assert rows[0]["success"] == 75.0 and rows[0]["n"] == 40 and not math.isnan(rows[0]["ci_lo"])


def test_quantize_error_shrinks_with_bits():
    torch = pytest.importorskip("torch")
    from vla_study.quantize import quantize_dequantize
    w = torch.randn(64, 256)
    errs = [((w - quantize_dequantize(w, b)).norm() / w.norm()).item() for b in (4, 8)]
    assert errs[1] < errs[0] < 0.2
    g = ((w - quantize_dequantize(w, 4, group_size=64)).norm() / w.norm()).item()
    assert g < errs[0]                                   # smaller groups -> smaller error


def test_apply_fake_weight_quant_filters_by_name():
    torch = pytest.importorskip("torch")
    from vla_study.quantize import apply_fake_weight_quant
    m = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 4))
    rep = apply_fake_weight_quant(m, 4, include=r"^0$", verbose=False)
    assert rep.n_layers == 1 and rep.est_bytes_after < rep.est_bytes_before
    with pytest.raises(RuntimeError):
        apply_fake_weight_quant(m, 4, include="nope", verbose=False)


def _fake_ckpt(root, step, complete=True):
    d = root / step
    (d / "pretrained_model").mkdir(parents=True)
    if complete:
        (d / "pretrained_model" / "train_config.json").write_text("{}")
        (d / "training_state").mkdir()
    return d


def test_prune_keeps_latest_complete(tmp_path):
    from vla_study.hub_sync import latest_complete, prune, step_dirs
    ck = tmp_path / "train" / "expert_only" / "checkpoints"
    for s in ("002000", "004000", "006000"):
        _fake_ckpt(ck, s)
    _fake_ckpt(ck, "008000", complete=False)       # a save still in progress
    (ck / "last").symlink_to("006000")
    assert latest_complete(ck).name == "006000"
    prune(tmp_path, keep=2)
    assert [d.name for d in step_dirs(ck)] == ["006000", "008000"]
