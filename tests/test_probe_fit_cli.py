"""probe-fit command line: --device reaches fit_condition (manager-owned wiring in cli.py)."""
from pathlib import Path

from s1stream import cli
from s1stream.probe import fit


def test_probe_fit_passes_device(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fit, "fit_condition", lambda features, out, **kwargs: calls.append((features, out, kwargs)))
    assert cli.main(["probe-fit", str(tmp_path / "features"), "--out", str(tmp_path / "out"), "--device", "cuda"]) == 0
    assert calls == [(Path(tmp_path / "features"), Path(tmp_path / "out"), {"device": "cuda"})]
