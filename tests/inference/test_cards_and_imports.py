import json
import subprocess
import sys

import pytest

from weatherai import inference as wi
from weatherai.inference import cards as C


def test_every_card_has_all_levels_and_no_skill_claim():
    for c in C.CARDS.values():
        assert set(C.LEVELS) <= set(c.status)
        assert c.status["independent_hindcast"].state == "no", c.model          # nobody has an independent hindcast yet
        assert c.validated_lead.value is None, c.model                            # technical horizon != validated horizon
        assert c.weights_redistributable in ("yes", "no", "unclear")


def test_specific_card_facts():
    assert C.get("fuxi-s2s").technical_max_lead.value == 42 and C.get("fuxi-s2s").backend == "onnx-official"
    assert C.get("fuxi-s2s").weights_redistributable == "no"                       # CC-BY-NC-ND: obtainable != redistributable
    assert C.get("fuxi-s2s").status["weights_obtainable"].state == "yes"
    assert not C.get("unicm").runnable and C.get("unicm").status["weights_obtainable"].state == "no"
    assert C.get("orca-dl").runnable and "proxy" in C.get("orca-dl").inputs[2].notes.lower()
    assert "NOT an SST forecast" in " ".join(C.get("ace2").limitations)
    assert C.get("neuralgcm").backend == "jax-official"


def test_json_export_roundtrip():
    d = json.loads(C.to_json())
    assert set(d) == {"orca-dl", "fuxi-s2s", "ace2", "neuralgcm", "unicm"}
    assert d["orca-dl"]["status"]["official_parity"]["state"] == "yes"


def test_import_is_light():
    code = "import sys, weatherai, weatherai.inference as wi; wi.card('orca-dl'); " \
           "print(','.join(m for m in ('torch','onnxruntime','jax','xarray') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "", out


def test_backend_selection_errors():
    with pytest.raises(wi.BackendUnavailableError):
        wi.load("orca-dl", backend="onnx")
    with pytest.raises(wi.BackendUnavailableError):
        wi.load("neuralgcm", backend="torch")
    with pytest.raises(wi.WeightsUnavailableError):
        wi.load("unicm")
    with pytest.raises(wi.WeightsUnavailableError):
        wi.load("orca-dl", root="/nonexistent")
    with pytest.raises(wi.WeightsUnavailableError):
        wi.load("fuxi-s2s", path="/nonexistent")


def test_committed_cards_json_is_in_sync():
    import os
    p = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "model_cards.json")
    assert json.load(open(p)) == json.loads(C.to_json()), "regenerate: python -m weatherai.inference cards --json > docs/model_cards.json"
