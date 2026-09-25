import hashlib
import io
import json
from pathlib import Path

import pytest

from scripts import v13_execute_jquants_t1_exclusion_private_read as harness
from scripts import v13_resolve_jquants_t1_exclusion_state as resolver
SENTINEL = "NON_T1_SENTINEL_PRIVATE_IDENTITY"
MEMBERS = [f"{i:04d}" for i in range(300)]
SYNTHETIC_SHA = hashlib.sha256(("\n".join(MEMBERS) + "\n").encode()).hexdigest()


def artifact():
    value = {key: expected for key, expected in resolver.EXPECTED.items()}
    value.update({key: "public-synthetic" for key in resolver.ROOT_KEYS - set(value) -
                  {"block_sizes", "block_hashes", "assignments", "raw_pages", "markets", "products"}})
    value["block_sizes"] = {key: 300 for key in resolver.BLOCK_KEYS}
    value["block_hashes"] = {key: SYNTHETIC_SHA for key in resolver.BLOCK_KEYS}
    value["assignments"] = {key: [SENTINEL] for key in resolver.ASSIGNMENT_KEYS}
    value["assignments"]["T1"] = MEMBERS
    value["raw_pages"] = [{"opaque": SENTINEL}]
    value["markets"] = ["0111", "0112"]
    value["products"] = ["011"]
    return value


def raw(value):
    return json.dumps(value, separators=(",", ":")).encode()


def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(resolver, "T1_SHA", SYNTHETIC_SHA)
    repo = tmp_path / "repo"; repo.mkdir()
    local = tmp_path / "local"; local.mkdir()
    source = local / "stock-analyzer" / "private" / harness.SOURCE_ROOT_NAME / "recovery.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(raw(artifact()))
    return repo, local, source


class Tracked(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.calls = []

    def read(self, size=-1):
        self.calls.append(size)
        return super().read(size)


def test_one_open_receipt_before_remainder_and_safe_outputs(tmp_path, monkeypatch, capsys):
    repo, local, source = setup(tmp_path, monkeypatch)
    opened = []
    receipts = []
    def opener(path):
        assert path == source
        stream = Tracked(source.read_bytes()); opened.append(stream)
        return stream
    def receipt_writer(path, payload):
        assert opened[0].calls == [1]
        receipts.append(path)
        harness._write_once(path, payload)
    report = harness.execute(repo, local, opener=opener, receipt_writer=receipt_writer)
    assert "EXECUTION_RESULT=PASS" in report
    assert len(opened) == 1 and opened[0].calls[0] == 1 and len(opened[0].calls) > 1
    assert len(receipts) == 1
    root = local / "stock-analyzer" / "private" / harness.ROOT_NAME
    state = (root / "t1-exclusion-state.json").read_text()
    receipt = (root / "consumed-receipt.json").read_text()
    assert SENTINEL not in state + receipt + report + capsys.readouterr().out
    assert str(source) not in state + receipt + report
    assert "NON_T1_IDENTITIES_RETAINED=false" in report
    assert "V13_UNIVERSE_SELECTED=false" in report
    assert "NETWORK_REQUESTS=0" in report
    assert root != source.parent and repo not in root.parents
    assert json.loads(state)["t1_membership"] == artifact()["assignments"]["T1"]


def test_receipt_failure_stops_before_remainder_and_is_consumed(tmp_path, monkeypatch):
    repo, local, source = setup(tmp_path, monkeypatch)
    opened = []
    def opener(path):
        stream = Tracked(source.read_bytes()); opened.append(stream); return stream
    def fail(path, payload):
        assert opened[0].calls == [1]
        raise OSError("secret path and identity must not escape")
    report = harness.execute(repo, local, opener=opener, receipt_writer=fail)
    assert opened[0].calls == [1]
    assert "EXECUTION_RESULT=POST_BOUNDARY_FAILURE" in report
    assert "AUTHORIZATION_CONSUMED=true" in report
    assert "AUTHORIZATION_REUSABLE=false" in report
    assert "secret" not in report


@pytest.mark.parametrize("existing", ["consumed-receipt.json", "t1-exclusion-state.json"])
def test_existing_output_blocks_before_open(tmp_path, monkeypatch, existing):
    repo, local, source = setup(tmp_path, monkeypatch)
    root = local / "stock-analyzer" / "private" / harness.ROOT_NAME
    root.mkdir(); (root / existing).write_text("occupied")
    report = harness.execute(repo, local, opener=lambda _: pytest.fail("opened source"))
    assert "EXECUTION_RESULT=PRE_GATE_STOP" in report
    assert "SOURCE_OPENS=0" in report


def test_root_inside_repo_blocks_before_open(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    local = repo / "local"; local.mkdir()
    source = local / "stock-analyzer" / "private" / harness.SOURCE_ROOT_NAME / "recovery.json"
    source.parent.mkdir(parents=True); source.write_bytes(b"synthetic")
    report = harness.execute(repo, local, opener=lambda _: pytest.fail("opened source"))
    assert "EXECUTION_RESULT=PRE_GATE_STOP" in report
    assert "SOURCE_OPENS=0" in report


def test_cli_requires_reviewed_wrapper_gate(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("V13_JQUANTS_T1_WRAPPER_GATE", raising=False)
    monkeypatch.setattr("sys.argv", ["harness", "--repository-root", str(tmp_path)])
    assert harness.main() == 1
    assert "PRE_GATE_WRAPPER_REQUIRED" in capsys.readouterr().out


def test_production_paths_are_narrow_and_pregate_wrapper_is_metadata_only():
    root = Path(__file__).resolve().parents[1]
    resolver_text = (root / "scripts/v13_resolve_jquants_t1_exclusion_state.py").read_text()
    harness_text = (root / "scripts/v13_execute_jquants_t1_exclusion_private_read.py").read_text()
    wrapper = (root / "scripts/run_v13_jquants_t1_exclusion_direct_windows.ps1").read_text()
    assert "urllib" not in resolver_text + harness_text
    assert "json.loads" not in resolver_text + harness_text
    assert "validate_recovery" not in resolver_text + harness_text
    assert 'source.open("rb", buffering=0)' in harness_text
    assert "read_bytes()" not in wrapper
    assert wrapper.index('if (-not $ExecuteReviewedPrivateRead)') < wrapper.index('v13_execute_jquants_t1_exclusion_private_read --repository-root')
    assert harness.APPROVAL_BLOB == "276feb56f417f9d5b931d594598a1d8591330bfd"
    assert resolver.T1_SHA == "262201792183776e3bead4638646ee949c05d35c894c7a4053556befa6230e1d"
