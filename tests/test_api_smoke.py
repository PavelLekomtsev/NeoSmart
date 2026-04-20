"""FastAPI smoke tests — plate management and stats endpoints.

Boots the real FastAPI app via ``TestClient`` but stubs the two heavy
module-level globals (``detector``, ``barrier_db``) so the test never
loads YOLO or touches the real SQLite file.

Using TestClient without the context-manager form skips the lifespan
hook — that's exactly what we want, since the lifespan is what pulls in
the plate recognizer and YOLO weights. The stub globals fill the same
contract the lifespan would have set up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# main.py is a standalone script under SmartParking/web_app/ — add it
# to sys.path so `import main` resolves correctly.
WEB_APP = Path(__file__).resolve().parent.parent / "SmartParking" / "web_app"
if str(WEB_APP) not in sys.path:
    sys.path.insert(0, str(WEB_APP))


class FakeDetector:
    """Minimal stand-in for ParkingDetector — only ``get_stats`` is used
    by /api/stats, and the shape of its return must match the real
    detector's dict so the aggregate math adds up correctly."""

    def get_stats(self, camera_id: str) -> dict:
        # Mimic the schema real ``ParkingDetector.get_stats`` returns.
        return {
            "total_spaces": 10,
            "occupied": 3,
            "available": 7,
            "cars_detected": 4,
            "wrong_count": 0,
        }


@pytest.fixture
def client(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    """Build a TestClient with stubbed detector + isolated barrier DB.

    Import is inside the fixture so any side effects of `import main`
    happen after pytest has set up its environment — and so other test
    files that don't need the web app don't pay the import cost.
    """
    import main as main_module
    from barrier_db import BarrierDatabase

    # Isolated SQLite so the suite doesn't mutate the real web_app DB.
    db = BarrierDatabase(db_path=str(tmp_path / "smoke.db"))
    # Auto-seed fired in __init__; strip it so tests start from a known state.
    conn = db._get_conn()
    try:
        conn.execute("DELETE FROM allowed_plates")
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(main_module, "detector", FakeDetector())
    monkeypatch.setattr(main_module, "barrier_db", db)
    monkeypatch.setattr(main_module, "barrier_controllers", {})

    return TestClient(main_module.app)


# --- /api/stats ----------------------------------------------------------


def test_stats_returns_per_camera_and_aggregate(client: TestClient) -> None:
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()

    # Every configured camera must appear in the response, plus aggregate.
    import main as main_module

    for cam_id in main_module.CAMERA_IDS:
        assert cam_id in data, f"missing camera: {cam_id}"

    agg = data["aggregate"]
    assert agg["total_spaces"] == 10 * len(main_module.CAMERA_IDS)
    assert agg["occupied"] == 3 * len(main_module.CAMERA_IDS)
    assert agg["available"] == agg["total_spaces"] - agg["occupied"]
    # Aggregate car count is intentionally camera1-only (dedup contract).
    assert agg["cars_detected"] == 4


# --- /api/barrier/plates CRUD -------------------------------------------


def test_plate_lifecycle_add_list_delete(client: TestClient) -> None:
    # Initially empty (we wiped the seed in the fixture).
    resp = client.get("/api/barrier/plates")
    assert resp.status_code == 200
    assert resp.json() == []

    # Add one plate.
    resp = client.post(
        "/api/barrier/plates",
        json={"plate_number": "a123bc77", "owner_name": "Smoke", "vehicle_description": "test"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    # Backend normalises to uppercase before storing.
    assert body["plate"] == "A123BC77"

    # List now contains it.
    plates = client.get("/api/barrier/plates").json()
    assert len(plates) == 1
    assert plates[0]["plate_number"] == "A123BC77"
    assert plates[0]["owner_name"] == "Smoke"

    # Delete it — 200 + status ok.
    resp = client.delete("/api/barrier/plates/A123BC77")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    # Second delete is idempotent from HTTP's POV but reports not_found.
    resp = client.delete("/api/barrier/plates/A123BC77")
    assert resp.status_code == 200
    assert resp.json()["status"] == "not_found"


def test_add_plate_rejects_empty_number(client: TestClient) -> None:
    resp = client.post("/api/barrier/plates", json={"plate_number": "   "})
    assert resp.status_code == 400
    assert "error" in resp.json()


# --- /api/barrier/{id} — unknown barrier -------------------------------


def test_unknown_barrier_id_returns_404(client: TestClient) -> None:
    """barrier_controllers is empty in the fixture — any lookup should 404.

    Pins the contract: UE5 must not get a stale state dict if the
    configured barrier isn't up yet."""
    resp = client.get("/api/barrier/entry")
    assert resp.status_code == 404
    assert "error" in resp.json()


def test_barrier_log_with_db_returns_list(client: TestClient) -> None:
    """Even with no entries, /api/barrier/log returns [] rather than
    erroring — dashboard JS assumes a JSON array here."""
    resp = client.get("/api/barrier/log")
    assert resp.status_code == 200
    assert resp.json() == []
