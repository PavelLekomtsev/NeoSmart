"""Barrier SQLite layer — whitelist CRUD, access log, parking sessions.

Every barrier decision is read through this module: the state machine
asks ``is_plate_allowed(text)`` and writes ``log_access`` / ``start_session``
as side-effects. Errors here silently flip access-control outcomes, so
the core CRUD paths and the session lifetime must stay pinned.

Each test gets its own SQLite file under ``tmp_path`` — the runtime code
path opens a fresh connection per operation, so parallel test files
don't share state.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

# barrier_db.py lives under SmartParking/web_app/ which is NOT on the
# install path; add it explicitly so `import barrier_db` works in tests.
WEB_APP = Path(__file__).resolve().parent.parent / "SmartParking" / "web_app"
if str(WEB_APP) not in sys.path:
    sys.path.insert(0, str(WEB_APP))

from barrier_db import BarrierDatabase  # noqa: E402


@pytest.fixture
def db(tmp_path: Path) -> BarrierDatabase:
    """Isolated BarrierDatabase backed by a file under tmp_path.

    We pass a nested path so the `db_dir.mkdir(...)` branch in __init__ is
    exercised too — matches what the app does on first run."""
    return BarrierDatabase(db_path=str(tmp_path / "nested" / "barrier.db"))


# --- Allowed-plates CRUD --------------------------------------------------


def test_add_and_query_plate_is_case_insensitive(db: BarrierDatabase) -> None:
    # Russian plates use Cyrillic O/A/B glyphs that render as Latin O/A/B
    # but have different codepoints; the DB must treat them as-received,
    # not as their Latin look-alikes.
    assert db.add_plate("О123АВ77", owner="Ivanov", description="sedan")  # noqa: RUF001
    # DB normalises to upper on write AND lookup, so any casing on the
    # client side resolves to the same row.
    assert db.is_plate_allowed("о123ав77")  # noqa: RUF001
    assert db.is_plate_allowed("О123АВ77")  # noqa: RUF001
    info = db.get_plate_info("О123АВ77")  # noqa: RUF001
    assert info is not None
    assert info["owner_name"] == "Ivanov"
    assert info["vehicle_description"] == "sedan"


def test_remove_plate_returns_true_only_when_something_was_removed(db: BarrierDatabase) -> None:
    db.add_plate("AAA111")
    assert db.remove_plate("aaa111") is True
    assert db.is_plate_allowed("AAA111") is False
    assert db.remove_plate("AAA111") is False  # idempotent


def test_update_plate_changes_only_fields_passed(db: BarrierDatabase) -> None:
    db.add_plate("BBB222", owner="Old", description="hatchback")

    assert db.update_plate("BBB222", owner="New") is True
    info = db.get_plate_info("BBB222")
    assert info is not None
    assert info["owner_name"] == "New"
    assert info["vehicle_description"] == "hatchback"  # unchanged

    # Updating a non-existent plate returns False
    assert db.update_plate("ZZZ999", owner="Nobody") is False


def test_add_plate_replaces_existing(db: BarrierDatabase) -> None:
    """INSERT OR REPLACE semantics — re-adding a plate overwrites owner info
    rather than raising a UNIQUE constraint error."""
    db.add_plate("CCC333", owner="First")
    db.add_plate("CCC333", owner="Second")
    info = db.get_plate_info("CCC333")
    assert info is not None
    assert info["owner_name"] == "Second"


def test_get_all_plates_returns_every_allowed_row(db: BarrierDatabase) -> None:
    db.add_plate("AAA111")
    db.add_plate("BBB222")
    numbers = {row["plate_number"] for row in db.get_all_plates()}
    assert {"AAA111", "BBB222"}.issubset(numbers)


# --- Access log -----------------------------------------------------------


def test_log_access_and_read_back(db: BarrierDatabase) -> None:
    log_id = db.log_access("entry", "O123AB77", 0.87, "granted")
    assert log_id > 0

    recent = db.get_recent_log(limit=10)
    assert len(recent) == 1
    entry = recent[0]
    assert entry["barrier_id"] == "entry"
    assert entry["plate_number"] == "O123AB77"
    assert entry["access_result"] == "granted"
    assert entry["confidence"] == pytest.approx(0.87)


def test_get_recent_log_respects_barrier_filter(db: BarrierDatabase) -> None:
    db.log_access("entry", "A", 0.9, "granted")
    db.log_access("exit", "A", 0.9, "granted")
    db.log_access("entry", "B", 0.9, "denied")

    entry_only = db.get_recent_log(barrier_id="entry")
    assert {row["barrier_id"] for row in entry_only} == {"entry"}
    assert len(entry_only) == 2


def test_today_stats_counts_by_access_result(db: BarrierDatabase) -> None:
    db.log_access("entry", "A", 0.9, "granted")
    db.log_access("entry", "B", 0.9, "granted")
    db.log_access("entry", "C", 0.7, "denied")
    db.log_access("exit", "A", 0.9, "granted")

    stats = db.get_today_stats()
    assert stats["entries"] == 2
    assert stats["exits"] == 1
    assert stats["denied"] == 1
    assert stats["currently_inside"] == 0  # no sessions yet


# --- Parking sessions -----------------------------------------------------


def test_session_lifecycle_computes_duration(db: BarrierDatabase) -> None:
    db.add_plate("O123AB77")
    db.start_session("O123AB77")

    # Wait long enough that the minute-resolution duration comes out > 0.
    time.sleep(1.1)

    result = db.end_session("O123AB77")
    assert result is not None
    assert result["plate"] == "O123AB77"
    assert result["duration_minutes"] >= 0.0
    assert db.get_active_sessions() == []


def test_starting_a_second_session_closes_the_first(db: BarrierDatabase) -> None:
    """Defensive behaviour documented in start_session: if the plate is
    already inside (stuck session), the old one is ended before the new
    one starts — otherwise ``currently_inside`` would double-count."""
    db.start_session("AAA111")
    db.start_session("AAA111")
    active = db.get_active_sessions()
    assert len(active) == 1


def test_end_session_with_no_active_returns_none(db: BarrierDatabase) -> None:
    assert db.end_session("NEVER-ENTERED") is None


def test_today_stats_currently_inside_tracks_active_sessions(db: BarrierDatabase) -> None:
    db.start_session("AAA111")
    db.start_session("BBB222")
    assert db.get_today_stats()["currently_inside"] == 2
    db.end_session("AAA111")
    assert db.get_today_stats()["currently_inside"] == 1


# --- Seeding and runtime reset --------------------------------------------


def test_seed_from_json_populates_empty_table(tmp_path: Path) -> None:
    """``_seed_from_file`` is idempotent on empty/non-empty DBs:
    fills an empty table, and refuses to re-seed if anything is already
    there (so repeated restarts don't duplicate rows)."""
    # Start with a DB that does NOT auto-seed from the real web_app data.
    # Wipe the auto-seeded rows, then seed from our fake JSON.
    db = BarrierDatabase(db_path=str(tmp_path / "seeded.db"))
    conn = db._get_conn()
    try:
        conn.execute("DELETE FROM allowed_plates")
        conn.commit()
    finally:
        conn.close()

    seed = [
        {"plate_number": "SEED001", "owner_name": "Alice", "vehicle_description": "a"},
        {"plate_number": "SEED002", "owner_name": "Bob",   "vehicle_description": "b"},
    ]
    seed_file = tmp_path / "seed_plates.json"
    seed_file.write_text(json.dumps(seed), encoding="utf-8")

    db._seed_from_file(seed_file)
    assert db.is_plate_allowed("SEED001")
    assert db.is_plate_allowed("SEED002")

    # Second call must be a no-op — the early-return guards against
    # re-inserting seed data over curated runtime additions.
    db.remove_plate("SEED001")
    db._seed_from_file(seed_file)
    assert db.is_plate_allowed("SEED001") is False


def test_reset_runtime_state_clears_logs_but_keeps_whitelist(db: BarrierDatabase) -> None:
    db.add_plate("KEEP001", owner="keeper")
    db.log_access("entry", "KEEP001", 0.9, "granted")
    db.start_session("KEEP001")

    db.reset_runtime_state()

    # Runtime tables are wiped
    assert db.get_recent_log() == []
    assert db.get_active_sessions() == []
    # Whitelist survives — seed data and curated plates must not be lost
    # across app restarts.
    assert db.is_plate_allowed("KEEP001")
