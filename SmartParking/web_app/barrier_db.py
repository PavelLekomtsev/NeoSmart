"""
Barrier Database Module
SQLite database for license plate whitelist, access logs, and parking sessions.
"""

import sqlite3
import json
import time
from datetime import datetime
from pathlib import Path


class BarrierDatabase:
    """SQLite database for barrier access control."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path(__file__).parent / "data" / "barrier.db")

        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self._init_db()

        # Seed default plates if table is empty
        seed_file = Path(__file__).parent / "data" / "seed_plates.json"
        if seed_file.exists():
            self._seed_from_file(seed_file)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS allowed_plates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    owner_name TEXT DEFAULT '',
                    vehicle_description TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active INTEGER DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    barrier_id TEXT NOT NULL,
                    plate_number TEXT,
                    confidence REAL,
                    access_result TEXT NOT NULL,
                    frame_snapshot_path TEXT
                );

                CREATE TABLE IF NOT EXISTS parking_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP,
                    duration_minutes REAL,
                    status TEXT DEFAULT 'active'
                );

                CREATE INDEX IF NOT EXISTS idx_access_log_timestamp
                    ON access_log(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_sessions_status
                    ON parking_sessions(status);
                CREATE INDEX IF NOT EXISTS idx_sessions_plate
                    ON parking_sessions(plate_number, status);
            """)
            conn.commit()
        finally:
            conn.close()

    def reset_runtime_state(self):
        """Clear per-run counters: access log + active parking sessions.

        Called at app startup so the dashboard's 'Today's Statistics' (entries,
        denied, inside) starts at zero each run instead of accumulating across
        restarts. The allowed_plates whitelist is preserved."""
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM access_log")
            conn.execute("DELETE FROM parking_sessions")
            conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('access_log', 'parking_sessions')")
            conn.commit()
            print("[BarrierDB] Runtime stats reset (access_log + parking_sessions cleared)")
        finally:
            conn.close()

    def _seed_from_file(self, seed_file: Path):
        """Seed allowed plates from JSON file if table is empty."""
        conn = self._get_conn()
        try:
            count = conn.execute("SELECT COUNT(*) FROM allowed_plates").fetchone()[0]
            if count > 0:
                return

            with open(seed_file, 'r', encoding='utf-8') as f:
                plates = json.load(f)

            for entry in plates:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO allowed_plates (plate_number, owner_name, vehicle_description) "
                        "VALUES (?, ?, ?)",
                        (entry["plate_number"],
                         entry.get("owner_name", ""),
                         entry.get("vehicle_description", ""))
                    )
                except sqlite3.IntegrityError:
                    pass

            conn.commit()
            print(f"[BarrierDB] Seeded {len(plates)} plates from {seed_file.name}")
        finally:
            conn.close()

    # --- Plate Management ---

    def is_plate_allowed(self, plate: str) -> bool:
        """Check if a plate number is in the allowed list."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT id FROM allowed_plates WHERE plate_number = ? COLLATE NOCASE AND is_active = 1",
                (plate.strip().upper(),)
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def add_plate(self, plate: str, owner: str = "", description: str = "") -> bool:
        """Add a plate to the allowed list. Returns True on success."""
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO allowed_plates (plate_number, owner_name, vehicle_description, is_active) "
                "VALUES (?, ?, ?, 1)",
                (plate.strip().upper(), owner, description)
            )
            conn.commit()
            return True
        except sqlite3.Error:
            return False
        finally:
            conn.close()

    def update_plate(self, plate: str, owner: str = None,
                     description: str = None) -> bool:
        """Update owner/description for an existing plate.
        Pass None to leave a field unchanged. Returns True if a row was updated."""
        sets = []
        params = []
        if owner is not None:
            sets.append("owner_name = ?")
            params.append(owner)
        if description is not None:
            sets.append("vehicle_description = ?")
            params.append(description)
        if not sets:
            return False
        params.append(plate.strip().upper())
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                f"UPDATE allowed_plates SET {', '.join(sets)} "
                f"WHERE plate_number = ? COLLATE NOCASE",
                params,
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def get_plate_info(self, plate: str) -> dict | None:
        """Return {plate_number, owner_name, vehicle_description} for a plate,
        or None if the plate is not whitelisted."""
        if not plate:
            return None
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT plate_number, owner_name, vehicle_description "
                "FROM allowed_plates WHERE plate_number = ? COLLATE NOCASE",
                (plate.strip().upper(),),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def remove_plate(self, plate: str) -> bool:
        """Remove a plate from the allowed list."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM allowed_plates WHERE plate_number = ? COLLATE NOCASE",
                (plate.strip().upper(),)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def get_all_plates(self) -> list[dict]:
        """Get all allowed plates."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT plate_number, owner_name, vehicle_description, created_at, is_active "
                "FROM allowed_plates ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # --- Access Log ---

    def log_access(self, barrier_id: str, plate: str, confidence: float,
                   result: str, frame_path: str = None) -> int:
        """Log an access event. Returns the log entry ID."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "INSERT INTO access_log (barrier_id, plate_number, confidence, access_result, frame_snapshot_path) "
                "VALUES (?, ?, ?, ?, ?)",
                (barrier_id, plate, confidence, result, frame_path)
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_recent_log(self, limit: int = 50, barrier_id: str = None) -> list[dict]:
        """Get recent access log entries."""
        conn = self._get_conn()
        try:
            if barrier_id:
                rows = conn.execute(
                    "SELECT id, timestamp, barrier_id, plate_number, confidence, access_result "
                    "FROM access_log WHERE barrier_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (barrier_id, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, timestamp, barrier_id, plate_number, confidence, access_result "
                    "FROM access_log ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_today_stats(self) -> dict:
        """Get today's access statistics."""
        conn = self._get_conn()
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            row = conn.execute(
                "SELECT "
                "  COUNT(CASE WHEN barrier_id = 'entry' AND access_result = 'granted' THEN 1 END) as entries, "
                "  COUNT(CASE WHEN barrier_id = 'exit' AND access_result = 'granted' THEN 1 END) as exits, "
                "  COUNT(CASE WHEN access_result = 'denied' THEN 1 END) as denied "
                "FROM access_log WHERE DATE(timestamp) = ?",
                (today,)
            ).fetchone()
            active = conn.execute(
                "SELECT COUNT(*) FROM parking_sessions WHERE status = 'active'"
            ).fetchone()[0]
            return {
                "entries": row["entries"],
                "exits": row["exits"],
                "denied": row["denied"],
                "currently_inside": active,
            }
        finally:
            conn.close()

    # --- Parking Sessions ---

    def start_session(self, plate: str) -> int:
        """Start a new parking session for a plate. Returns session ID."""
        conn = self._get_conn()
        try:
            # Close any existing active session for this plate (edge case)
            self._end_session_internal(conn, plate)
            cursor = conn.execute(
                "INSERT INTO parking_sessions (plate_number, entry_time, status) "
                "VALUES (?, CURRENT_TIMESTAMP, 'active')",
                (plate.strip().upper(),)
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def end_session(self, plate: str) -> dict | None:
        """End active session for a plate. Returns session info with duration."""
        conn = self._get_conn()
        try:
            result = self._end_session_internal(conn, plate)
            conn.commit()
            return result
        finally:
            conn.close()

    def _end_session_internal(self, conn: sqlite3.Connection, plate: str) -> dict | None:
        """Internal: end session within existing connection."""
        row = conn.execute(
            "SELECT id, entry_time FROM parking_sessions "
            "WHERE plate_number = ? COLLATE NOCASE AND status = 'active' "
            "ORDER BY entry_time DESC LIMIT 1",
            (plate.strip().upper(),)
        ).fetchone()

        if row is None:
            return None

        entry_time = datetime.fromisoformat(row["entry_time"])
        duration = (datetime.now() - entry_time).total_seconds() / 60.0

        conn.execute(
            "UPDATE parking_sessions SET exit_time = CURRENT_TIMESTAMP, "
            "duration_minutes = ?, status = 'completed' WHERE id = ?",
            (round(duration, 1), row["id"])
        )

        return {
            "session_id": row["id"],
            "plate": plate.strip().upper(),
            "entry_time": row["entry_time"],
            "duration_minutes": round(duration, 1),
        }

    def get_active_sessions(self) -> list[dict]:
        """Get all currently active parking sessions."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT id, plate_number, entry_time FROM parking_sessions "
                "WHERE status = 'active' ORDER BY entry_time DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
