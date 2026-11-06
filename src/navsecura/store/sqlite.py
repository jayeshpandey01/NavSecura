"""SQLite WAL event persistence engine for visit memory.

Implements structured transactional storage adhering strictly to Section 4.3
of docs/plan.md, enabling ACID event durability, WAL concurrency, relational
indexing, and forensic evidence tracking across edge deployments.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Union

from navsecura.events.models import Visit

logger = logging.getLogger(__name__)


class SQLiteEventStore:
    """Production SQLite storage engine for visit events and evidence catalogs.

    Configured with Write-Ahead Logging (WAL), synchronous=NORMAL, and busy_timeout=5000
    for high edge throughput and zero-lock concurrency. Provides typed insertion,
    relational filtering, summary metrics, and bi-directional JSON/CSV serialization.
    """

    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS visits (
        visit_id            TEXT PRIMARY KEY,
        camera_id           TEXT NOT NULL,
        roi_id              TEXT NOT NULL,
        track_id            INTEGER NOT NULL,
        merged_track_ids    TEXT NOT NULL,
        merged_tracks_json  TEXT NOT NULL,
        parent_visit_id     TEXT,
        person_id           TEXT,
        entered_at          TEXT,
        exited_at           TEXT,
        t_enter             REAL NOT NULL,
        t_exit              REAL NOT NULL,
        video_t_enter_s     REAL NOT NULL,
        video_t_exit_s      REAL NOT NULL,
        entry_frame_idx     INTEGER NOT NULL,
        exit_frame_idx      INTEGER NOT NULL,
        frame_enter         INTEGER NOT NULL,
        frame_exit          INTEGER NOT NULL,
        duration_s          REAL NOT NULL,
        mean_conf           REAL,
        peak_conf           REAL NOT NULL,
        peak_confidence     REAL NOT NULL,
        close_reason        TEXT NOT NULL,
        snapshot_entry      TEXT,
        snapshot_peak       TEXT,
        clip_path           TEXT,
        config_hash         TEXT,
        model_version       TEXT,
        created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    INDEXES_SQL = """
    CREATE INDEX IF NOT EXISTS idx_visits_lookup ON visits (camera_id, roi_id, t_enter);
    CREATE INDEX IF NOT EXISTS idx_visits_person ON visits (person_id);
    CREATE INDEX IF NOT EXISTS idx_visits_time ON visits (camera_id, roi_id, video_t_enter_s);
    CREATE INDEX IF NOT EXISTS idx_visits_track ON visits (roi_id, track_id);
    CREATE INDEX IF NOT EXISTS idx_visits_duration ON visits (duration_s);
    """

    def __init__(self, db_path: Union[str, Path] = "outputs/database/events.db") -> None:
        """Initialize SQLite event store and execute schema migration.

        Args:
            db_path: Path to the SQLite database file.
        """
        p = Path(db_path)
        # Transparent backward-compatibility redirect if legacy outputs/events.db is requested
        if str(p) == "outputs/events.db" and not p.exists() and Path("outputs/database/events.db").exists():
            p = Path("outputs/database/events.db")
        self.db_path = p
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
        )
        self.conn.row_factory = sqlite3.Row

        # Optimize for edge performance and durability
        self._configure_pragmas()
        self._initialize_schema()

    def _configure_pragmas(self) -> None:
        """Enable Write-Ahead Logging (WAL) and optimized pragmas."""
        with self.conn:
            self.conn.execute("PRAGMA journal_mode = WAL;")
            self.conn.execute("PRAGMA synchronous = NORMAL;")
            self.conn.execute("PRAGMA foreign_keys = ON;")
            self.conn.execute("PRAGMA busy_timeout = 5000;")

    def _initialize_schema(self) -> None:
        """Create tables and performance indexes with non-destructive column migration."""
        with self.conn:
            self.conn.executescript(self.CREATE_TABLE_SQL)

            # Ensure all columns exist in case of schema evolution on existing database
            cursor = self.conn.execute("PRAGMA table_info(visits);")
            existing_cols = {row[1] for row in cursor.fetchall()}

            col_defs = {
                "t_enter": "REAL DEFAULT 0.0",
                "t_exit": "REAL DEFAULT 0.0",
                "entry_frame_idx": "INTEGER DEFAULT 0",
                "exit_frame_idx": "INTEGER DEFAULT 0",
                "merged_tracks_json": "TEXT DEFAULT '[]'",
                "peak_confidence": "REAL DEFAULT 0.0",
                "person_id": "TEXT",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            }
            for col_name, col_def in col_defs.items():
                if col_name not in existing_cols:
                    try:
                        self.conn.execute(f"ALTER TABLE visits ADD COLUMN {col_name} {col_def};")
                    except Exception:
                        pass

            # Update existing rows if t_enter is 0.0
            try:
                self.conn.execute(
                    "UPDATE visits SET t_enter = video_t_enter_s, t_exit = video_t_exit_s "
                    "WHERE t_enter = 0.0 AND video_t_enter_s > 0.0;"
                )
                self.conn.execute(
                    "UPDATE visits SET entry_frame_idx = frame_enter, exit_frame_idx = frame_exit "
                    "WHERE entry_frame_idx = 0 AND frame_enter > 0;"
                )
                self.conn.execute(
                    "UPDATE visits SET peak_confidence = peak_conf "
                    "WHERE peak_confidence = 0.0 AND peak_conf > 0.0;"
                )
                self.conn.execute(
                    "UPDATE visits SET merged_tracks_json = merged_track_ids "
                    "WHERE merged_tracks_json = '[]' AND merged_track_ids IS NOT NULL;"
                )
            except Exception:
                pass

            # Create indexes now that all columns are present
            self.conn.executescript(self.INDEXES_SQL)

        logger.debug("Initialized SQLiteEventStore schema at %s", self.db_path)

    def __enter__(self) -> "SQLiteEventStore":
        """Enter runtime context."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit runtime context and close database connection."""
        self.close()

    def close(self) -> None:
        """Close SQLite connection safely."""
        try:
            self.conn.close()
            logger.debug("Closed SQLiteEventStore connection: %s", self.db_path)
        except Exception as exc:
            logger.warning("Error closing SQLiteEventStore: %s", exc)

    def insert_visit(
        self,
        visit: Visit,
        snapshot_entry: Optional[str] = None,
        snapshot_peak: Optional[str] = None,
        clip_path: Optional[str] = None,
        config_hash: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> str:
        """Insert or replace a comprehensive visit event record into SQLite.

        Args:
            visit: Strongly typed Visit model instance.
            snapshot_entry: Optional override for entry snapshot filepath.
            snapshot_peak: Optional override for peak confidence snapshot filepath.
            clip_path: Optional override for evidence video clip filepath.
            config_hash: Optional override for pipeline configuration SHA-256 hash.
            model_version: Optional override for detector/tracker version tag.

        Returns:
            UUID string of the inserted visit.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        entered_at = visit.entered_at or now_iso
        exited_at = visit.exited_at or now_iso

        merged_json = json.dumps(visit.merged_track_ids or [visit.track_id])
        snap_entry = snapshot_entry or visit.snapshot_entry
        snap_peak = snapshot_peak or visit.snapshot_peak
        clip = clip_path or visit.clip_path
        cfg_hash = config_hash or "default_hash"
        mod_ver = model_version or "yolo11n_bytetrack"
        peak_conf_val = float(visit.peak_conf) if visit.peak_conf else 0.0

        query = """
        INSERT OR REPLACE INTO visits (
            visit_id, camera_id, roi_id, track_id, merged_track_ids, merged_tracks_json,
            parent_visit_id, person_id, entered_at, exited_at, t_enter, t_exit,
            video_t_enter_s, video_t_exit_s, entry_frame_idx, exit_frame_idx,
            frame_enter, frame_exit, duration_s, mean_conf, peak_conf, peak_confidence,
            close_reason, snapshot_entry, snapshot_peak, clip_path, config_hash, model_version
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        );
        """

        with self.conn:
            self.conn.execute(
                query,
                (
                    visit.visit_id,
                    visit.camera_id,
                    visit.roi_id,
                    visit.track_id,
                    merged_json,
                    merged_json,
                    visit.parent_visit_id,
                    visit.person_id or f"P{visit.track_id}",
                    entered_at,
                    exited_at,
                    visit.video_t_enter_s,
                    visit.video_t_exit_s,
                    visit.video_t_enter_s,
                    visit.video_t_exit_s,
                    visit.frame_enter,
                    visit.frame_exit,
                    visit.frame_enter,
                    visit.frame_exit,
                    visit.duration_s,
                    visit.mean_conf,
                    peak_conf_val,
                    peak_conf_val,
                    visit.close_reason,
                    snap_entry,
                    snap_peak,
                    clip,
                    cfg_hash,
                    mod_ver,
                ),
            )

        logger.debug("Persisted visit %s (track %d) to SQLite", visit.visit_id[:8], visit.track_id)
        return visit.visit_id

    def get_visit(self, visit_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific visit record by exact UUID or prefix.

        Args:
            visit_id: Full UUID or initial prefix string.

        Returns:
            Dictionary of visit attributes if found; None otherwise.
        """
        if len(visit_id) < 36:
            cursor = self.conn.execute(
                "SELECT * FROM visits WHERE visit_id LIKE ? LIMIT 1;",
                (f"{visit_id}%",),
            )
        else:
            cursor = self.conn.execute(
                "SELECT * FROM visits WHERE visit_id = ? LIMIT 1;",
                (visit_id,),
            )

        row = cursor.fetchone()
        if row is None:
            return None

        record = dict(row)
        if record.get("merged_track_ids"):
            try:
                record["merged_track_ids"] = json.loads(record["merged_track_ids"])
            except Exception:
                pass
        return record

    def list_visits(
        self,
        camera_id: Optional[str] = None,
        roi_id: Optional[str] = None,
        min_duration_s: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Query stored visits with optional filters, ordered by entry time.

        Args:
            camera_id: Optional camera identifier filter.
            roi_id: Optional ROI polygon filter.
            min_duration_s: Optional minimum dwell duration threshold.

        Returns:
            List of dictionary visit records ordered by t_enter ASC.
        """
        query = "SELECT * FROM visits WHERE 1=1"
        params: List[Any] = []

        if camera_id is not None:
            query += " AND camera_id = ?"
            params.append(camera_id)

        if roi_id is not None:
            query += " AND roi_id = ?"
            params.append(roi_id)

        if min_duration_s is not None:
            query += " AND duration_s >= ?"
            params.append(float(min_duration_s))

        query += " ORDER BY video_t_enter_s ASC;"

        cursor = self.conn.execute(query, params)
        rows = cursor.fetchall()

        results: List[Dict[str, Any]] = []
        for r in rows:
            rec = dict(r)
            if rec.get("merged_track_ids"):
                try:
                    rec["merged_track_ids"] = json.loads(rec["merged_track_ids"])
                except Exception:
                    pass
            results.append(rec)

        return results

    def get_visit_count(
        self,
        camera_id: Optional[str] = None,
        roi_id: Optional[str] = None,
    ) -> int:
        """Count visits matching optional filtering criteria.

        Args:
            camera_id: Optional camera identifier filter.
            roi_id: Optional ROI polygon filter.

        Returns:
            Total integer count.
        """
        query = "SELECT COUNT(*) FROM visits WHERE 1=1"
        params: List[Any] = []

        if camera_id is not None:
            query += " AND camera_id = ?"
            params.append(camera_id)

        if roi_id is not None:
            query += " AND roi_id = ?"
            params.append(roi_id)

        cursor = self.conn.execute(query, params)
        return int(cursor.fetchone()[0])

    def get_summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics for the stored visit history.

        Returns:
            Dictionary containing total_visits, mean_duration_s, min_duration_s,
            max_duration_s, total_raw_alerts_collapsed, and alert_reduction_ratio.
        """
        cursor = self.conn.execute(
            """
            SELECT
                COUNT(*),
                AVG(duration_s),
                MIN(duration_s),
                MAX(duration_s),
                SUM(exit_frame_idx - entry_frame_idx + 1)
            FROM visits;
            """
        )
        row = cursor.fetchone()
        total_visits = int(row[0]) if row and row[0] is not None else 0
        avg_dwell = float(row[1]) if row and row[1] is not None else 0.0
        min_dwell = float(row[2]) if row and row[2] is not None else 0.0
        max_dwell = float(row[3]) if row and row[3] is not None else 0.0
        raw_frame_alerts = int(row[4]) if row and row[4] is not None else (total_visits * 57)

        reduction_ratio = (
            f"{(raw_frame_alerts / total_visits):.1f}:1"
            if total_visits > 0
            else "1.0:1"
        )

        return {
            "total_visits": total_visits,
            "mean_duration_s": round(avg_dwell, 3),
            "min_duration_s": round(min_dwell, 3),
            "max_duration_s": round(max_dwell, 3),
            "total_raw_alerts_collapsed": raw_frame_alerts,
            "alert_reduction_ratio": reduction_ratio,
        }

    def export_json(self, output_path: Union[str, Path]) -> None:
        """Export all stored visits to a structured JSON file with summary telemetry."""
        self.export_to_json(output_path)

    def export_csv(self, output_path: Union[str, Path]) -> None:
        """Export all stored visits to a standard CSV file."""
        self.export_to_csv(output_path)

    def export_to_json(self, output_path: Union[str, Path]) -> None:
        """Export all stored visits to a structured JSON file.

        Args:
            output_path: Target JSON destination path.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        visits = self.list_visits()
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(visits, f, indent=2)
        logger.info("Exported %d SQLite visits to JSON: %s", len(visits), out_p)

    def export_to_csv(self, output_path: Union[str, Path]) -> None:
        """Export all stored visits to a flat CSV file.

        Args:
            output_path: Target CSV destination path.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        visits = self.list_visits()
        if not visits:
            logger.warning("No visits to export to CSV: %s", out_p)
            out_p.touch()
            return

        fieldnames = list(visits[0].keys())
        with open(out_p, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for v in visits:
                row = dict(v)
                if isinstance(row.get("merged_track_ids"), list):
                    row["merged_track_ids"] = json.dumps(row["merged_track_ids"])
                writer.writerow(row)

        logger.info("Exported %d SQLite visits to CSV: %s", len(visits), out_p)
