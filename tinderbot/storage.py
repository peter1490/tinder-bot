"""Local SQLite storage: profiles, photos, embeddings, decisions, reference sets and events.

Vectors are stored as float32 BLOBs. The database uses WAL mode so the bot can write while the
``stats``/``review`` commands read. Nothing here ever leaves the machine.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    id            TEXT PRIMARY KEY,
    name          TEXT,
    age           INTEGER,
    bio           TEXT,
    distance_km   REAL,
    verified      INTEGER DEFAULT 0,
    jobs          TEXT,
    schools       TEXT,
    interests     TEXT,
    photo_count   INTEGER DEFAULT 0,
    raw_json      TEXT,
    first_seen    REAL,
    last_seen     REAL
);
CREATE TABLE IF NOT EXISTS photos (
    id            TEXT PRIMARY KEY,
    profile_id    TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    url           TEXT,
    local_path    TEXT,
    width         INTEGER,
    height        INTEGER,
    face_count    INTEGER,
    quality       REAL,
    position      INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS photos_profile ON photos(profile_id);
CREATE TABLE IF NOT EXISTS embeddings (
    photo_id      TEXT NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL,          -- 'face' | 'clip'
    idx           INTEGER DEFAULT 0,      -- face index within the photo
    model         TEXT,
    dim           INTEGER,
    vector        BLOB NOT NULL,
    meta          TEXT,
    PRIMARY KEY (photo_id, kind, idx)
);
CREATE TABLE IF NOT EXISTS decisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id    TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    action        TEXT NOT NULL,          -- 'like' | 'nope' | 'skip'
    label         INTEGER,                -- 1 like / 0 nope, as training target (manual review may override)
    score         REAL,
    source        TEXT,                   -- 'auto' | 'manual' | 'import'
    reasons       TEXT,
    features      TEXT,
    ts            REAL
);
CREATE INDEX IF NOT EXISTS decisions_profile ON decisions(profile_id);
CREATE TABLE IF NOT EXISTS reference_vectors (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    path          TEXT,
    label         TEXT NOT NULL,          -- 'liked' | 'disliked'
    kind          TEXT NOT NULL,          -- 'face' | 'clip'
    model         TEXT,
    dim           INTEGER,
    vector        BLOB NOT NULL,
    UNIQUE(path, label, kind, model)
);
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL,
    kind          TEXT,
    detail        TEXT
);
CREATE TABLE IF NOT EXISTS meta (
    key           TEXT PRIMARY KEY,
    value         TEXT
);
"""


def to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(blob: bytes, dim: int | None = None) -> np.ndarray:
    v = np.frombuffer(blob, dtype=np.float32)
    return v if dim is None else v[:dim]


def recurring_vector(vectors: list[np.ndarray]) -> np.ndarray:
    """The vector most similar (cosine) to all the others: the person who recurs across a profile's photos."""
    if len(vectors) == 1:
        return vectors[0]
    m = np.stack(vectors).astype(np.float32)
    n = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
    sims = n @ n.T
    np.fill_diagonal(sims, 0.0)
    return m[int(np.argmax(sims.sum(axis=1)))]


@dataclass
class ProfileRecord:
    id: str
    name: str = ""
    age: int | None = None
    bio: str = ""
    distance_km: float | None = None
    verified: bool = False
    jobs: list[str] = field(default_factory=list)
    schools: list[str] = field(default_factory=list)
    interests: list[str] = field(default_factory=list)
    photo_urls: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    s_number: int | None = None

    @property
    def photo_count(self) -> int:
        return len(self.photo_urls)


class Storage:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)

    # ---- profiles ---------------------------------------------------------
    def upsert_profile(self, p: ProfileRecord) -> None:
        now = time.time()
        self.conn.execute(
            """INSERT INTO profiles(id,name,age,bio,distance_km,verified,jobs,schools,interests,photo_count,
                                    raw_json,first_seen,last_seen)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET name=excluded.name, age=excluded.age, bio=excluded.bio,
                    distance_km=excluded.distance_km, verified=excluded.verified, jobs=excluded.jobs,
                    schools=excluded.schools, interests=excluded.interests, photo_count=excluded.photo_count,
                    raw_json=excluded.raw_json, last_seen=excluded.last_seen""",
            (p.id, p.name, p.age, p.bio, p.distance_km, int(p.verified), json.dumps(p.jobs),
             json.dumps(p.schools), json.dumps(p.interests), p.photo_count,
             json.dumps(p.raw, ensure_ascii=False) if p.raw else None, now, now),
        )
        self.conn.commit()

    def get_profile(self, profile_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM profiles WHERE id=?", (profile_id,)).fetchone()

    def profile_record(self, profile_id: str) -> ProfileRecord | None:
        """Rebuild a :class:`ProfileRecord` (with the stored photo urls) from the database."""
        r = self.get_profile(profile_id)
        if r is None:
            return None

        def lst(s: str | None) -> list[str]:
            with contextlib.suppress(ValueError, TypeError):
                v = json.loads(s or "[]")
                return [str(x) for x in v] if isinstance(v, list) else []
            return []

        urls = [row["url"] or "" for row in self.photos_for_profile(profile_id)]
        return ProfileRecord(id=r["id"], name=r["name"] or "", age=r["age"], bio=r["bio"] or "",
                             distance_km=r["distance_km"], verified=bool(r["verified"]), jobs=lst(r["jobs"]),
                             schools=lst(r["schools"]), interests=lst(r["interests"]), photo_urls=urls)

    def has_decision(self, profile_id: str) -> bool:
        return self.conn.execute("SELECT 1 FROM decisions WHERE profile_id=? LIMIT 1", (profile_id,)).fetchone() is not None

    # ---- photos / embeddings ----------------------------------------------
    def upsert_photo(self, photo_id: str, profile_id: str, url: str, local_path: str | None,
                     width: int | None = None, height: int | None = None, face_count: int | None = None,
                     quality: float | None = None, position: int = 0) -> None:
        self.conn.execute(
            """INSERT INTO photos(id,profile_id,url,local_path,width,height,face_count,quality,position)
               VALUES(?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET local_path=COALESCE(excluded.local_path, photos.local_path),
                    width=COALESCE(excluded.width, photos.width), height=COALESCE(excluded.height, photos.height),
                    face_count=COALESCE(excluded.face_count, photos.face_count),
                    quality=COALESCE(excluded.quality, photos.quality)""",
            (photo_id, profile_id, url, local_path, width, height, face_count, quality, position),
        )
        self.conn.commit()

    def put_embedding(self, photo_id: str, kind: str, vec: np.ndarray, model: str, idx: int = 0,
                      meta: dict | None = None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings(photo_id,kind,idx,model,dim,vector,meta) VALUES(?,?,?,?,?,?,?)",
            (photo_id, kind, idx, model, int(vec.shape[-1]), to_blob(vec), json.dumps(meta) if meta else None),
        )
        self.conn.commit()

    def embeddings_for_profile(self, profile_id: str, kind: str) -> list[np.ndarray]:
        rows = self.conn.execute(
            """SELECT e.vector, e.dim FROM embeddings e JOIN photos p ON p.id=e.photo_id
               WHERE p.profile_id=? AND e.kind=? ORDER BY p.position, e.idx""",
            (profile_id, kind),
        ).fetchall()
        return [from_blob(r["vector"], r["dim"]) for r in rows]

    def photo_embedding_rows(self, profile_id: str) -> list[sqlite3.Row]:
        """Every stored photo of a profile joined with its embeddings (one row per embedding, photos
        without embeddings still appear once with NULL embedding columns)."""
        return self.conn.execute(
            """SELECT p.id AS photo_id, p.url, p.local_path, p.width, p.height, p.quality, p.position,
                      e.kind, e.idx, e.vector, e.dim, e.meta
               FROM photos p LEFT JOIN embeddings e ON e.photo_id=p.id
               WHERE p.profile_id=? ORDER BY p.position, e.kind, e.idx""",
            (profile_id,),
        ).fetchall()

    # ---- decisions ----------------------------------------------------------
    def add_decision(self, profile_id: str, action: str, score: float | None, source: str,
                     reasons: Iterable[str] = (), features: dict | None = None, label: int | None = None) -> int:
        if label is None:
            label = {"like": 1, "superlike": 1, "nope": 0}.get(action)
        cur = self.conn.execute(
            "INSERT INTO decisions(profile_id,action,label,score,source,reasons,features,ts) VALUES(?,?,?,?,?,?,?,?)",
            (profile_id, action, label, score, source, json.dumps(list(reasons)),
             json.dumps(features) if features else None, time.time()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def relabel(self, profile_id: str, label: int) -> None:
        """Manual review: override the training label for every decision on this profile."""
        self.conn.execute("UPDATE decisions SET label=?, source='manual' WHERE profile_id=?", (label, profile_id))
        self.conn.commit()

    def training_examples(self) -> list[sqlite3.Row]:
        """Latest decision per profile that carries a label and stored features."""
        return self.conn.execute(
            """SELECT d.profile_id, d.label, d.features, d.source FROM decisions d
               JOIN (SELECT profile_id, MAX(id) AS mid FROM decisions GROUP BY profile_id) m ON m.mid=d.id
               WHERE d.label IS NOT NULL AND d.features IS NOT NULL"""
        ).fetchall()

    def count_decisions(self, since_ts: float | None = None, action: str | None = None,
                        source: str | Iterable[str] | None = None) -> int:
        q, args = "SELECT COUNT(*) FROM decisions WHERE 1=1", []
        if since_ts is not None:
            q += " AND ts>=?"
            args.append(since_ts)
        if action:
            q += " AND action=?"
            args.append(action)
        if source:
            sources = [source] if isinstance(source, str) else list(source)
            q += f" AND source IN ({','.join('?' * len(sources))})"
            args.extend(sources)
        return int(self.conn.execute(q, args).fetchone()[0])

    def recent_actions(self, limit: int, source: str | None = "auto", since_ts: float | None = None) -> list[str]:
        """Actions of the most recent decisions, newest first (for the like-ratio governor)."""
        q, args = "SELECT action FROM decisions WHERE 1=1", []
        if source:
            q += " AND source=?"
            args.append(source)
        if since_ts is not None:
            q += " AND ts>=?"
            args.append(since_ts)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(int(limit))
        return [r[0] for r in self.conn.execute(q, args)]

    def first_decision_ts(self, source: str | None = "auto") -> float | None:
        """Timestamp of the first stored decision (used to ramp up a freshly automated account)."""
        q, args = "SELECT MIN(ts) FROM decisions", []
        if source:
            q += " WHERE source=?"
            args.append(source)
        v = self.conn.execute(q, args).fetchone()[0]
        return float(v) if v is not None else None

    def uncertain_for_review(self, limit: int = 50) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT d.id, d.profile_id, d.action, d.score, d.reasons, p.name, p.age, p.bio
               FROM decisions d JOIN profiles p ON p.id=d.profile_id
               WHERE d.source='auto' AND d.reasons LIKE '%uncertain%' ORDER BY d.ts DESC LIMIT ?""",
            (limit,),
        ).fetchall()

    # ---- reference vectors ----------------------------------------------------
    def put_reference(self, path: str, label: str, kind: str, model: str, vec: np.ndarray) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO reference_vectors(path,label,kind,model,dim,vector) VALUES(?,?,?,?,?,?)""",
            (path, label, kind, model, int(vec.shape[-1]), to_blob(vec)),
        )
        self.conn.commit()

    def references(self, label: str, kind: str) -> np.ndarray:
        rows = self.conn.execute(
            "SELECT vector, dim FROM reference_vectors WHERE label=? AND kind=?", (label, kind)
        ).fetchall()
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([from_blob(r["vector"], r["dim"]) for r in rows])

    def clear_references(self, label: str | None = None) -> None:
        if label:
            self.conn.execute("DELETE FROM reference_vectors WHERE label=?", (label,))
        else:
            self.conn.execute("DELETE FROM reference_vectors")
        self.conn.commit()

    def labelled_profile_vectors(self, kind: str, label: int) -> tuple[list[str], np.ndarray]:
        """One reference vector per profile whose latest training label is ``label`` (1 like / 0 nope).

        * ``face``: the profile's *primary* face (the largest face of each photo, then the one that
          recurs across photos) so friends in group shots never enter the reference pool;
        * ``clip``: the mean whole-photo embedding.

        Returns ``(profile_ids, matrix)`` aligned by row, so a caller can leave a profile's own vector
        out when scoring that same profile.
        """
        rows = self.conn.execute(
            """SELECT p.profile_id, e.idx, e.vector, e.dim FROM decisions d
               JOIN (SELECT profile_id, MAX(id) AS mid FROM decisions GROUP BY profile_id) m ON m.mid=d.id
               JOIN photos p ON p.profile_id=d.profile_id JOIN embeddings e ON e.photo_id=p.id
               WHERE d.label=? AND e.kind=? ORDER BY p.profile_id, p.position, e.idx""",
            (label, kind),
        ).fetchall()
        per: dict[str, list[np.ndarray]] = {}
        for r in rows:
            if kind == "face" and r["idx"] != 0:
                continue  # only the largest face of every photo
            per.setdefault(r["profile_id"], []).append(from_blob(r["vector"], r["dim"]))
        if not per:
            return [], np.zeros((0, 0), dtype=np.float32)
        ids = list(per)
        if kind == "face":
            m = np.stack([recurring_vector(per[i]) for i in ids])
        else:
            m = np.stack([np.mean(per[i], axis=0) for i in ids])
        return ids, m.astype(np.float32)

    # ---- browsing / management (used by ``tinderbot web``) ---------------------
    _LATEST = """SELECT d.* FROM decisions d
                 JOIN (SELECT profile_id, MAX(id) AS mid FROM decisions GROUP BY profile_id) m ON m.mid=d.id"""

    def browse_profiles(self, filter: str = "all", q: str = "", limit: int = 60, offset: int = 0,
                        sort: str = "recent") -> tuple[list[dict[str, Any]], int]:
        """Profiles with their latest decision; ``filter`` is all|liked|noped|uncertain|unlabelled|manual.

        Returns ``(rows, total)`` where *total* is the count before ``limit``/``offset``.
        """
        where, args = [], []
        if filter == "liked":
            where.append("d.label=1")
        elif filter == "noped":
            where.append("d.label=0")
        elif filter == "uncertain":
            where.append("d.source='auto' AND d.reasons LIKE '%uncertain%'")
        elif filter == "unlabelled":
            where.append("d.id IS NULL")
        elif filter == "manual":
            where.append("d.source='manual'")
        if q:
            where.append("(p.name LIKE ? OR p.bio LIKE ? OR p.id LIKE ?)")
            args += [f"%{q}%"] * 3
        cond = (" WHERE " + " AND ".join(where)) if where else ""
        order = {
            "recent": "COALESCE(d.ts, p.last_seen) DESC",
            "oldest": "COALESCE(d.ts, p.last_seen) ASC",
            "score": "d.score DESC NULLS LAST",
            "name": "p.name COLLATE NOCASE ASC",
        }.get(sort, "COALESCE(d.ts, p.last_seen) DESC")
        base = f"FROM profiles p LEFT JOIN ({self._LATEST}) d ON d.profile_id=p.id{cond}"
        total = int(self.conn.execute(f"SELECT COUNT(*) {base}", args).fetchone()[0])
        rows = self.conn.execute(
            f"""SELECT p.id, p.name, p.age, p.bio, p.distance_km, p.verified, p.photo_count, p.last_seen,
                       d.id AS decision_id, d.action, d.label, d.score, d.source, d.reasons, d.ts,
                       (SELECT id FROM photos ph WHERE ph.profile_id=p.id AND ph.local_path IS NOT NULL
                        ORDER BY ph.position LIMIT 1) AS cover_photo_id,
                       (SELECT COUNT(*) FROM photos ph WHERE ph.profile_id=p.id AND ph.local_path IS NOT NULL)
                        AS stored_photos
                {base} ORDER BY {order} LIMIT ? OFFSET ?""",
            [*args, limit, offset],
        ).fetchall()
        return [self._profile_dict(r) for r in rows], total

    @staticmethod
    def _profile_dict(r: sqlite3.Row) -> dict[str, Any]:
        d = dict(r)
        for k in ("jobs", "schools", "interests", "reasons", "features"):
            if k in d and isinstance(d[k], str):
                with contextlib.suppress(ValueError):
                    d[k] = json.loads(d[k])
        d.pop("raw_json", None)
        return d

    def profile_detail(self, profile_id: str) -> dict[str, Any] | None:
        p = self.get_profile(profile_id)
        if p is None:
            return None
        d = self._profile_dict(p)
        d["photos"] = [dict(r) for r in self.photos_for_profile(profile_id)]
        d["decisions"] = [self._profile_dict(r) for r in self.decisions_for_profile(profile_id)]
        return d

    def photos_for_profile(self, profile_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, url, local_path, width, height, face_count, quality, position FROM photos "
            "WHERE profile_id=? ORDER BY position", (profile_id,)
        ).fetchall()

    def get_photo(self, photo_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM photos WHERE id=?", (photo_id,)).fetchone()

    def decisions_for_profile(self, profile_id: str) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM decisions WHERE profile_id=? ORDER BY id DESC", (profile_id,)).fetchall()

    def set_label(self, profile_id: str, label: int) -> None:
        """Set the training label for a profile: relabel existing decisions or add a manual one."""
        if self.has_decision(profile_id):
            self.relabel(profile_id, label)
        else:
            self.add_decision(profile_id, "like" if label else "nope", None, "manual", ["web"], label=label)

    def delete_profile(self, profile_id: str) -> bool:
        """Remove a profile with its photos, embeddings and decisions (cascade)."""
        cur = self.conn.execute("DELETE FROM profiles WHERE id=?", (profile_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def summary(self) -> dict[str, Any]:
        day = time.time() - (time.time() % 86400)
        one = lambda sql, *a: int(self.conn.execute(sql, a).fetchone()[0])  # noqa: E731
        latest = f"SELECT label, source, reasons, action FROM ({self._LATEST})"
        return {
            "profiles": one("SELECT COUNT(*) FROM profiles"),
            "photos": one("SELECT COUNT(*) FROM photos"),
            "liked": one(f"SELECT COUNT(*) FROM ({latest}) WHERE label=1"),
            "superliked": one(f"SELECT COUNT(*) FROM ({latest}) WHERE action='superlike'"),
            "noped": one(f"SELECT COUNT(*) FROM ({latest}) WHERE label=0"),
            "uncertain": one(f"SELECT COUNT(*) FROM ({latest}) WHERE source='auto' AND reasons LIKE '%uncertain%'"),
            "manual": one(f"SELECT COUNT(*) FROM ({latest}) WHERE source='manual'"),
            "unlabelled": one("SELECT COUNT(*) FROM profiles p WHERE NOT EXISTS "
                              "(SELECT 1 FROM decisions d WHERE d.profile_id=p.id)"),
            "decisions_today": self.count_decisions(since_ts=day),
            "references": {r[0]: int(r[1]) for r in self.conn.execute(
                "SELECT label||'/'||kind, COUNT(*) FROM reference_vectors GROUP BY 1")},
        }

    # ---- events / meta --------------------------------------------------------
    def log_event(self, kind: str, detail: str | dict = "") -> None:
        if isinstance(detail, dict):
            detail = json.dumps(detail)
        self.conn.execute("INSERT INTO events(ts,kind,detail) VALUES(?,?,?)", (time.time(), kind, detail))
        self.conn.commit()

    def count_events(self, kind: str, since_ts: float) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM events WHERE kind=? AND ts>=?", (kind, since_ts)).fetchone()[0])

    def last_event(self, kind: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM events WHERE kind=? ORDER BY id DESC LIMIT 1", (kind,)).fetchone()

    def recent_events(self, limit: int = 20, since_ts: float | None = None) -> list[sqlite3.Row]:
        q, args = "SELECT * FROM events", []
        if since_ts is not None:
            q += " WHERE ts>=?"
            args.append(since_ts)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(int(limit))
        return self.conn.execute(q, args).fetchall()

    def delete_meta(self, key: str) -> None:
        self.conn.execute("DELETE FROM meta WHERE key=?", (key,))
        self.conn.commit()

    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, json.dumps(value)))
        self.conn.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def iter_profiles(self) -> Iterator[sqlite3.Row]:
        yield from self.conn.execute("SELECT * FROM profiles ORDER BY last_seen DESC")

    def close(self) -> None:
        self.conn.close()
