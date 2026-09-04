"""``tinderbot web``: JSON API, photo serving and the management actions, against a real HTTP server."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from tinderbot.storage import ProfileRecord, Storage
from tinderbot.webapp import make_server


@pytest.fixture
def seeded(cfg, storage: Storage):
    """Three profiles: a liked one with a stored photo, an uncertain nope, and one without a decision."""
    storage.upsert_profile(ProfileRecord(id="ana", name="Ana", age=27, bio="hiking and coffee", photo_urls=["u1"],
                                         verified=True, interests=["hiking"]))
    d = cfg.photos_path / "ana"
    d.mkdir(parents=True)
    (d / "00.jpg").write_bytes(b"\xff\xd8\xff\xe0jpegdata")
    storage.upsert_photo("ana_p0", "ana", "u1", str(d / "00.jpg"), 640, 480, face_count=1, quality=0.8)
    storage.upsert_photo("ana_p1", "ana", "u2", None, position=1)  # never downloaded
    storage.add_decision("ana", "like", 0.91, "auto", ["face_sim"], {"f": 1.0})

    storage.upsert_profile(ProfileRecord(id="bea", name="Bea", age=31, bio="travel"))
    storage.add_decision("bea", "nope", 0.51, "auto", ["uncertain"], {"f": 0.0})

    storage.upsert_profile(ProfileRecord(id="cid", name="Cid", age=24))
    return storage


@pytest.fixture
def server(cfg, seeded):
    srv = make_server(cfg, seeded, "127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def call(url: str, method: str = "GET", body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if r.headers.get_content_type() == "application/json" else raw)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_index_and_summary(server):
    status, html = call(server + "/")
    assert status == 200 and b"liked database" in html
    status, s = call(server + "/api/summary")
    assert status == 200
    assert (s["profiles"], s["liked"], s["noped"], s["uncertain"], s["unlabelled"]) == (3, 1, 1, 1, 1)


def test_list_filters_search_and_sort(server):
    _, j = call(server + "/api/profiles?filter=liked")
    assert [p["id"] for p in j["profiles"]] == ["ana"] and j["total"] == 1
    ana = j["profiles"][0]
    assert ana["label"] == 1 and ana["cover_photo_id"] == "ana_p0" and ana["reasons"] == ["face_sim"]
    _, j = call(server + "/api/profiles?filter=uncertain")
    assert [p["id"] for p in j["profiles"]] == ["bea"]
    _, j = call(server + "/api/profiles?filter=unlabelled")
    assert [p["id"] for p in j["profiles"]] == ["cid"] and j["profiles"][0]["cover_photo_id"] is None
    _, j = call(server + "/api/profiles?q=coffee")
    assert [p["id"] for p in j["profiles"]] == ["ana"]
    _, j = call(server + "/api/profiles?sort=name&limit=2")
    assert [p["id"] for p in j["profiles"]] == ["ana", "bea"] and j["total"] == 3
    _, j = call(server + "/api/profiles?sort=name&limit=2&offset=2")
    assert [p["id"] for p in j["profiles"]] == ["cid"]
    _, j = call(server + "/api/profiles?filter=bogus&sort=bogus&limit=abc")
    assert j["filter"] == "all" and j["sort"] == "recent" and j["total"] == 3


def test_detail_and_photo(server):
    status, d = call(server + "/api/profiles/ana")
    assert status == 200 and d["name"] == "Ana" and d["interests"] == ["hiking"]
    assert [(p["id"], p["available"]) for p in d["photos"]] == [("ana_p0", True), ("ana_p1", False)]
    assert d["decisions"][0]["features"] == {"f": 1.0}
    status, raw = call(server + "/photo/ana_p0")
    assert status == 200 and raw.startswith(b"\xff\xd8")
    assert call(server + "/photo/ana_p1")[0] == 404
    assert call(server + "/photo/nope")[0] == 404
    assert call(server + "/api/profiles/missing")[0] == 404


def test_photo_outside_data_dir_is_refused(server, seeded: Storage, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("not a photo")
    seeded.upsert_photo("bea_p0", "bea", "u", str(secret))
    assert call(server + "/photo/bea_p0")[0] == 404


def test_label_and_delete(server, seeded: Storage, cfg):
    # correct the uncertain nope -> like: existing decisions are relabelled and marked manual
    status, d = call(server + "/api/profiles/bea/label", "POST", {"label": 1})
    assert status == 200 and d["decisions"][0]["label"] == 1 and d["decisions"][0]["source"] == "manual"
    assert call(server + "/api/profiles?filter=uncertain")[1]["total"] == 0
    assert call(server + "/api/profiles?filter=liked")[1]["total"] == 2
    # a profile without any decision gets a manual one
    status, d = call(server + "/api/profiles/cid/label", "POST", {"label": 0})
    assert status == 200 and d["decisions"][0]["action"] == "nope" and d["decisions"][0]["source"] == "manual"
    assert call(server + "/api/summary")[1]["unlabelled"] == 0
    assert call(server + "/api/profiles/cid/label", "POST", {"label": 5})[0] == 400
    assert call(server + "/api/profiles/missing/label", "POST", {"label": 1})[0] == 404
    # delete removes rows (cascade) and the photo folder
    assert (cfg.photos_path / "ana").exists()
    status, j = call(server + "/api/profiles/ana", "DELETE")
    assert status == 200 and j == {"deleted": True}
    assert seeded.get_profile("ana") is None and seeded.get_photo("ana_p0") is None
    assert not seeded.has_decision("ana")
    assert not (cfg.photos_path / "ana").exists()
    assert call(server + "/api/profiles/ana", "DELETE")[0] == 404
    assert call(server + "/api/summary")[1]["profiles"] == 2


def test_cli_registers_web_command():
    from typer.testing import CliRunner

    from tinderbot.cli import app

    r = CliRunner().invoke(app, ["web", "--help"])
    assert r.exit_code == 0 and "--no-open" in r.output and "--port" in r.output
