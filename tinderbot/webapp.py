"""``tinderbot web``: a small local web app to browse and manage the liked/noped database.

Standard library only (``http.server``), bound to localhost by default. It reads the same SQLite file
the bot writes (WAL mode, so both can run at once) and serves the photos stored under ``data/photos``.

Routes
    GET  /                           single-page UI
    GET  /api/summary                counts (profiles, liked, noped, uncertain, ...)
    GET  /api/profiles               ?filter=all|liked|noped|uncertain|unlabelled|manual &q= &sort= &limit= &offset=
    GET  /api/profiles/<id>          profile + photos + decision history
    POST /api/profiles/<id>/label    {"label": 1|0}  -> relabel (or add a manual decision)
    DELETE /api/profiles/<id>        remove the profile, its photos, embeddings and decisions
    GET  /photo/<photo_id>           the locally stored photo file
"""

from __future__ import annotations

import json
import mimetypes
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from .config import Config
from .storage import Storage

FILTERS = ("all", "liked", "noped", "uncertain", "unlabelled", "manual")
SORTS = ("recent", "oldest", "score", "name")


def _int(value: str | None, default: int) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except ValueError:
        return default


class WebApp:
    """Request handlers, independent of the HTTP plumbing (easy to unit-test)."""

    def __init__(self, cfg: Config, storage: Storage):
        self.cfg = cfg
        self.storage = storage
        self.lock = threading.Lock()  # one sqlite connection shared by the server threads

    # ---- API ------------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        with self.lock:
            return self.storage.summary()

    def list_profiles(self, params: dict[str, str]) -> dict[str, Any]:
        filt = params.get("filter", "all")
        if filt not in FILTERS:
            filt = "all"
        sort = params.get("sort", "recent")
        if sort not in SORTS:
            sort = "recent"
        limit = max(1, min(_int(params.get("limit"), 60), 500))
        offset = max(0, _int(params.get("offset"), 0))
        with self.lock:
            rows, total = self.storage.browse_profiles(filt, params.get("q", "").strip(), limit, offset, sort)
        return {"profiles": rows, "total": total, "offset": offset, "limit": limit, "filter": filt, "sort": sort}

    def profile(self, profile_id: str) -> dict[str, Any] | None:
        with self.lock:
            d = self.storage.profile_detail(profile_id)
        if d is None:
            return None
        for ph in d["photos"]:
            ph["available"] = self._photo_file(ph["local_path"]) is not None
        return d

    def set_label(self, profile_id: str, label: int) -> dict[str, Any] | None:
        with self.lock:
            if self.storage.get_profile(profile_id) is None:
                return None
            self.storage.set_label(profile_id, label)
            self.storage.log_event("web_label", {"profile_id": profile_id, "label": label})
        return self.profile(profile_id)

    def delete_profile(self, profile_id: str) -> bool:
        with self.lock:
            ok = self.storage.delete_profile(profile_id)
            if ok:
                self.storage.log_event("web_delete", {"profile_id": profile_id})
        if ok:
            folder = self.cfg.photos_path / profile_id
            if folder.is_dir() and self._inside_photos(folder):
                shutil.rmtree(folder, ignore_errors=True)
        return ok

    # ---- photos --------------------------------------------------------------------
    def _inside_photos(self, p: Path) -> bool:
        try:
            p.resolve().relative_to(self.cfg.photos_path.resolve())
            return True
        except ValueError:
            return False

    def _photo_file(self, local_path: str | None) -> Path | None:
        if not local_path:
            return None
        p = Path(local_path)
        if not p.is_absolute():
            p = self.cfg.photos_path / p
        if p.is_file() and self._inside_photos(p):
            return p
        return None

    def photo_path(self, photo_id: str) -> Path | None:
        with self.lock:
            row = self.storage.get_photo(photo_id)
        return self._photo_file(row["local_path"]) if row else None


class _Handler(BaseHTTPRequestHandler):
    app: WebApp  # set by ``make_server``
    server_version = "tinderbot-web/1.0"
    quiet = True

    # ---- helpers -------------------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401 - silence default logging
        if not self.quiet:
            super().log_message(fmt, *args)

    def _send(self, status: int, body: bytes, ctype: str, cache: str = "no-store") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj: Any, status: int = 200) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _error(self, status: int, msg: str) -> None:
        self._json({"error": msg}, status)

    def _body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    def _route(self) -> tuple[list[str], dict[str, str]]:
        u = urlsplit(self.path)
        parts = [unquote(p) for p in u.path.split("/") if p]
        params = {k: v[0] for k, v in parse_qs(u.query).items()}
        return parts, params

    # ---- verbs ------------------------------------------------------------------------
    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        parts, params = self._route()
        try:
            if not parts:
                self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8")
            elif parts == ["favicon.ico"]:
                self._send(200, FAVICON_SVG.encode(), "image/svg+xml", cache="private, max-age=86400")
            elif parts == ["api", "summary"]:
                self._json(self.app.summary())
            elif parts == ["api", "profiles"]:
                self._json(self.app.list_profiles(params))
            elif len(parts) == 3 and parts[:2] == ["api", "profiles"]:
                d = self.app.profile(parts[2])
                self._json(d) if d else self._error(404, "no such profile")
            elif len(parts) == 2 and parts[0] == "photo":
                p = self.app.photo_path(parts[1])
                if p is None:
                    self._error(404, "no such photo")
                else:
                    ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
                    self._send(200, p.read_bytes(), ctype, cache="private, max-age=86400")
            else:
                self._error(404, "not found")
        except ValueError as e:
            self._error(400, str(e))

    def do_POST(self) -> None:  # noqa: N802
        parts, _ = self._route()
        if len(parts) == 4 and parts[:2] == ["api", "profiles"] and parts[3] == "label":
            label = self._body().get("label")
            if label not in (0, 1, True, False):
                return self._error(400, "label must be 0 or 1")
            d = self.app.set_label(parts[2], int(label))
            return self._json(d) if d else self._error(404, "no such profile")
        self._error(404, "not found")

    def do_DELETE(self) -> None:  # noqa: N802
        parts, _ = self._route()
        if len(parts) == 3 and parts[:2] == ["api", "profiles"]:
            ok = self.app.delete_profile(parts[2])
            return self._json({"deleted": ok}) if ok else self._error(404, "no such profile")
        self._error(404, "not found")


def make_server(cfg: Config, storage: Storage, host: str = "127.0.0.1", port: int = 8765,
                quiet: bool = True) -> ThreadingHTTPServer:
    """Build (but do not start) the HTTP server. ``port=0`` picks a free port."""
    webapp = WebApp(cfg, storage)
    handler = type("Handler", (_Handler,), {"app": webapp, "quiet": quiet})
    srv = ThreadingHTTPServer((host, port), handler)
    srv.daemon_threads = True
    return srv


def serve(cfg: Config, storage: Storage, host: str = "127.0.0.1", port: int = 8765,
          open_browser: bool = True, log=print) -> None:
    """Run the web app until Ctrl-C."""
    import webbrowser

    srv = make_server(cfg, storage, host, port)
    url = f"http://{host}:{srv.server_address[1]}/"
    log(f"tinderbot web app on {url}  (database: {storage.path})  Ctrl-C to stop")
    if open_browser:
        threading.Timer(0.4, webbrowser.open, (url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


FAVICON_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
               '<circle cx="8" cy="8" r="7" fill="#e0245e"/></svg>')

# ------------------------------------------------------------------------------------------
# Single-page UI (no build step, no external assets; everything stays on the machine).
# ------------------------------------------------------------------------------------------
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="/favicon.ico">
<title>tinderbot · liked database</title>
<style>
:root{
  --bg:#f6f7f9;--panel:#ffffff;--ink:#16181d;--muted:#6b7280;--line:#e5e7eb;--accent:#e0245e;
  --like:#16a34a;--like-bg:#dcfce7;--nope:#dc2626;--nope-bg:#fee2e2;--warn:#d97706;--warn-bg:#fef3c7;
  --neutral:#6b7280;--neutral-bg:#f3f4f6;--shadow:0 1px 2px rgba(0,0,0,.06),0 8px 24px rgba(0,0,0,.06);
}
@media (prefers-color-scheme: dark){
  :root{--bg:#0f1115;--panel:#181b22;--ink:#e6e8ee;--muted:#9aa1ad;--line:#272b35;
    --like-bg:#123521;--nope-bg:#3b1414;--warn-bg:#3a2a08;--neutral-bg:#23262e;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.35)}
}
*{box-sizing:border-box}
[hidden]{display:none!important}
html,body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
a{color:inherit}
button{font:inherit;color:inherit;cursor:pointer;border:1px solid var(--line);background:var(--panel);border-radius:8px;padding:6px 10px}
button:hover{border-color:var(--muted)}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button.like{background:var(--like);border-color:var(--like);color:#fff}
button.nope{background:var(--nope);border-color:var(--nope);color:#fff}
button.danger{color:var(--nope)}
button:disabled{opacity:.5;cursor:default}
input,select{font:inherit;color:inherit;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:6px 10px}
header{position:sticky;top:0;z-index:5;background:var(--panel);border-bottom:1px solid var(--line);box-shadow:var(--shadow)}
.bar{max-width:1400px;margin:0 auto;padding:10px 16px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
.brand{font-weight:700;font-size:16px;display:flex;align-items:center;gap:8px;margin-right:auto}
.brand .dot{width:10px;height:10px;border-radius:50%;background:var(--accent)}
.stats{display:flex;gap:6px;flex-wrap:wrap}
.chip{border:1px solid var(--line);border-radius:999px;padding:3px 10px;font-size:12px;color:var(--muted);background:var(--bg)}
.chip b{color:var(--ink)}
.tabs{display:flex;gap:4px;flex-wrap:wrap}
.tabs button{border-radius:999px;padding:5px 12px}
.tabs button.on{background:var(--ink);color:var(--bg);border-color:var(--ink)}
main{max-width:1400px;margin:0 auto;padding:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:var(--shadow);cursor:pointer;position:relative;transition:transform .12s}
.card:hover{transform:translateY(-2px)}
.card.sel{outline:2px solid var(--accent)}
.card .ph{aspect-ratio:3/4;background:var(--neutral-bg);display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:42px;overflow:hidden}
.card .ph img{width:100%;height:100%;object-fit:cover;display:block}
.card .meta{padding:8px 10px 10px}
.card .name{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card .sub{color:var(--muted);font-size:12px;display:flex;justify-content:space-between;gap:6px;margin-top:2px}
.badge{position:absolute;top:8px;left:8px;font-size:11px;font-weight:700;letter-spacing:.03em;border-radius:999px;padding:3px 8px;background:var(--neutral-bg);color:var(--neutral)}
.badge.like{background:var(--like-bg);color:var(--like)}
.badge.nope{background:var(--nope-bg);color:var(--nope)}
.badge.uncertain{background:var(--warn-bg);color:var(--warn)}
.badge.manual::after{content:" ✎"}
.ver{position:absolute;top:8px;right:8px;background:#2563eb;color:#fff;border-radius:50%;width:20px;height:20px;font-size:12px;display:flex;align-items:center;justify-content:center}
.card .quick{position:absolute;bottom:58px;right:8px;display:none;gap:4px}
.card:hover .quick{display:flex}
.quick button{padding:4px 7px;font-size:13px;border-radius:999px}
.empty{color:var(--muted);text-align:center;padding:60px 0}
.more{display:flex;justify-content:center;padding:20px}
/* drawer */
.drawer{position:fixed;inset:0;z-index:10;display:none}
.drawer.open{display:block}
.drawer .back{position:absolute;inset:0;background:rgba(0,0,0,.45)}
.drawer .panel{position:absolute;top:0;right:0;bottom:0;width:min(720px,100%);background:var(--panel);overflow:auto;box-shadow:var(--shadow);display:flex;flex-direction:column}
.panel .head{display:flex;align-items:center;gap:8px;padding:12px 16px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--panel);z-index:2}
.panel .head .badge{position:static}
.panel .head h2{margin:0;font-size:18px;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.gallery{display:flex;gap:6px;overflow-x:auto;padding:12px 16px;scroll-snap-type:x mandatory}
.gallery img{height:360px;max-width:80%;object-fit:cover;border-radius:10px;scroll-snap-align:start;background:var(--neutral-bg)}
.gallery .nophoto{height:360px;width:270px;display:flex;align-items:center;justify-content:center;color:var(--muted);background:var(--neutral-bg);border-radius:10px;flex:none}
.section{padding:8px 16px}
.section h3{margin:10px 0 6px;font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.bio{white-space:pre-wrap}
.tags{display:flex;flex-wrap:wrap;gap:6px}
.tags span{background:var(--neutral-bg);border-radius:999px;padding:2px 10px;font-size:12px}
.actions{display:flex;gap:8px;padding:12px 16px;border-top:1px solid var(--line);position:sticky;bottom:0;background:var(--panel)}
.actions .sp{flex:1}
table{border-collapse:collapse;width:100%;font-size:12px}
td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:500}
td.num{text-align:right;font-variant-numeric:tabular-nums}
details summary{cursor:pointer;color:var(--muted);font-size:12px}
.kbd{font-size:11px;color:var(--muted)}
.kbd kbd{border:1px solid var(--line);border-radius:4px;padding:0 5px;background:var(--bg)}
.toast{position:fixed;left:50%;bottom:20px;transform:translateX(-50%);background:var(--ink);color:var(--bg);padding:8px 14px;border-radius:999px;z-index:20;opacity:0;transition:opacity .2s;pointer-events:none}
.toast.show{opacity:1}
</style>
</head>
<body>
<header>
  <div class="bar">
    <div class="brand"><span class="dot"></span>tinderbot · liked database</div>
    <div class="stats" id="stats"></div>
  </div>
  <div class="bar" style="padding-top:0">
    <div class="tabs" id="tabs"></div>
    <input id="q" type="search" placeholder="search name / bio / id…" style="min-width:220px">
    <select id="sort">
      <option value="recent">newest first</option>
      <option value="oldest">oldest first</option>
      <option value="score">highest score</option>
      <option value="name">name A→Z</option>
    </select>
    <span class="kbd"><kbd>←</kbd><kbd>→</kbd> browse · <kbd>L</kbd> like · <kbd>N</kbd> nope · <kbd>⌫</kbd> delete · <kbd>Esc</kbd> close</span>
  </div>
</header>
<main>
  <div class="grid" id="grid"></div>
  <div class="empty" id="empty" hidden>nothing here yet — run <code>tinderbot swipe</code> or <code>tinderbot swipe --shadow</code></div>
  <div class="more" id="more" hidden><button id="moreBtn">load more</button></div>
</main>

<div class="drawer" id="drawer">
  <div class="back" id="back"></div>
  <div class="panel">
    <div class="head">
      <button id="prev" title="previous (←)">‹</button>
      <button id="next" title="next (→)">›</button>
      <h2 id="dTitle"></h2>
      <span id="dBadge"></span>
      <button id="close" title="close (Esc)">✕</button>
    </div>
    <div class="gallery" id="gallery"></div>
    <div class="section" id="dInfo"></div>
    <div class="section"><h3>bio</h3><div class="bio" id="dBio"></div></div>
    <div class="section" id="dTags"></div>
    <div class="section"><h3>decisions</h3><div id="dDecisions"></div></div>
    <div class="section" id="dFeatures"></div>
    <div class="actions">
      <button class="like" id="aLike">👍 like (L)</button>
      <button class="nope" id="aNope">👎 nope (N)</button>
      <span class="sp"></span>
      <button class="danger" id="aDelete">🗑 delete</button>
    </div>
  </div>
</div>
<div class="toast" id="toast"></div>

<script>
const FILTERS=[["all","all"],["liked","liked"],["noped","noped"],["uncertain","uncertain"],["manual","reviewed"],["unlabelled","unlabelled"]];
const state={filter:"liked",q:"",sort:"recent",offset:0,limit:60,total:0,items:[],sel:-1,detail:null};
const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const api=async(url,opt)=>{const r=await fetch(url,opt);const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(j.error||r.statusText);return j;};
let toastT;function toast(m){const t=$("toast");t.textContent=m;t.classList.add("show");clearTimeout(toastT);toastT=setTimeout(()=>t.classList.remove("show"),1600);}
const fmtTs=t=>t?new Date(t*1000).toLocaleString():"";
function verdictOf(p){
  if(p.label===1)return "like";if(p.label===0)return "nope";
  if(p.action)return p.action;return "";
}
function badge(p){
  const v=verdictOf(p);if(!v&&!p.action)return '<span class="badge">new</span>';
  const unc=p.source==="auto"&&(p.reasons||[]).some(r=>String(r).includes("uncertain"));
  const cls=[ "badge", v, unc?"uncertain":"", p.source==="manual"?"manual":"" ].join(" ");
  const sc=p.score!=null?` ${Number(p.score).toFixed(2)}`:"";
  return `<span class="${cls}">${v.toUpperCase()}${sc}</span>`;
}

async function loadStats(){
  const s=await api("/api/summary");
  $("stats").innerHTML=[["profiles",s.profiles],["liked",s.liked],["noped",s.noped],["uncertain",s.uncertain],["reviewed",s.manual],["unlabelled",s.unlabelled],["today",s.decisions_today]]
    .map(([k,v])=>`<span class="chip">${k} <b>${v}</b></span>`).join("")
    +Object.entries(s.references||{}).map(([k,v])=>`<span class="chip">ref ${k} <b>${v}</b></span>`).join("");
}
function renderTabs(){
  $("tabs").innerHTML=FILTERS.map(([k,l])=>`<button data-f="${k}" class="${state.filter===k?"on":""}">${l}</button>`).join("");
  $("tabs").querySelectorAll("button").forEach(b=>b.onclick=()=>{state.filter=b.dataset.f;reload();});
}
async function reload(){state.offset=0;state.items=[];state.sel=-1;renderTabs();await loadMore(true);}
async function loadMore(reset){
  const u=new URLSearchParams({filter:state.filter,q:state.q,sort:state.sort,limit:state.limit,offset:state.offset});
  const j=await api("/api/profiles?"+u);
  state.total=j.total;state.items=reset?j.profiles:state.items.concat(j.profiles);state.offset=state.items.length;
  renderGrid();
}
function card(p,i){
  const img=p.cover_photo_id?`<img loading="lazy" src="/photo/${encodeURIComponent(p.cover_photo_id)}" alt="">`:"👤";
  const age=p.age?` ${p.age}`:"";
  const dist=p.distance_km!=null?`${Math.round(p.distance_km)} km`:"";
  return `<div class="card ${i===state.sel?"sel":""}" data-i="${i}">
    <div class="ph">${img}</div>${badge(p)}${p.verified?'<span class="ver" title="verified">✓</span>':""}
    <div class="quick"><button data-a="like" title="like">👍</button><button data-a="nope" title="nope">👎</button><button data-a="del" title="delete">🗑</button></div>
    <div class="meta"><div class="name">${esc(p.name)||"<i>unknown</i>"}${age}</div>
    <div class="sub"><span>${dist}</span><span>${p.stored_photos||0} photos</span></div></div></div>`;
}
function renderGrid(){
  $("grid").innerHTML=state.items.map(card).join("");
  $("empty").hidden=state.items.length>0;
  $("more").hidden=state.items.length>=state.total;
  $("moreBtn").textContent=`load more (${state.items.length}/${state.total})`;
  $("grid").querySelectorAll(".card").forEach(c=>{
    c.onclick=e=>{const a=e.target.closest("button[data-a]");const i=+c.dataset.i;
      if(a){e.stopPropagation();quick(i,a.dataset.a);}else open(i);};
  });
}
async function quick(i,a){
  const p=state.items[i];
  if(a==="del")return del(i);
  const j=await api(`/api/profiles/${encodeURIComponent(p.id)}/label`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({label:a==="like"?1:0})});
  Object.assign(p,{label:j.decisions[0].label,action:j.decisions[0].action,source:j.decisions[0].source,score:j.decisions[0].score,reasons:j.decisions[0].reasons});
  toast(`${p.name||p.id}: ${a}`);renderGrid();loadStats();
  if(state.detail&&state.detail.id===p.id)renderDetail(j);
}
async function del(i){
  const p=state.items[i];
  if(!confirm(`Delete ${p.name||p.id} (photos, embeddings and decisions)?`))return;
  await api(`/api/profiles/${encodeURIComponent(p.id)}`,{method:"DELETE"});
  state.items.splice(i,1);state.total--;state.offset--;toast("deleted");loadStats();
  if(state.detail&&state.detail.id===p.id){ if(state.items.length) open(Math.min(i,state.items.length-1)); else closeDrawer(); }
  else renderGrid();
}
async function open(i){
  if(i<0||i>=state.items.length)return;
  state.sel=i;renderGrid();
  const p=state.items[i];
  const d=await api(`/api/profiles/${encodeURIComponent(p.id)}`);
  renderDetail(d);$("drawer").classList.add("open");
  document.querySelector(`.card[data-i="${i}"]`)?.scrollIntoView({block:"nearest"});
  if(i>=state.items.length-8&&state.items.length<state.total)loadMore(false);
}
function renderDetail(d){
  state.detail=d;const latest=d.decisions[0]||{};
  $("dTitle").textContent=`${d.name||"unknown"}${d.age?" "+d.age:""}${d.verified?" ✓":""}`;
  $("dBadge").innerHTML=badge({...latest,label:latest.label,reasons:latest.reasons});
  const ph=d.photos.filter(p=>p.available);
  $("gallery").innerHTML=ph.length?ph.map(p=>`<img src="/photo/${encodeURIComponent(p.id)}" title="faces: ${p.face_count??"?"} · quality ${p.quality!=null?Number(p.quality).toFixed(2):"?"}">`).join(""):'<div class="nophoto">no stored photos</div>';
  $("dInfo").innerHTML=[d.distance_km!=null?`${Math.round(d.distance_km)} km away`:"",
    `${d.photos.length} stored photos`,`first seen ${fmtTs(d.first_seen)}`,`<code>${esc(d.id)}</code>`].filter(Boolean).join(" · ");
  $("dBio").textContent=d.bio||"—";
  const tags=[...(d.jobs||[]),...(d.schools||[]),...(d.interests||[])];
  $("dTags").innerHTML=tags.length?`<h3>details</h3><div class="tags">${tags.map(t=>`<span>${esc(t)}</span>`).join("")}</div>`:"";
  $("dDecisions").innerHTML=d.decisions.length?`<table><tr><th>when</th><th>action</th><th>label</th><th>p</th><th>source</th><th>reasons</th></tr>`+
    d.decisions.map(x=>`<tr><td>${fmtTs(x.ts)}</td><td>${x.action}</td><td>${x.label==null?"—":x.label?"like":"nope"}</td><td class="num">${x.score!=null?Number(x.score).toFixed(3):"—"}</td><td>${x.source||""}</td><td>${esc((x.reasons||[]).join(", "))}</td></tr>`).join("")+"</table>":'<span style="color:var(--muted)">no decision yet</span>';
  const f=latest.features;
  $("dFeatures").innerHTML=f&&typeof f==="object"?`<details><summary>features (${Object.keys(f).length})</summary><table>${Object.entries(f).map(([k,v])=>`<tr><td>${esc(k)}</td><td class="num">${typeof v==="number"?v.toFixed(3):esc(v)}</td></tr>`).join("")}</table></details>`:"";
  $("aLike").disabled=latest.label===1;$("aNope").disabled=latest.label===0;
}
function closeDrawer(){$("drawer").classList.remove("open");state.detail=null;}
$("back").onclick=closeDrawer;$("close").onclick=closeDrawer;
$("prev").onclick=()=>open(state.sel-1);$("next").onclick=()=>open(state.sel+1);
$("aLike").onclick=()=>quick(state.sel,"like");$("aNope").onclick=()=>quick(state.sel,"nope");$("aDelete").onclick=()=>del(state.sel);
$("moreBtn").onclick=()=>loadMore(false);
$("sort").onchange=e=>{state.sort=e.target.value;reload();};
let qT;$("q").oninput=e=>{clearTimeout(qT);qT=setTimeout(()=>{state.q=e.target.value;reload();},250);};
document.addEventListener("keydown",e=>{
  if(e.target.tagName==="INPUT"||e.target.tagName==="SELECT")return;
  const openD=$("drawer").classList.contains("open");
  if(e.key==="Escape"&&openD)return closeDrawer();
  if(e.key==="ArrowRight")return open(state.sel+1);
  if(e.key==="ArrowLeft")return open(state.sel-1);
  if(state.sel<0)return;
  if(e.key==="l"||e.key==="L")return quick(state.sel,"like");
  if(e.key==="n"||e.key==="N")return quick(state.sel,"nope");
  if(e.key==="Backspace"||e.key==="Delete")return del(state.sel);
});
loadStats();reload();
</script>
</body>
</html>
"""
