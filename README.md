# tinder-bot 2.0

A Tinder auto-swiper that runs **entirely on your machine**:

* **Local storage** – SQLite (`data/tinderbot.db`) + photos on disk. No cloud, no telemetry, no API keys.
* **Local models** – SCRFD face detection, ArcFace identity embeddings and CLIP image embeddings through
  `onnxruntime` (CPU is fine; CUDA is used automatically when available). Models are downloaded once.
* **Advanced likeness scoring** – identity similarity to people you like *and* whole-photo style
  similarity, zero-shot text prompts, photo-quality statistics, bio keywords, hard filters, and a
  logistic model that keeps learning from your own swipes.
* **Captcha strategy** – avoid triggering Arkose challenges in the first place (real Chrome, persistent
  profile, Patchright, human-like mouse/timing, conservative pacing) and, when one appears, hand it to you,
  wait, then back off. It never tries to solve challenges automatically (see *Scope* below).

> Automating Tinder is against Tinder's Terms of Use and can get an account banned. This is a personal
> research project; use it on your own account, at your own risk, and keep the pacing conservative.

## How it works

```
Chrome (persistent profile, Patchright) ──► tinder.com/app/recs
        │  passive listener on the app's own /v2/recs/core responses  → name, age, bio, photo URLs, verified…
        │  DOM fallback ([itemprop=name]/[itemprop=age], background-image urls)
        ▼
photos ─► SCRFD ─► ArcFace 512-d  ─┐
       ─► CLIP ViT-B/32 512-d      ├─► features (26) ─► prior (weights) ⊕ learned (logistic) ─► LIKE / NOPE
       ─► blur / face-size / group ┘        ▲
                                            └── reference sets: img/accepted, img/denied, facedir/known_faces,
                                                and every profile you already labelled
        ▼
humanised action: Bezier mouse path → click Like/Nope (keyboard fallback) → reading time, photo browsing,
occasional profile peek, micro-breaks, session breaks, daily budget.
```

Everything is written to `data/` (git-ignored): `browser-profile/` (cookies, login), `photos/<profile>/`,
`models/` (ONNX files + the learned classifier) and `tinderbot.db`.

## Install

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[stealth,notify,dev]"               # patchright + plyer + pytest
patchright install chromium                          # only needed if you don't have Google Chrome
cp config.example.toml config.toml                   # then edit
```

Google Chrome installed on the machine is strongly recommended (`browser.channel = "chrome"`); the bot
falls back to the bundled Chromium automatically.

## Use

```bash
tinderbot download-models                # ~700 MB once (face det/rec + CLIP). Fully offline afterwards.
tinderbot login                          # log in once in the window; the profile keeps the session
tinderbot enroll                         # uses img/accepted (liked), img/denied (disliked), facedir/known_faces (liked)
tinderbot enroll --liked ~/pics/yes --disliked ~/pics/no --reset
tinderbot score ~/pics/someone/*.jpg     # sanity-check the scorer offline, prints every feature
tinderbot swipe --shadow                 # YOU swipe; the bot prints its verdict and learns from your choices
tinderbot swipe --max 40                 # one automatic session (browser stays open, you watch)
tinderbot swipe --loop                   # sessions with breaks until the daily budget is used
tinderbot auto                           # FULLY UNATTENDED: random sessions per day, browser closed in between
tinderbot plan / status / resume         # see the day plan, the halt/pause state, clear a halt
tinderbot review                         # confirm/correct uncertain auto decisions (training labels)
tinderbot retrain                        # refit the learned model
tinderbot web                            # small local web app: browse photos, fix labels, delete profiles
tinderbot stats / export
```

Recommended ramp-up: `enroll` → a few `--shadow` sessions (this both validates the verdicts and labels
40+ examples so the learned model activates) → short auto sessions (`--max 30`) → `tinderbot auto`.

## Unattended mode (`tinderbot auto`)

`tinderbot auto` runs without anyone at the keyboard. Each day it draws its own plan and persists it:
maybe a **rest day** (12 %), otherwise **1-4 sessions** at random times weighted towards lunch and the
evening, each with a **right-skewed random swipe count**, the whole day bounded by a randomised budget
that **ramps up** over the first week of automation. Between sessions the **browser is closed**; at a
slot's time Chrome opens, orients for a few seconds, swipes with a per-session tempo and
keyboard/mouse preference, and closes again. Slots the machine slept through are skipped, not run late.

```
$ tinderbot plan --days 3
2026-09-07: 3 session(s), 84 swipes planned (budget 96, ramp x0.66)
  12:24   18 swipes  ~3 min  [pending]
  19:29   41 swipes  ~7 min  [pending]
  21:53   25 swipes  ~4 min  [pending]
(preview) 2026-09-08: rest day
(preview) 2026-09-09: 2 session(s), 57 swipes planned (budget 71, ramp x0.72)
```

Safety policy while unattended (details in `docs/BAN_SAFETY.md`):

* a **like-rate governor** keeps the share of right swipes among the last 50 decisions at or below
  `max_like_ratio` (0.55) by swiping left on the weakest LIKE verdicts (labels stay honest);
* a challenge **nobody solves** within `wait_for_human_max_minutes` closes the browser, cancels the rest
  of the day and pauses the bot 12-24 h; the second unsolved one in a row **halts** it;
* an **account notice** (ban / review / logged out) or a **lost login** halts immediately; the bot never
  logs in by itself;
* three **errors** in a row (browser crash, DOM change) halt; a single one just waits for the next slot;
* a halt is a persisted latch: fix the cause, then `tinderbot resume`. `tinderbot status` shows why.

Run it on a real desktop with a logged-in display, on the account's usual residential connection, and
do not swipe on the phone while a session is running.

## Likeness scoring

Per profile the extractor computes 26 features (`tinderbot/likeness/features.py`), among them:

| group | features | meaning |
|---|---|---|
| identity | `primary_face_sim_liked/disliked`, `face_sim_liked_max/top3`, `face_knn_liked_frac`, `face_margin` | ArcFace cosine similarity of the profile's *primary face* (the face recurring across photos, so friends in group shots don't count) against your liked / disliked reference faces, plus a k-NN vote |
| style | `clip_sim_liked_mean/max`, `clip_sim_disliked_max`, `clip_knn_liked_frac`, `clip_margin` | CLIP whole-photo similarity to photos you liked/disliked (setting, style, vibe) |
| taste hints | `prompt_score` | mean similarity to your `positive` prompts minus `negative` prompts (zero-shot CLIP) |
| quality | `quality_mean/max` (Laplacian sharpness), `face_photo_ratio`, `group_photo_ratio`, `no_face_ratio`, `face_size_mean`, `identity_consistency` | photo hygiene and "is this actually one person" |
| meta | `photo_count`, `bio_len`, `bio_keyword_hits`, `verified`, `age`, `distance_km` | |

Decision (`tinderbot/likeness/scorer.py`):

1. **Hard filters** – age range, max distance, verified-only, blocked bio keywords, "no detectable face".
2. **Prior** – weighted logit of the features (weights in `config.toml`), calibrated so ArcFace ≈0.30 and
   CLIP ≈0.60 are neutral. Works from the first swipe with only your reference folders.
3. **Learned model** – `StandardScaler + LogisticRegression` on the same features, trained from every
   labelled decision (shadow-mode swipes, `review`, auto decisions). Blended in with a weight that ramps
   from `min_examples` to `blend_full_at`; retrained every `retrain_every` decisions.
4. Scores within `uncertain_band` of the threshold are flagged for `tinderbot review`.

Reference sets grow automatically: every labelled profile's embeddings join the liked/disliked pools.

## Managing the database (`tinderbot web`)

`tinderbot web` starts a small web app on `http://127.0.0.1:8765/` (standard library only, nothing leaves
the machine) and opens it in your browser. It shows every stored profile with its photos, bio, score,
decision history and features, filterable by liked / noped / uncertain / reviewed / unlabelled and
searchable by name or bio. From there you can:

* **correct a label** (👍 / 👎, or `L` / `N` on the keyboard) – this relabels the profile's decisions as
  `manual`, exactly like `tinderbot review`, so the learned model and the reference pools pick it up on
  the next retrain;
* **delete a profile** (🗑 / `Backspace`) – removes its photos, embeddings and decisions from the DB and the
  photo folder from `data/photos/`;
* browse with `←` / `→`, `Esc` closes the detail panel.

Use `--port`, `--host` and `--no-open` to change where it listens; keep it on localhost, there is no auth.

## Anti-detection / captcha strategy

Tinder's challenge is Arkose Labs FunCaptcha and it is *risk-scored*: it shows up when the session looks
automated. The bot invests in not looking automated rather than in solving puzzles:

* **Browser** – real Google Chrome via Patchright (patches the CDP `Runtime.enable`/`Console.enable` leaks
  and automation flags), headed, `no_viewport`, persistent `user_data_dir`, **no** UA/fingerprint spoofing
  (inconsistent fingerprints score worse than honest ones).
* **Input** – cubic-Bezier mouse paths with minimum-jerk velocity, jitter, overshoot + correction,
  Gaussian click positions, log-normal delays, wheel scrolling in bursts, idle wiggles.
* **Behaviour** – reading time scaled by bio length, photos browsed **and the verdict** (linger on likes),
  photo browsing (Space / click with a per-session preference), occasional profile peek, micro-breaks,
  warm-up pause, random early exits, like-rate cap, daily cap (default 160/day randomised per day,
  15-80 per session right-skewed, 1-4 sessions at random times, rest days).
* **Data** – no direct API calls; the bot only reads responses the web app fetched itself and clicks the
  same buttons a user clicks.
* **When a challenge appears** – the bot detects the Arkose iframe / wording, rings the terminal bell,
  sends a desktop notification, brings the window to front and waits for **you** to solve it. It then logs
  the event, cools down 30-90 min, multiplies all pauses by 1.6, and stops for the day after
  `max_captchas_per_day`. Account-level notices (ban / review) stop the run immediately. Unattended, an
  unsolved challenge closes the browser and pauses for 12-24 h instead of being reloaded (see above).

### Scope

The bot does not integrate captcha-solving services or automated puzzle solvers, and it does not create
accounts or rotate identities/proxies. Those cross from "automate my own swiping" into abuse tooling and
also get accounts banned faster.

## Configuration

See `config.example.toml`: browser channel, pacing budgets and persona ranges, the `[schedule]` day rhythm
(rest days, hour weights, ramp-up, unattended safety limits), captcha policy, likeness filters, weights,
prompts, learning schedule and model selection (`clip-vit-base-patch32`, `-q8` for a fast INT8 variant,
`clip-vit-large-patch14`, or `siglip-base-patch16-224`).

If Tinder changes its DOM, adjust `SELECTORS`, `POPUP_DISMISS_TEXTS` in `tinderbot/browser/tinder_page.py`;
the like/nope actions already fall back to Tinder's keyboard shortcuts (← / →).

## Development

```bash
pytest -q          # 64 tests: storage, recs parsing, humanised paths, SCRFD decoding, likeness pipeline
                   # on synthetic ONNX models, page driver + captcha policy on a mock Tinder page (Chromium),
                   # pacing/like governor, day planner + unattended scheduler on a simulated clock
```

`docs/RESEARCH.md` documents the research behind the model and stealth choices, with sources.

## Migration from 1.x

The old `functions.py`/`tinder_bot.py` (dlib `face_recognition`, screen-grab capture, Selenium + Facebook
login, `dataset_faces.dat`) are gone. Your `img/accepted`, `img/denied` and `facedir/known_faces` folders
are still the input: run `tinderbot enroll` to build the new reference sets. The pickled dlib encodings
cannot be converted (different embedding space) and were removed.
