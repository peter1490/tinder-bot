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
tinderbot swipe --max 40                 # one automatic session
tinderbot swipe --loop                   # sessions with breaks until the daily budget is used
tinderbot review                         # confirm/correct uncertain auto decisions (training labels)
tinderbot retrain                        # refit the learned model
tinderbot stats / export
```

Recommended ramp-up: `enroll` → a few `--shadow` sessions (this both validates the verdicts and labels
40+ examples so the learned model activates) → short auto sessions (`--max 30`) → `--loop`.

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

## Anti-detection / captcha strategy

Tinder's challenge is Arkose Labs FunCaptcha and it is *risk-scored*: it shows up when the session looks
automated. The bot invests in not looking automated rather than in solving puzzles:

* **Browser** – real Google Chrome via Patchright (patches the CDP `Runtime.enable`/`Console.enable` leaks
  and automation flags), headed, `no_viewport`, persistent `user_data_dir`, **no** UA/fingerprint spoofing
  (inconsistent fingerprints score worse than honest ones).
* **Input** – cubic-Bezier mouse paths with minimum-jerk velocity, jitter, overshoot + correction,
  Gaussian click positions, log-normal delays, wheel scrolling in bursts, idle wiggles.
* **Behaviour** – reading time scaled by bio length and photos browsed, photo browsing (Space / click),
  occasional profile peek, micro-breaks, session breaks, active hours, daily cap (default 200/day,
  40-90 per session, 3 sessions).
* **Data** – no direct API calls; the bot only reads responses the web app fetched itself and clicks the
  same buttons a user clicks.
* **When a challenge appears** – the bot detects the Arkose iframe / wording, rings the terminal bell,
  sends a desktop notification, brings the window to front and waits for **you** to solve it. It then logs
  the event, cools down 30-90 min, multiplies all pauses by 1.6, and stops for the day after
  `max_captchas_per_day`. Account-level notices (ban / review) stop the run immediately.

### Scope

The bot does not integrate captcha-solving services or automated puzzle solvers, and it does not create
accounts or rotate identities/proxies. Those cross from "automate my own swiping" into abuse tooling and
also get accounts banned faster.

## Configuration

See `config.example.toml`: browser channel, pacing budgets, captcha policy, likeness filters, weights,
prompts, learning schedule and model selection (`clip-vit-base-patch32`, `-q8` for a fast INT8 variant,
`clip-vit-large-patch14`, or `siglip-base-patch16-224`).

If Tinder changes its DOM, adjust `SELECTORS`, `POPUP_DISMISS_TEXTS` in `tinderbot/browser/tinder_page.py`;
the like/nope actions already fall back to Tinder's keyboard shortcuts (← / →).

## Development

```bash
pytest -q          # 32 tests: storage, recs parsing, humanised paths, SCRFD decoding, likeness pipeline
                   # on synthetic ONNX models, page driver + captcha policy on a mock Tinder page (Chromium)
```

`docs/RESEARCH.md` documents the research behind the model and stealth choices, with sources.

## Migration from 1.x

The old `functions.py`/`tinder_bot.py` (dlib `face_recognition`, screen-grab capture, Selenium + Facebook
login, `dataset_faces.dat`) are gone. Your `img/accepted`, `img/denied` and `facedir/known_faces` folders
are still the input: run `tinderbot enroll` to build the new reference sets. The pickled dlib encodings
cannot be converted (different embedding space) and were removed.
