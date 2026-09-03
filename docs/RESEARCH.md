# Research notes (September 2026)

What was checked before rebuilding the project, and which finding drove which decision.

## 1. Fully local storage and models

**Storage.** SQLite in WAL mode is the simplest robust local store for a single-user desktop tool: one
file, concurrent reader (stats/review) while the bot writes, embeddings as float32 BLOBs (a 512-d vector
is 2 KB; 100k photos ≈ 200 MB). No vector DB is needed at this scale: numpy cosine similarity over a few
thousand reference vectors takes microseconds. Photos are kept as files under `data/photos/<profile>/`
so they can be reviewed with any image viewer.

**Face identity.** InsightFace's `buffalo_l` pack (SCRFD-10G detector + ArcFace ResNet-50 trained on
WebFace600K) is still the de-facto open standard for face embeddings; ArcFace (additive angular margin)
and AdaFace (quality-adaptive margin) are the two mainstream losses and both ship as ONNX
([insightface](https://github.com/deepinsight/insightface),
[AdaFace](https://github.com/mk-minchul/adaface),
[OpenVINO model zoo ArcFace ONNX](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/face-recognition-resnet100-arcface-onnx/README.md)).
ONNX Runtime on CPU is fast enough (tens of ms per face) and INT8 quantisation costs ~0.02 % embedding
error, so no GPU is required. The `insightface` pip package needs a C++ toolchain, so the small pre/post
processing (anchor decoding, NMS, 5-point similarity alignment to the 112×112 ArcFace template) was
re-implemented in `tinderbot/models/face.py` with numpy + OpenCV only. Model files are pulled from the
Immich mirror on Hugging Face (actively maintained) with the original GitHub release zip as a fallback.

**Whole-image similarity.** CLIP-family embeddings capture "vibe" (setting, style, composition) that face
identity misses. SigLIP/SigLIP 2 (sigmoid pairwise loss, Feb 2025) beat CLIP on retrieval at every
scale, and ONNX exports exist for CLIP ViT-B/32, ViT-L/14 and SigLIP base
([SigLIP 2 blog](https://huggingface.co/blog/siglip2), [SigLIP docs](https://huggingface.co/docs/transformers/en/model_doc/siglip),
[vision-at-a-clip ONNX](https://github.com/rhysdg/vision-at-a-clip),
[MobileCLIP2 ONNX](https://huggingface.co/RuteNL/MobileCLIP2-S2-OpenCLIP-ONNX)).
Default is CLIP ViT-B/32 (small, well-tested export with text tower for prompts); the registry also lists
the INT8 variant, ViT-L/14 and SigLIP-base, selectable in `config.toml`. Text prompts run through the
`tokenizers` library with the export's own `tokenizer.json`.

## 2. Profile likeness for auto-liking

Design decisions taken from the literature on few-shot preference learning and from the data Tinder's own
web app exposes:

* **Structured profile data without scraping.** The web app fetches `https://api.gotinder.com/v2/recs/core`
  (now also `/v3/recs/core`); each result carries `user._id`, `name`, `bio`, `birth_date`, `photos[]` with
  `processedFiles[{url,width,height}]`, `badges` (`selfie_verified`), `jobs`, `schools`, interests, and
  `distance_mi` + `s_number` ([auto-tinder](https://github.com/joelbarmettlerUZH/auto-tinder),
  [Tinder API gist](https://gist.github.com/rtt/10403467),
  [Burp exploration](https://medium.com/@isiah_lloyd/exploring-tinders-api-using-burp-suite-175e38ced769)).
  Playwright's response events give us that JSON passively
  ([Playwright network interception](https://oneuptime.com/blog/post/2026-02-02-playwright-network-interception/view)),
  so no extra API calls are made and the like/pass still go through the real buttons.
* **Primary-face logic.** Group photos are common; scoring "any face" against liked references rewards a
  photogenic friend. The extractor picks the face that recurs across the profile (highest mean
  similarity to the other photos' largest faces) as the identity to score, and `identity_consistency`
  penalises profiles whose largest faces are all different people.
* **Two similarity families + kNN vote.** Max/top-3 cosine to the liked set, max to the disliked set,
  their margin, and a k-NN vote over the pooled labelled vectors (k ≤ half the pool) for both face and
  CLIP spaces. Calibration constants (ArcFace ≈0.30 neutral, ≥0.5 same person; CLIP ≈0.60 neutral for
  portraits) come from the models' known score distributions.
* **Prior → learned blend.** A hand-weighted logistic prior gives sensible verdicts from the first swipe;
  a `LogisticRegression` (balanced, L2) on the same 26 features takes over as labels accumulate (ramp
  40→300 examples). Shadow mode records the human's real like/pass (observed as `/like/{id}` and
  `/pass/{id}` requests) as labels, which is the cheapest way to collect ground truth.

## 3. Anti-captcha techniques

**What Tinder uses.** Arkose Labs FunCaptcha (3D-object rotation, image matching), adaptive difficulty,
rendered in an iframe on `*.arkoselabs.com`. It collects UA, screen, WebGL timing, mouse telemetry and
previously issued tokens and escalates when the session looks synthetic
([Arkose principles](https://medium.com/@kentavr00000009/funcaptcha-arkose-labs-principles-of-operation-features-and-methods-for-automated-bypass-780ef786d7c5),
[uCaptcha overview](https://ucaptcha.net/blog/funcaptcha-arkose-labs/),
[Tinder iOS research, Apr 2025](https://dev.to/neverlow512/breaking-the-unbreakable-bypassing-arkose-labs-on-ios-2mnj),
[DatingZest puzzle loop](https://datingzest.com/dating-app-puzzle-verification-loop/)).
Key takeaway: the challenge is *risk-triggered*, so the lever is the risk score, not the puzzle.

**Stealth browser (2026 benchmarks).** Across the public anti-detect benchmarks, nodriver (CDP-direct,
successor of undetected-chromedriver), Patchright (Playwright fork) and Camoufox (patched Firefox) are the
open-source options; Patchright is the drop-in for Playwright code and is actively maintained, Camoufox
is strongest on hard fingerprinting targets but slow, nodriver's last push was May 2026
([benchmark: 7 tools / 31 targets](https://dev.to/ianlpaterson/anti-detect-browser-benchmark-2026-7-stealth-tools-31-cloudflare-targets-651-verdicts-4361),
[Scrapfly stealth browsers 2026](https://scrapfly.io/blog/posts/best-stealth-browsers),
[nodriver vs Patchright](https://www.scrapeless.com/en/blog/nodriver-patchright-undetected),
[OSS stealth benchmark](https://www.scraping.club/p/open-source-browser-automation-stealth-benchmark),
[Patchright README](https://github.com/Kaliiiiiiiiii-Vinyzu/patchright-python)).
Patchright's own guidance was adopted verbatim: `launch_persistent_context(user_data_dir, channel="chrome",
headless=False, no_viewport=True)` and **no custom headers/UA**. Patchright removes the `Runtime.enable`
and `Console.enable` CDP leaks and the `--enable-automation` flags; the driver falls back to plain
Playwright with `--disable-blink-features=AutomationControlled` when Patchright is absent.

**Human-like input.** Ghost-cursor style Bezier paths (random control points perpendicular to the travel
direction), Fitts-law durations, overshoot and settle, and Gaussian click positions are the accepted
approach in ghost-cursor / HumanCursor / humanization-playwright
([ghost-cursor](https://github.com/Xetera/ghost-cursor), [human-cursor](https://github.com/CloverLabsAI/human-cursor),
[humanization-playwright](https://pypi.org/project/humanization-playwright/),
[Oxymouse + Playwright](https://substack.thewebscraping.club/p/oxymouse-and-playwright-mouse-movements)).
`tinderbot/browser/humanize.py` implements this without an extra dependency and adds a minimum-jerk
velocity profile, log-normal pauses, wheel-burst scrolling and idle wiggles.

**Behavioural pacing.** Older open-source bots (TinderBotz and friends) recommend ≥1 s between swipes and
randomised behaviour ([TinderBotz](https://github.com/frederikme/TinderBotz)); ban reports cluster around
high-volume, uniform-cadence swiping. Defaults here: 1.2-4.5 s base reading time plus per-photo and
per-bio-character time, 65 % chance to browse photos, 15 % to open the profile, micro-breaks, 40-90 swipes
per session, 3 sessions/day with 20-75 min breaks, 200 swipes/day cap, active hours 09-23.

**Human-in-the-loop instead of solvers.** Commercial "FunCaptcha solver" APIs exist
([2captcha](https://2captcha.com/p/funcaptcha), [CapMonster](https://capmonster.cloud/en/funcaptcha/)) but they
need the session's `blob` and token, are detected by Arkose's token/timing checks, and turn a personal
swiping tool into account-farming infrastructure. The bot therefore detects the challenge (iframe host or
wording), notifies, waits for the human, and applies a cooldown + slowdown + daily challenge cap. Solving
it inside the same, aged, real-Chrome profile is exactly what a human user would do and keeps the risk
score low afterwards.

## Selectors used (Tinder web, 2025-2026)

* Like / Nope: `button[aria-label="Like"]`, `button[aria-label="Nope"]`
  ([tinder-auto-swiper](https://github.com/OrchaniousS/tinder-auto-swiper)), fallbacks on the hidden
  `span.Hidden` labels and keyboard shortcuts → / ←.
* Name / age: schema.org microdata `[itemprop="name"]`, `[itemprop="age"]`.
* Photos: inline `background-image` URLs on `images-ssl.gotinder.com`.
* Popups: `button[title="Back to Tinder"]`, dialog close buttons and "No Thanks / Not interested / Maybe
  later / Keep swiping" texts (TinderBotz's positional XPaths were replaced by text/role matching).
* Captcha: any frame whose URL contains `arkoselabs` / `funcaptcha`, or wording such as "verify you're
  human", "let's make sure you're a real person".
