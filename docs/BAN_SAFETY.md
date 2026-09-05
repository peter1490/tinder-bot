# Ban-safety review and the unattended mode

This document records the ban-safety audit of the project done before adding the fully automatic
mode (`tinderbot auto`), what was changed because of it, and what still cannot be solved in code.

Threat model: Tinder's risk engine (plus Arkose Labs for challenges) scores a session on three
layers. Most open-source bots only address the first one.

| layer | what is scored | what this project does |
|---|---|---|
| **browser / device** | headless, automation flags, CDP leaks, fingerprint inconsistencies, fresh profiles | real Chrome, persistent profile, Patchright, headed, `no_viewport`, no spoofing (unchanged, already good) |
| **input telemetry** | teleporting cursor, constant timings, pixel-perfect clicks | Bezier paths, minimum-jerk velocity, jitter/overshoot, Gaussian click points, log-normal pauses (unchanged, already good) |
| **behaviour over time** | swipe volume, like share, cadence regularity, session timing, device co-use, how challenges are handled | **this is where the project was weak; everything below addresses it** |

## 1. Findings

Ordered by how much ban risk each one carried for an unattended bot.

1. **No cap on the like share.** The scorer decided purely on taste. With a permissive threshold or a
   good match between references and the local population it would right-swipe most cards. Liking
   nearly everyone is the most consistently reported trigger for shadow-bans ("no matches anymore")
   and for the Arkose puzzle loop, because it is exactly what spam accounts do.
2. **Clockwork day rhythm.** `swipe --loop` always started at `active_hours[0]` plus 0-40 minutes,
   always ran exactly `sessions_per_day` sessions back to back with 20-75 minute breaks, every day,
   and always tried to spend the same 200-swipe budget. Sessions clustered every morning at 09:xx;
   no rest days, no evening peak, no weekend difference.
3. **Browser open all day.** Between sessions and while waiting for active hours the Tinder tab stayed
   open (websocket/heartbeat traffic for hours with no interaction), which no phone user does.
4. **Unattended challenge handling was undefined.** On a challenge the bot waited 30 minutes for a
   human with the puzzle iframe open, then stopped the run. Nothing stopped the next run from loading
   the challenge again the next day. Repeatedly loading and abandoning Arkose challenges is the
   strongest possible "bot" signal and is how accounts end up in the permanent puzzle loop.
5. **No latch for account-level notices or a lost login.** A "we've noticed unusual activity" banner or
   a logged-out session stopped the current run only; the next run would try again.
6. **Uniform session sizes and cadence.** Session length was uniform in 40-90; humans have many short
   sessions and a few long ones. Dwell time ignored the verdict, while real users linger on profiles
   they like and flick past the rest. Photo browsing and like/nope clicks mixed keyboard and mouse
   50/50 per action; people are consistent within a session and differ between sessions.
7. **Photo downloads outside the page.** Profile photos were fetched with Playwright's request
   context (Node-side fetch with the context's cookies): same cookies, different header set, no
   HTTP cache, and up to six photos per card even when the page only rendered one.
8. **Budget bookkeeping gaps.** Shadow-mode swipes did not count towards the daily budget; the captcha
   day counter used UTC midnight while the swipe counter used local midnight.
9. **No ramp-up.** A freshly automated account went straight to full volume.
10. **Uncaught exceptions.** A crashed browser or a DOM change raised out of the run loop; in a daemon
    that either kills the process or, worse, retries in a tight loop.

Things checked and found fine: the network listener is passive (no direct API calls), the like/nope
actions go through the real buttons with keyboard fallback, swipes are only counted once Tinder
visibly advances (no duplicate actions), the challenge is never solved automatically, no proxy or
identity rotation exists.

## 2. What changed

### Like-rate governor (`LikeGovernor`, `tinderbot/browser/pacing.py`)
The share of LIKEs among the last `like_ratio_window` auto decisions is kept at or below
`max_like_ratio` (default 0.55). When a LIKE would push it over, the card is swiped left instead and
the decision is stored with `like_ratio_cap` in its reasons. The **training label still records the
scorer's verdict** so the learned model is not taught to dislike what it liked.

### Day planner and scheduler (`tinderbot/schedule.py`, `tinderbot auto`)
Every calendar day gets its own random plan, persisted in the database (restarts keep it):

* **rest days** (`p_rest_day`, default 12 %),
* a random **number of sessions** in `sessions_per_day` (now a range, default 1-4, weekend factor),
* **start times** drawn from an hour-of-day weight table (default: small morning bump, lunch bump,
  strong evening peak), at least `min_gap_minutes` apart, never past `active_hours`,
* **right-skewed session sizes** (log-normal in `swipes_per_session`) scaled into a **randomised day
  budget** (`max_swipes_per_day` x `day_budget_jitter`) and a **ramp-up factor** for the first
  `ramp_days` of automation.

The scheduler sleeps with the **browser closed** until the next slot, then opens Chrome, swipes,
and closes it again, like a person opening and closing the app. A slot the process wakes up too late
for (laptop asleep, reboot) is skipped rather than run late.

### Unattended safety policy (outcome of every session -> what happens next)

| session ended because | policy |
|---|---|
| planned count reached / random early exit | nothing, next slot as planned |
| no card / swipe not confirmed | nothing (soft), next slot as planned |
| out of likes / daily challenge limit / budget | rest of the day cancelled |
| challenge **nobody solved** within `wait_for_human_max_minutes` | browser closed, rest of the day cancelled, bot paused 12-24 h; second one in a row -> **halt** |
| challenge solved by whoever was at the machine | session ends, normal cooldown, streak reset, pace slowed |
| account notice (ban / review / logged out wording) | **halt** immediately |
| not logged in | **halt** immediately (never tries to log in by itself) |
| exception (browser crash, network, DOM) | logged; third in a row -> **halt** |

A halt is a persisted latch: `tinderbot auto` refuses to start until you fixed the cause and ran
`tinderbot resume`. `tinderbot status` shows the latch, the pause, today's plan and recent events.

### Inside a session
* **Per-session persona**: tempo (x0.8-1.35 on every pause) and keyboard-vs-mouse preference are drawn
  once per session, so a session is internally consistent and sessions differ from each other.
* **Verdict-aware dwell**: likes get x1.1-1.8 reading time and more photo browsing, nopes x0.5-1.0.
* **Warm-up** pause and a cursor wander after the recs screen appears; short pause before closing.
* **Random early exit** (`p_end_session_early` per swipe after half the plan).
* **Photos are fetched by the page itself** (`fetch()` with `force-cache`, i.e. the browser's cache
  and the app's own headers), with the request-context download as fallback.
* Shadow-mode swipes now count towards the daily budget; both day counters use local midnight.

## 3. What code cannot fix (operational rules)

* **Run it on a real desktop machine, logged-in display, headed Chrome.** Headless Chrome and Xvfb
  give away the game (SwiftShader WebGL, odd screen metrics). A machine that sleeps is fine: missed
  slots are skipped.
* **Keep the network identity boring and stable.** Residential connection, the same one the account
  normally uses. No datacenter IP, no rotating VPN, no shared "bot" IP.
* **Do not swipe on the phone while a session runs.** Two devices swiping at the same time is a
  device-co-use signal. Using the app normally at other times is good, not bad.
* **Age the profile.** Log in once by hand, then leave the browser profile alone. Never delete
  `data/browser-profile/` to "start fresh"; a fresh profile scores worse, not better.
* **Solve challenges yourself, in that window, when you are around.** One solved challenge from an
  aged real-Chrome profile is normal; the bot then slows down. Two unsolved ones halt the bot on
  purpose. If challenges keep coming, stop for a week; do not lower the wait time.
* **Start small.** `ramp_days`/`ramp_start` do that automatically; also keep `max_like_ratio` low and
  use `tinderbot swipe --shadow` first so the scorer's verdicts are worth acting on.
* **Watch the match rate.** A sudden drop in matches with unchanged volume is the usual sign of a
  shadow-ban; there is no reliable way to detect it from the web app, so look at it yourself.

## 4. Residual risks

* Tinder can still decide the account is automated from signals we cannot see (server-side device
  history, the account's age, the phone number's reputation). The scheduler makes the behaviour
  plausible; it does not make it invisible.
* Any DOM change can end sessions with `no_card`; the scheduler treats that as harmless, but three
  crashes in a row halt the bot, so check `tinderbot status` when it goes quiet.
* The like governor caps the share, not the absolute number of likes; the budget and Tinder's own
  like cap do that.
