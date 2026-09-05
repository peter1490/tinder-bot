"""Session orchestration: browser -> card -> photos -> local models -> verdict -> humanised swipe.

Three modes:
  * auto     : the bot swipes according to the verdict, within the pacing budget (``swipe``)
  * shadow   : YOU swipe; the bot scores every card, prints its verdict, and records your real
               like/pass (observed on the network) as a training label. Best way to bootstrap the
               learned model without giving the bot control yet.
  * session  : one self-contained unattended session (open browser, swipe N, close browser) with a
               typed outcome, used by the day scheduler (``tinderbot auto``)
"""

from __future__ import annotations

import contextlib
import random
import time
from dataclasses import dataclass, field

from rich.console import Console

from .browser.captcha import CaptchaPolicy, detect_challenge
from .browser.driver import launch
from .browser.pacing import LikeGovernor, Pacer
from .browser.recs import dom_profile_id
from .browser.tinder_page import RECS_URL, CardInfo, TinderPage
from .config import Config
from .likeness.features import FeatureExtractor
from .likeness.scorer import Scorer, Verdict
from .models.loader import Models
from .storage import ProfileRecord, Storage

console = Console(stderr=True)

# How a session ended.  The scheduler decides what each one means for the rest of the day.
STOP_PLANNED = "planned"                 # all planned swipes done
STOP_EARLY = "ended_early"               # random "got bored" exit
STOP_OUT_OF_LIKES = "out_of_likes"       # Tinder's daily like cap
STOP_NO_CARD = "no_card"                 # nothing to swipe / DOM changed
STOP_UNCONFIRMED = "swipe_unconfirmed"   # click not reflected by a new card
STOP_CAPTCHA_SOLVED = "captcha_solved"   # a human solved a challenge during the session
STOP_CAPTCHA_UNSOLVED = "captcha_unsolved"
STOP_CAPTCHA_LIMIT = "captcha_limit"     # max_captchas_per_day reached
STOP_ACCOUNT = "account_notice"          # ban / review / logged out wording on screen
STOP_NEEDS_LOGIN = "needs_login"
STOP_ERROR = "error"
STOP_BUDGET = "budget"                   # nothing left in today's budget

HALTING = {STOP_ACCOUNT, STOP_NEEDS_LOGIN}
DAY_ENDING = {STOP_OUT_OF_LIKES, STOP_CAPTCHA_UNSOLVED, STOP_CAPTCHA_LIMIT, STOP_BUDGET}


@dataclass
class RunStats:
    liked: int = 0
    noped: int = 0
    skipped: int = 0
    captchas: int = 0
    downgraded: int = 0   # LIKE verdicts turned into NOPE by the like-ratio governor

    @property
    def total(self) -> int:
        return self.liked + self.noped


@dataclass
class SessionResult:
    planned: int
    done: int
    reason: str
    error: str = ""
    stats: RunStats = field(default_factory=RunStats)

    @property
    def halting(self) -> bool:
        return self.reason in HALTING


class Runner:
    def __init__(self, cfg: Config, storage: Storage, models: Models, seed: int | None = None):
        self.cfg = cfg
        self.storage = storage
        self.models = models
        self.rng = random.Random(seed)
        self.extractor = FeatureExtractor(cfg, models, storage)
        self.scorer = Scorer(cfg, storage)
        self.pacer = Pacer(cfg.pacing, storage, self.rng)
        self.governor = LikeGovernor(cfg.pacing, storage)
        self.captcha = CaptchaPolicy(cfg.captcha, storage, self.rng)
        self.stats = RunStats()
        self.stop_reason: str = STOP_PLANNED

    # ---- profile resolution ----------------------------------------------------------
    def resolve_profile(self, tp: TinderPage, card: CardInfo) -> ProfileRecord:
        rec = tp.queue.match(card.name, card.age, card.photo_urls)
        if rec is None:
            rec = ProfileRecord(id=dom_profile_id(card.name, card.age, card.photo_urls[0] if card.photo_urls else None),
                                name=card.name, age=card.age, photo_urls=list(card.photo_urls))
        elif not rec.photo_urls and card.photo_urls:
            rec.photo_urls = list(card.photo_urls)
        return rec

    def score_profile(self, tp: TinderPage, profile: ProfileRecord, max_photos: int = 6) -> tuple[Verdict, dict]:
        images: list[bytes] = []
        for url in profile.photo_urls[:max_photos]:
            data = tp.fetch_photo(url)
            if data:
                images.append(data)
        if not images:
            shot = tp.screenshot_card()
            if shot:
                images.append(shot)
        analysis = self.extractor.analyse_profile(profile, images, persist=True)
        feats = self.extractor.features(analysis)
        verdict = self.scorer.decide(profile, feats)
        return verdict, feats

    # ---- main loops --------------------------------------------------------------------
    def _ensure_logged_in(self, tp: TinderPage, wait_minutes: float = 15.0) -> bool:
        tp.goto_recs()
        if tp.wait_ready(timeout_s=15 if wait_minutes > 0 else 30):
            return True
        if detect_challenge(tp.page) is not None:
            # a challenge or account notice is what is blocking the recs screen; let the session
            # loop handle it (it knows how to notify / halt)
            return True
        if wait_minutes <= 0:
            # Unattended: never wait for a human to log in.  Still inside the app (e.g. "no one new
            # around you") counts as logged in; the session then ends softly with no_card.
            return "/app/" in (tp.page.url or "")
        console.print("[yellow]Not logged in (or the recs screen is not visible). Log in in the browser window; "
                      "the persistent profile keeps the session for next time.[/yellow]")
        tp.page.goto("https://tinder.com/", wait_until="domcontentloaded")
        if not tp.wait_for_login(max_minutes=wait_minutes):
            console.print("[red]Timed out waiting for login.[/red]")
            return False
        tp.goto_recs()
        return tp.wait_ready(timeout_s=30)

    def _check_challenge(self, tp: TinderPage, unattended: bool = False) -> str:
        ch = detect_challenge(tp.page)
        if ch is None:
            return "ok"
        self.stats.captchas += 1
        outcome = self.captcha.handle(tp.page, ch)
        if ch.kind == "account":
            self.stop_reason = STOP_ACCOUNT
            return "stop"
        if outcome == "solved":
            self.pacer.slowdown = self.captcha.slowdown
            if unattended:
                # Solved by whoever was at the machine.  End the session here; the scheduler
                # applies the cooldown with the browser closed instead of idling on the page.
                self.stop_reason = STOP_CAPTCHA_SOLVED
                return "stop"
            cd = self.captcha.cooldown_seconds()
            console.print(f"[yellow]Challenge solved. Cooling down {cd / 60:.0f} min and slowing pace x{self.pacer.slowdown:.1f}.[/yellow]")
            time.sleep(cd)
            tp.goto_recs()
            tp.wait_ready()
            return "ok"
        self.stop_reason = STOP_CAPTCHA_UNSOLVED if outcome == "timeout" else STOP_CAPTCHA_LIMIT
        return "stop"

    def run_auto(self, max_swipes: int | None = None, once: bool = True) -> RunStats:
        """Interactive auto mode (``tinderbot swipe``): the browser stays open between sessions."""
        session = launch(self.cfg, RECS_URL)
        try:
            tp = TinderPage(session.page, self.rng)
            console.print(f"[green]Browser: {session.backend}[/green]  references: {self.extractor.reference_summary()}")
            if not self._ensure_logged_in(tp):
                return self.stats
            while True:
                wait = self.pacer.seconds_until_active()
                if wait > 0:
                    console.print(f"Outside active hours; sleeping {wait / 60:.0f} min")
                    time.sleep(wait)
                plan = self.pacer.start_session(max_swipes)
                if plan is None:
                    console.print("Daily budget or session count reached. Done for today.")
                    break
                tp.keyboard_pref = self.pacer.persona.keyboard_pref
                console.print(f"Session: {plan.swipes} swipes planned")
                done = self._swipe_session(tp, plan.swipes)
                self.pacer.end_session(done, self.stop_reason)
                if once or done < plan.swipes:
                    break
                b = self.pacer.break_seconds()
                console.print(f"Break for {b / 60:.0f} min")
                time.sleep(b)
        finally:
            session.close()
        return self.stats

    def run_session(self, swipes: int, login_wait_minutes: float = 0.0) -> SessionResult:
        """One unattended session: open the browser, swipe up to ``swipes`` cards, close the browser.

        Never raises: every failure is reported through ``SessionResult.reason`` so the scheduler
        can back off, cancel the day or halt the bot.
        """
        self.stats = RunStats()
        self.stop_reason = STOP_PLANNED
        plan = self.pacer.begin_session(swipes)
        if plan is None:
            return SessionResult(planned=0, done=0, reason=STOP_BUDGET, stats=self.stats)
        done = 0
        session = None
        result_error = ""
        try:
            session = launch(self.cfg, RECS_URL)
            tp = TinderPage(session.page, self.rng, keyboard_pref=self.pacer.persona.keyboard_pref)
            console.print(f"[green]Browser: {session.backend}[/green]  session of {plan.swipes} swipes  "
                          f"tempo x{self.pacer.persona.tempo:.2f}  references: {self.extractor.reference_summary()}")
            if not self._ensure_logged_in(tp, wait_minutes=login_wait_minutes):
                self.stop_reason = STOP_NEEDS_LOGIN
            else:
                self._warm_up(tp)
                done = self._swipe_session(tp, plan.swipes, unattended=True)
                # a short, natural pause before "closing the app"
                time.sleep(min(10.0, self.pacer.post_action_delay() * 3))
        except Exception as e:  # browser crashed, network down, DOM exploded, ...
            self.stop_reason = STOP_ERROR
            self.storage.log_event("session_error", {"error": f"{type(e).__name__}: {e}"[:500]})
            console.print(f"[red]Session error: {type(e).__name__}: {e}[/red]")
            result_error = f"{type(e).__name__}: {e}"
        finally:
            if session is not None:
                with contextlib.suppress(Exception):
                    session.close()
            self.pacer.end_session(done, self.stop_reason)
        self.stats.downgraded = self.governor.downgraded
        return SessionResult(planned=plan.swipes, done=done, reason=self.stop_reason, error=result_error,
                             stats=self.stats)

    def _warm_up(self, tp: TinderPage) -> None:
        """Orient like a person who just opened the app: dismiss prompts, glance around, wait a bit."""
        tp.dismiss_popups()
        try:
            vp = tp.page.viewport_size or {"width": 1280, "height": 800}
            tp.mouse.wander(vp["width"], vp["height"])
        except Exception:
            pass
        time.sleep(self.pacer.warmup_seconds())

    def _swipe_session(self, tp: TinderPage, n: int, unattended: bool = False) -> int:
        done = 0
        last_key = ""
        self.stop_reason = STOP_PLANNED
        while done < n:
            if self._check_challenge(tp, unattended) == "stop":
                return done
            tp.dismiss_popups()
            if tp.out_of_likes():
                console.print("[yellow]Tinder says you are out of likes. Ending session.[/yellow]")
                self.stop_reason = STOP_OUT_OF_LIKES
                return done
            card = tp.current_card()
            if not card.name:
                card = tp.wait_for_new_card(last_key, timeout_s=15)
                if card is None:
                    console.print("[red]No card visible; stopping the session.[/red]")
                    self.stop_reason = STOP_NO_CARD
                    return done
            last_key = card.key
            profile = self.resolve_profile(tp, card)
            verdict, feats = self.score_profile(tp, profile)
            action_like = verdict.like
            reasons = list(verdict.reasons)
            if action_like and not self.governor.allow_like():
                action_like = False
                reasons.append("like_ratio_cap")
            # look at the card like a person would (the scoring above already took a moment)
            plan = self.pacer.plan_profile(len(profile.photo_urls) or 1, len(profile.bio or ""), like=action_like)
            if plan["browse_photos"]:
                tp.browse_photos(plan["browse_photos"])
            if plan["open_profile"]:
                tp.peek_profile()
            if self.rng.random() < 0.5:
                tp.mouse.wiggle()
            time.sleep(plan["read_seconds"])
            ok = tp.like() if action_like else tp.nope()
            if not ok:
                self.stats.skipped += 1
                tp.dismiss_popups()
                continue

            # A successful click only means the input was dispatched.  Do not
            # count or persist it until Tinder visibly advances to another
            # card.  Otherwise a slow animation or blocking dialog can cause
            # the same profile to be scored and clicked repeatedly.
            time.sleep(self.pacer.post_action_delay())
            tp.dismiss_popups()
            next_card = tp.wait_for_new_card(last_key, timeout_s=12)
            if next_card is None:
                console.print("[yellow]Card transition stalled; refreshing once to verify the swipe.[/yellow]")
                if tp.reload_recs(timeout_s=30):
                    next_card = tp.wait_for_new_card(last_key, timeout_s=5)
            if next_card is None:
                self.stats.skipped += 1
                self.storage.log_event(
                    "swipe_unconfirmed",
                    {"profile_id": profile.id, "action": "like" if action_like else "nope", "card_key": last_key},
                )
                console.print(
                    "[yellow]Swipe was not confirmed by a new card; stopping to avoid a duplicate action.[/yellow]"
                )
                self.stop_reason = STOP_UNCONFIRMED
                return done

            # The training label follows the scorer's verdict, the action records what was done.
            self.storage.add_decision(profile.id, "like" if action_like else "nope", verdict.score, "auto",
                                      reasons, feats, label=1 if verdict.like else 0)
            tp.queue.pop(profile.id)
            self.scorer.maybe_retrain()
            done += 1
            if action_like:
                self.stats.liked += 1
            else:
                self.stats.noped += 1
            tag = "[bold green]LIKE[/bold green]" if action_like else "[bold red]NOPE[/bold red]"
            console.print(f"{tag} {profile.name} {profile.age or ''}  p={verdict.score:.2f}  {', '.join(reasons)}")
            if plan["micro_break"]:
                time.sleep(plan["micro_break"])
            if done < n and self.pacer.should_end_early(done, n):
                console.print("[dim]Ending the session early (natural variation).[/dim]")
                self.stop_reason = STOP_EARLY
                return done
        return done

    def run_shadow(self, max_cards: int | None = None) -> RunStats:
        """Score cards while the human swipes; record the human's like/pass as labels."""
        session = launch(self.cfg, RECS_URL)
        try:
            tp = TinderPage(session.page, self.rng)
            console.print("[cyan]Shadow mode: swipe yourself in the browser; verdicts are printed here and your "
                          "choices become training labels.[/cyan]")
            if not self._ensure_logged_in(tp):
                return self.stats
            last_key = ""
            pending: dict[str, tuple[ProfileRecord, Verdict, dict]] = {}
            seen = 0
            while max_cards is None or seen < max_cards:
                ch = detect_challenge(tp.page)
                if ch is not None:
                    console.print(f"[yellow]Challenge visible ({ch.detail}); solve it, I will keep watching.[/yellow]")
                    time.sleep(5)
                    continue
                card = tp.current_card()
                if card.name and card.key != last_key:
                    last_key = card.key
                    profile = self.resolve_profile(tp, card)
                    verdict, feats = self.score_profile(tp, profile)
                    pending[profile.id] = (profile, verdict, feats)
                    seen += 1
                    tag = "[green]would LIKE[/green]" if verdict.like else "[red]would NOPE[/red]"
                    console.print(f"{tag} {profile.name} {profile.age or ''} p={verdict.score:.2f} {', '.join(verdict.reasons)}")
                while tp.human_actions:
                    action, pid = tp.human_actions.pop(0)
                    entry = pending.pop(pid, None)
                    label = 1 if action == "like" else 0
                    if entry is None:  # DOM-id fallback: attribute to the most recent pending card
                        if not pending:
                            continue
                        pid, entry = pending.popitem()
                    profile, verdict, feats = entry
                    self.storage.add_decision(profile.id, "like" if label else "nope", verdict.score, "manual",
                                              verdict.reasons, feats, label=label)
                    agree = (label == 1) == verdict.like
                    console.print(f"   you: {'LIKE' if label else 'NOPE'} -> {'agree' if agree else 'DISAGREE'}")
                    self.scorer.maybe_retrain()
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            session.close()
        return self.stats
