"""Session orchestration: browser -> card -> photos -> local models -> verdict -> humanised swipe.

Two modes:
  * auto   : the bot swipes according to the verdict, within the pacing budget
  * shadow : YOU swipe; the bot scores every card, prints its verdict, and records your real
             like/pass (observed on the network) as a training label. Best way to bootstrap the
             learned model without giving the bot control yet.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from rich.console import Console

from .browser.captcha import CaptchaPolicy, detect_challenge
from .browser.driver import launch
from .browser.pacing import Pacer
from .browser.recs import dom_profile_id
from .browser.tinder_page import RECS_URL, CardInfo, TinderPage
from .config import Config
from .likeness.features import FeatureExtractor
from .likeness.scorer import Scorer, Verdict
from .models.loader import Models
from .storage import ProfileRecord, Storage

console = Console(stderr=True)


@dataclass
class RunStats:
    liked: int = 0
    noped: int = 0
    skipped: int = 0
    captchas: int = 0

    @property
    def total(self) -> int:
        return self.liked + self.noped


class Runner:
    def __init__(self, cfg: Config, storage: Storage, models: Models, seed: int | None = None):
        self.cfg = cfg
        self.storage = storage
        self.models = models
        self.rng = random.Random(seed)
        self.extractor = FeatureExtractor(cfg, models, storage)
        self.scorer = Scorer(cfg, storage)
        self.pacer = Pacer(cfg.pacing, storage, self.rng)
        self.captcha = CaptchaPolicy(cfg.captcha, storage, self.rng)
        self.stats = RunStats()

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
    def _ensure_logged_in(self, tp: TinderPage) -> bool:
        tp.goto_recs()
        if tp.wait_ready(timeout_s=15):
            return True
        console.print("[yellow]Not logged in (or the recs screen is not visible). Log in in the browser window; "
                      "the persistent profile keeps the session for next time.[/yellow]")
        tp.page.goto("https://tinder.com/", wait_until="domcontentloaded")
        if not tp.wait_for_login(max_minutes=15):
            console.print("[red]Timed out waiting for login.[/red]")
            return False
        tp.goto_recs()
        return tp.wait_ready(timeout_s=30)

    def _check_challenge(self, tp: TinderPage) -> str:
        ch = detect_challenge(tp.page)
        if ch is None:
            return "ok"
        self.stats.captchas += 1
        outcome = self.captcha.handle(tp.page, ch)
        if outcome == "solved":
            self.pacer.slowdown = self.captcha.slowdown
            cd = self.captcha.cooldown_seconds()
            console.print(f"[yellow]Challenge solved. Cooling down {cd / 60:.0f} min and slowing pace x{self.pacer.slowdown:.1f}.[/yellow]")
            time.sleep(cd)
            tp.goto_recs()
            tp.wait_ready()
            return "ok"
        return "stop"

    def run_auto(self, max_swipes: int | None = None, once: bool = True) -> RunStats:
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
                console.print(f"Session: {plan.swipes} swipes planned")
                done = self._swipe_session(tp, plan.swipes)
                self.pacer.end_session(done)
                if once or done < plan.swipes:
                    break
                b = self.pacer.break_seconds()
                console.print(f"Break for {b / 60:.0f} min")
                time.sleep(b)
        finally:
            session.close()
        return self.stats

    def _swipe_session(self, tp: TinderPage, n: int) -> int:
        done = 0
        last_key = ""
        while done < n:
            if self._check_challenge(tp) == "stop":
                return done
            tp.dismiss_popups()
            if tp.out_of_likes():
                console.print("[yellow]Tinder says you are out of likes. Ending session.[/yellow]")
                return done
            card = tp.current_card()
            if not card.name:
                card = tp.wait_for_new_card(last_key, timeout_s=15)
                if card is None:
                    console.print("[red]No card visible; stopping the session.[/red]")
                    return done
            last_key = card.key
            profile = self.resolve_profile(tp, card)
            plan = self.pacer.plan_profile(len(profile.photo_urls) or 1, len(profile.bio or ""))
            verdict, feats = self.score_profile(tp, profile)
            # look at the card like a person would (the scoring above already took a moment)
            if plan["browse_photos"]:
                tp.browse_photos(plan["browse_photos"])
            if plan["open_profile"]:
                tp.peek_profile()
            if self.rng.random() < 0.5:
                tp.mouse.wiggle()
            time.sleep(plan["read_seconds"])
            ok = tp.like() if verdict.like else tp.nope()
            if not ok:
                self.stats.skipped += 1
                tp.dismiss_popups()
                continue
            self.storage.add_decision(profile.id, verdict.action, verdict.score, "auto", verdict.reasons, feats)
            self.scorer.maybe_retrain()
            done += 1
            if verdict.like:
                self.stats.liked += 1
            else:
                self.stats.noped += 1
            tag = "[bold green]LIKE[/bold green]" if verdict.like else "[bold red]NOPE[/bold red]"
            console.print(f"{tag} {profile.name} {profile.age or ''}  p={verdict.score:.2f}  {', '.join(verdict.reasons)}")
            time.sleep(self.pacer.post_action_delay())
            tp.dismiss_popups()
            if plan["micro_break"]:
                time.sleep(plan["micro_break"])
            tp.wait_for_new_card(last_key, timeout_s=10)
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
