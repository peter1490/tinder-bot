"""End-to-end likeness pipeline on the synthetic models (no downloads)."""

import pickle

import numpy as np

from tests.conftest import encode_jpeg, make_image
from tinderbot.config import Config
from tinderbot.likeness.features import FEATURE_KEYS, FeatureExtractor, keyword_hits
from tinderbot.likeness.references import enroll_folder
from tinderbot.likeness.scorer import Scorer
from tinderbot.storage import ProfileRecord


def test_face_engine_on_fake_models(models):
    img = make_image((120, 80, 200), noise=10)
    faces = models.face.analyse(img)
    assert len(faces) == 1
    f = faces[0]
    assert f.embedding.shape == (512,) and abs(np.linalg.norm(f.embedding) - 1) < 1e-4
    assert 100 < f.bbox[0] < 140 and f.score > 0.9
    v = models.clip.embed_image(img)
    assert v.shape == (512,) and abs(np.linalg.norm(v) - 1) < 1e-4


def test_enroll_and_features_and_verdict(cfg: Config, storage, models, tmp_path):
    liked = tmp_path / "liked"
    disliked = tmp_path / "disliked"
    liked.mkdir()
    disliked.mkdir()
    import cv2

    for i in range(3):
        cv2.imwrite(str(liked / f"{i}.jpg"), make_image((200, 40, 40), noise=8, seed=i))      # "red" people
        cv2.imwrite(str(disliked / f"{i}.jpg"), make_image((40, 40, 200), noise=8, seed=10 + i))  # "blue" people
    s1 = enroll_folder(cfg, models, storage, liked, "liked")
    s2 = enroll_folder(cfg, models, storage, disliked, "disliked")
    assert s1 == {"images": 3, "faces": 3, "skipped": 0} and s2["faces"] == 3

    ex = FeatureExtractor(cfg, models, storage)
    assert ex.reference_summary() == {"liked_faces": 3, "disliked_faces": 3, "liked_clip": 3, "disliked_clip": 3}
    assert ex._pool_ids["liked_faces"] == [None] * 3  # enrolled images belong to no profile

    red = ProfileRecord(id="red", name="R", age=30, bio="I love hiking", photo_urls=["u1", "u2"])
    blue = ProfileRecord(id="blue", name="B", age=30, photo_urls=["u3"])
    a_red = ex.analyse_profile(red, [encode_jpeg(make_image((210, 50, 40), noise=8, seed=99)),
                                     encode_jpeg(make_image((190, 45, 50), noise=8, seed=98))])
    a_blue = ex.analyse_profile(blue, [encode_jpeg(make_image((50, 40, 210), noise=8, seed=97))])
    f_red, f_blue = ex.features(a_red), ex.features(a_blue)
    assert set(f_red) == set(FEATURE_KEYS)
    assert f_red["face_sim_liked_max"] > f_red["face_sim_disliked_max"]
    assert f_blue["face_sim_disliked_max"] > f_blue["face_sim_liked_max"]
    assert f_red["face_knn_liked_frac"] > 0.5 > f_blue["face_knn_liked_frac"]
    assert f_red["photo_count"] == 2 and f_red["face_photo_ratio"] == 1.0
    assert f_red["identity_consistency"] > 0.9

    cfg.likeness.liked_bio_keywords = ["hiking"]
    ex2 = FeatureExtractor(cfg, models, storage)
    assert ex2.features(a_red)["bio_keyword_hits"] == 1

    sc = Scorer(cfg, storage)
    v_red, v_blue = sc.decide(red, f_red), sc.decide(blue, f_blue)
    assert v_red.score > v_blue.score
    assert v_red.like and not v_blue.like
    assert any("face" in r for r in v_red.reasons)

    # persisted: photos on disk + embeddings in DB
    assert (cfg.photos_path / "red" / "00.jpg").exists()
    assert len(storage.embeddings_for_profile("red", "face")) == 2
    assert len(storage.embeddings_for_profile("red", "clip")) == 2


def test_hard_filters(cfg: Config, storage):
    sc = Scorer(cfg, storage)
    feats = {k: 0.0 for k in FEATURE_KEYS}
    feats["photo_count"] = 2
    cfg.likeness.min_age, cfg.likeness.max_age = 25, 35
    v = sc.decide(ProfileRecord(id="x", age=40), feats)
    assert v.filtered and not v.like and "age" in v.reasons[0]
    cfg.likeness.max_distance_km = 30
    v = sc.decide(ProfileRecord(id="x", age=30, distance_km=100), feats)
    assert v.filtered and "distance" in v.reasons[0]
    cfg.likeness.blocked_bio_keywords = ["cashapp"]
    v = sc.decide(ProfileRecord(id="x", age=30, distance_km=5, bio="send me on CashApp"), feats)
    assert v.filtered
    feats["no_face_ratio"] = 1.0
    v = sc.decide(ProfileRecord(id="x", age=30, distance_km=5, bio="hi"), feats)
    assert v.filtered and "face" in v.reasons[0]
    cfg.likeness.require_verified = True
    feats["no_face_ratio"] = 0.0
    v = sc.decide(ProfileRecord(id="x", age=30, distance_km=5, verified=False), feats)
    assert v.filtered


def test_learned_model_blend(cfg: Config, storage):
    cfg.likeness.learning.min_examples = 20
    cfg.likeness.learning.blend_full_at = 40
    sc = Scorer(cfg, storage)
    assert sc.learned_weight() == 0.0
    rng = np.random.default_rng(0)
    for i in range(60):
        pid = f"p{i}"
        storage.upsert_profile(ProfileRecord(id=pid))
        like = i % 2 == 0
        feats = {k: float(rng.normal(0, 0.05)) for k in FEATURE_KEYS}
        feats["face_sim_liked_max"] = 0.6 if like else 0.1
        feats["photo_count"] = 3
        storage.add_decision(pid, "like" if like else "nope", 0.5, "manual", [], feats, label=int(like))
    n = sc.retrain()
    assert n == 60 and sc.model is not None and sc.learned_weight() == 1.0
    assert sc.model_path.exists()
    good = {k: 0.0 for k in FEATURE_KEYS}
    good.update(face_sim_liked_max=0.6, photo_count=3)
    bad = dict(good, face_sim_liked_max=0.1)
    vg, vb = sc.decide(ProfileRecord(id="q", age=30), good), sc.decide(ProfileRecord(id="q", age=30), bad)
    assert vg.learned is not None and vg.learned > 0.8 and vb.learned < 0.2
    assert vg.like and not vb.like
    # reload from disk
    sc2 = Scorer(cfg, storage)
    assert sc2.model is not None and sc2.n_train == 60


def test_uncertain_band(cfg: Config, storage):
    sc = Scorer(cfg, storage)
    feats = {k: 0.0 for k in FEATURE_KEYS}
    feats["photo_count"] = 1
    # neutral features -> prior sigmoid(bias + quality terms) ~ close to threshold? force it
    cfg.likeness.like_threshold = 0.5
    cfg.likeness.uncertain_band = 0.5
    v = sc.decide(ProfileRecord(id="x", age=30), feats)
    assert v.uncertain and "uncertain" in v.reasons


def test_keyword_hits():
    assert keyword_hits("I like Hiking and climbing!", ["hiking", "climb", "climbing"]) == 2
    assert keyword_hits("", ["x"]) == 0


def _profile_with_photos(ex: FeatureExtractor, pid: str, colour: tuple[int, int, int], seed: int, n: int = 2):
    prof = ProfileRecord(id=pid, name=pid.title(), age=27, photo_urls=[f"u_{pid}_{i}" for i in range(n)])
    imgs = [encode_jpeg(make_image(colour, noise=8, seed=seed + i)) for i in range(n)]
    return prof, ex.analyse_profile(prof, imgs)


def test_retrain_recomputes_features_leave_one_out(cfg: Config, storage, models):
    """Training features are rebuilt from the stored embeddings against the current pools, and a
    profile never sees its own vectors in the pool it is scored against."""
    ex = FeatureExtractor(cfg, models, storage)
    assert ex.reference_summary() == {"liked_faces": 0, "disliked_faces": 0, "liked_clip": 0, "disliked_clip": 0}
    stale = {k: 0.0 for k in FEATURE_KEYS}
    for i in range(4):
        prof, _ = _profile_with_photos(ex, f"red{i}", (200, 40, 40), seed=10 * i)
        storage.add_decision(prof.id, "like", 0.5, "manual", ["shadow"], stale, label=1)   # features stored: all zero
    for i in range(4):
        prof, _ = _profile_with_photos(ex, f"blue{i}", (40, 40, 200), seed=100 + 10 * i)
        storage.add_decision(prof.id, "nope", 0.5, "manual", ["shadow"], stale, label=0)
    ex.refresh_references()
    assert ex.reference_summary() == {"liked_faces": 4, "disliked_faces": 4, "liked_clip": 4, "disliked_clip": 4}
    assert ex._pool("liked_faces", "red0").shape == (3, 512) and ex._pool("liked_faces", "zzz").shape == (4, 512)
    assert ex._pool("disliked_faces", "red0").shape == (4, 512)

    a = ex.analysis_from_storage("red0")
    assert a is not None and len(a.photos) == 2 and all(len(ph.faces) == 1 and ph.clip is not None for ph in a.photos)
    with_self = ex.features(a)
    loo = ex.features(a, exclude_profile_id="red0")
    assert with_self["primary_face_sim_liked"] > 0.999          # its own vector is in the pool
    assert loo["primary_face_sim_liked"] < with_self["primary_face_sim_liked"]
    assert loo["face_knn_liked_frac"] > 0.5 > ex.features(ex.analysis_from_storage("blue1"), "blue1")["face_knn_liked_frac"]
    assert ex.analysis_from_storage("nobody") is None

    cfg.likeness.learning.min_examples = 4
    cfg.likeness.learning.blend_full_at = 8
    sc = Scorer(cfg, storage, extractor=ex)
    X, y, recomputed = sc.training_matrix()
    assert X.shape == (8, len(FEATURE_KEYS)) and recomputed == 8 and sorted(y.tolist()) == [0] * 4 + [1] * 4
    assert np.abs(X).sum() > 0                                   # the zero features on disk were not used
    assert sc.retrain() == 8 and sc.model is not None
    red = sc.decide(ProfileRecord(id="new", age=27), loo)
    blue = sc.decide(ProfileRecord(id="new", age=27), ex.features(ex.analysis_from_storage("blue1"), "blue1"))
    assert red.learned > 0.5 > blue.learned


def test_stale_model_file_is_discarded(cfg: Config, storage):
    path = cfg.models_path / "likeness_lr.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"model": object(), "n": 99, "keys": ["some", "old", "layout"]}, fh)
    sc = Scorer(cfg, storage)
    assert sc.model is None and sc.stale and sc.learned_weight() == 0.0
    assert sc.retrain() == 0 and not sc.stale                     # nothing to train on yet: prior only


def test_distance_is_log_compressed_for_the_learned_model():
    f = {k: 0.0 for k in FEATURE_KEYS}
    f["distance_km"] = 10000.0
    v = Scorer._vec(f)
    assert 9 < v[FEATURE_KEYS.index("distance_km")] < 10


def test_crush_flags(cfg: Config, storage):
    cfg.likeness.learning.min_examples = 20
    cfg.likeness.learning.blend_full_at = 40
    sc = Scorer(cfg, storage)
    rng = np.random.default_rng(1)
    for i in range(60):
        storage.upsert_profile(ProfileRecord(id=f"p{i}"))
        like = i % 2 == 0
        feats = {k: float(rng.normal(0, 0.05)) for k in FEATURE_KEYS}
        feats.update(face_margin=0.3 if like else -0.3, primary_face_knn_liked_frac=0.9 if like else 0.1, photo_count=3)
        storage.add_decision(f"p{i}", "like" if like else "nope", 0.5, "manual", [], feats, label=int(like))
    assert sc.retrain() == 60
    strong = {k: 0.0 for k in FEATURE_KEYS}
    strong.update(face_margin=0.5, primary_face_knn_liked_frac=1.0, photo_count=3)
    v = sc.decide(ProfileRecord(id="q", age=30), strong)
    assert v.like and v.score > 0.95 and v.crush and v.super_crush and "super crush" in v.reasons
    cfg.crush.message_threshold = 1.01
    v = sc.decide(ProfileRecord(id="q", age=30), strong)
    assert v.crush and not v.super_crush and "crush" in v.reasons
    cfg.crush.enabled = False
    assert not sc.decide(ProfileRecord(id="q", age=30), strong).crush
    # without a learned model the prior alone never earns a Super Like (require_learned)
    cfg.crush.enabled = True
    sc.model = None
    v = sc.decide(ProfileRecord(id="q", age=30), strong)
    assert v.learned is None and not v.crush
    cfg.crush.require_learned = False
    assert sc.decide(ProfileRecord(id="q", age=30), strong).crush == (sc.decide(ProfileRecord(id="q", age=30), strong).score >= 0.9)
