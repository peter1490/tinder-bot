import numpy as np

from tinderbot.storage import ProfileRecord, Storage


def test_profile_photo_embedding_roundtrip(storage: Storage):
    p = ProfileRecord(id="u1", name="Ana", age=27, bio="hi", photo_urls=["http://x/1.jpg", "http://x/2.jpg"], verified=True)
    storage.upsert_profile(p)
    storage.upsert_photo("u1_a", "u1", p.photo_urls[0], None, 640, 480, face_count=1, quality=0.7, position=0)
    storage.upsert_photo("u1_b", "u1", p.photo_urls[1], None, 640, 480, face_count=0, quality=0.2, position=1)
    v = np.arange(512, dtype=np.float32)
    storage.put_embedding("u1_a", "face", v, "m", idx=0)
    storage.put_embedding("u1_a", "clip", v * 2, "c")
    row = storage.get_profile("u1")
    assert row["name"] == "Ana" and row["verified"] == 1 and row["photo_count"] == 2
    faces = storage.embeddings_for_profile("u1", "face")
    assert len(faces) == 1 and np.allclose(faces[0], v)
    clips = storage.embeddings_for_profile("u1", "clip")
    assert np.allclose(clips[0], v * 2)


def test_decisions_labels_and_training_examples(storage: Storage):
    storage.upsert_profile(ProfileRecord(id="a"))
    storage.upsert_profile(ProfileRecord(id="b"))
    storage.add_decision("a", "like", 0.9, "auto", ["x"], {"f": 1.0})
    storage.add_decision("b", "nope", 0.1, "auto", ["uncertain"], {"f": 0.0})
    assert storage.count_decisions() == 2
    assert storage.count_decisions(action="like") == 1
    assert storage.has_decision("a") and not storage.has_decision("zzz")
    storage.relabel("b", 1)
    ex = {r["profile_id"]: r["label"] for r in storage.training_examples()}
    assert ex == {"a": 1, "b": 1}
    assert [r["profile_id"] for r in storage.uncertain_for_review()] == []  # relabel switched source to manual


def test_reference_vectors_and_profile_vectors(storage: Storage):
    storage.put_reference("p1", "liked", "face", "m", np.ones(4, np.float32))
    storage.put_reference("p2", "liked", "face", "m", np.zeros(4, np.float32))
    storage.put_reference("p3", "disliked", "clip", "c", np.ones(3, np.float32))
    assert storage.references("liked", "face").shape == (2, 4)
    assert storage.references("disliked", "face").size == 0
    assert storage.references("disliked", "clip").shape == (1, 3)
    # labelled profile mean vectors
    storage.upsert_profile(ProfileRecord(id="u"))
    storage.upsert_photo("u_1", "u", "", None)
    storage.upsert_photo("u_2", "u", "", None)
    storage.put_embedding("u_1", "clip", np.array([1, 0], np.float32), "c")
    storage.put_embedding("u_2", "clip", np.array([0, 1], np.float32), "c")
    storage.add_decision("u", "like", 0.8, "auto")
    m = storage.liked_profile_vectors("clip", 1)
    assert m.shape == (1, 2) and np.allclose(m[0], [0.5, 0.5])
    storage.clear_references("liked")
    assert storage.references("liked", "face").size == 0


def test_events_and_meta(storage: Storage):
    storage.log_event("captcha", {"kind": "captcha"})
    assert storage.count_events("captcha", 0) == 1
    storage.set_meta("k", {"a": 1})
    assert storage.get_meta("k") == {"a": 1}
    assert storage.get_meta("missing", 5) == 5
