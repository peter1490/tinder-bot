import datetime as dt
import json

from tinderbot.browser.recs import RECS_URL_PATTERN, RecsQueue, dom_profile_id, parse_recs_payload

BIRTH = (dt.date.today() - dt.timedelta(days=365 * 25 + 40)).isoformat() + "T00:00:00.000Z"

PAYLOAD = {
    "meta": {"status": 200},
    "data": {
        "results": [
            {
                "type": "user",
                "user": {
                    "_id": "5f1a2b3c4d",
                    "name": "Ana",
                    "bio": "hiking & coffee",
                    "birth_date": BIRTH,
                    "badges": [{"type": "selfie_verified"}],
                    "jobs": [{"title": {"name": "Designer"}, "company": {"name": "ACME"}}],
                    "schools": [{"name": "UCL"}],
                    "photos": [
                        {
                            "id": "p1",
                            "url": "https://images-ssl.gotinder.com/u/p1/original.jpg",
                            "processedFiles": [
                                {"url": "https://images-ssl.gotinder.com/u/p1/640x800.jpg", "width": 640, "height": 800},
                                {"url": "https://images-ssl.gotinder.com/u/p1/172x216.jpg", "width": 172, "height": 216},
                            ],
                        },
                        {"id": "p2", "url": "https://images-ssl.gotinder.com/u/p2/original.jpg"},
                    ],
                },
                "s_number": 123456,
                "distance_mi": 5,
                "experiment_info": {"user_interests": {"selected_interests": [{"name": "Hiking"}, {"name": "Coffee"}]}},
            },
            {"type": "ad", "id": "x"},
            {"type": "user", "user": {"_id": "u2", "name": "Bea", "birth_date": "1990-05-05", "photos": []}},
        ]
    },
}


def test_parse_recs_payload():
    recs = parse_recs_payload(json.dumps(PAYLOAD))
    assert [r.id for r in recs] == ["5f1a2b3c4d", "u2"]
    a = recs[0]
    assert a.name == "Ana" and a.age == 25 and a.verified and a.s_number == 123456
    assert a.jobs == ["Designer"] and a.schools == ["UCL"] and a.interests == ["Hiking", "Coffee"]
    assert abs(a.distance_km - 8.05) < 0.01
    # prefers the ~640px processed file, falls back to the original url
    assert a.photo_urls == ["https://images-ssl.gotinder.com/u/p1/640x800.jpg",
                            "https://images-ssl.gotinder.com/u/p2/original.jpg"]
    assert recs[1].age == dt.date.today().year - 1990 - ((dt.date.today().month, dt.date.today().day) < (5, 5))


def test_parse_garbage():
    assert parse_recs_payload("not json") == []
    assert parse_recs_payload({"data": {}}) == []


def test_queue_matching():
    q = RecsQueue(max_size=2)
    assert q.add_payload(PAYLOAD) == 2
    assert len(q) == 2
    # by photo url (query string ignored)
    r = q.match("Whoever", None, ["https://images-ssl.gotinder.com/u/p1/640x800.jpg?sig=abc"])
    assert r and r.id == "5f1a2b3c4d"
    # by name + age
    assert q.match("ana", 25).id == "5f1a2b3c4d"
    assert q.match("ana", 40) is None
    assert q.match("Bea", None).id == "u2"
    q.pop("u2")
    assert q.match("Bea", None) is None


def test_url_pattern_and_dom_id():
    assert RECS_URL_PATTERN.search("https://api.gotinder.com/v2/recs/core?locale=en")
    assert RECS_URL_PATTERN.search("https://api.gotinder.com/v3/recs/core")
    assert not RECS_URL_PATTERN.search("https://api.gotinder.com/v2/profile")
    assert dom_profile_id("Ana", 25, "u") == dom_profile_id("Ana", 25, "u")
    assert dom_profile_id("Ana", 25, "u") != dom_profile_id("Ana", 26, "u")
