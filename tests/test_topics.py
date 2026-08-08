import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("QDRANT_URL", "http://test:6333")

import pytest
from fastapi.testclient import TestClient
from app.main import app, TOPIC_TREE, VERSES_BY_NUMBER


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_verses_by_number_loaded_fully():
    assert len(VERSES_BY_NUMBER) == 945


def test_topic_tree_covers_every_verse_exactly_once():
    seen = []

    def walk(node):
        if node.get("children"):
            for child in node["children"]:
                walk(child)
        else:
            seen.extend(node["verse_numbers"])

    walk(TOPIC_TREE)
    assert len(seen) == 945
    assert len(set(seen)) == 945
    assert set(seen) == set(range(1, 946))


def test_topics_endpoint_returns_full_tree(client):
    resp = client.get("/topics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "root"
    assert len(data["children"]) > 0


def test_verses_batch_returns_requested_verses(client):
    resp = client.post("/verses/batch", json={"verse_numbers": [1, 42, 945]})
    assert resp.status_code == 200
    data = resp.json()
    numbers = {v["verse_number"] for v in data}
    assert numbers == {1, 42, 945}


def test_verses_batch_skips_unknown_numbers(client):
    resp = client.post("/verses/batch", json={"verse_numbers": [1, 99999]})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["verse_number"] == 1


def test_verses_batch_rejects_empty_list(client):
    resp = client.post("/verses/batch", json={"verse_numbers": []})
    assert resp.status_code == 422


def test_verses_batch_rejects_too_many(client):
    resp = client.post("/verses/batch", json={"verse_numbers": list(range(1, 52))})
    assert resp.status_code == 422
