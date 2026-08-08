"""
One-time offline pipeline: cluster the 945 Kagga verses into a
topic -> subtopic -> verse-numbers hierarchy for the "Explore" 3D graph.

Not part of the deployed API — run locally with topic_tree_requirements.txt
installed in a throwaway venv:

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r scripts/topic_tree_requirements.txt
    OPENAI_API_KEY=sk-... python3 scripts/generate_topic_tree.py

Writes backend/data/topic_tree.json.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from sklearn.cluster import KMeans
from tqdm import tqdm

DATA_PATH = Path(__file__).parent.parent / "data" / "kaggas.json"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "topic_tree.json"
CACHE_PATH = Path(__file__).parent / ".cache" / "verse_embeddings.json"

TOP_K = 10
LEAF_MAX = 18
MAX_DEPTH = 3
REPRESENTATIVES_PER_CLUSTER = 5

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    print("ERROR: OPENAI_API_KEY not set")
    sys.exit(1)

openai = AsyncOpenAI(api_key=OPENAI_KEY)


def load_verses() -> list[dict]:
    with open(DATA_PATH) as f:
        verses = json.load(f)
    print(f"Loaded {len(verses)} verses")
    return verses


async def embed_batch(texts: list[str]) -> list[list[float]]:
    response = await openai.embeddings.create(model="text-embedding-3-large", input=texts)
    return [d.embedding for d in response.data]


async def get_embeddings(verses: list[dict]) -> np.ndarray:
    if CACHE_PATH.exists():
        print(f"Using cached embeddings at {CACHE_PATH}")
        with open(CACHE_PATH) as f:
            cached = json.load(f)
        return np.array([cached[str(v["id"])] for v in verses], dtype=np.float32)

    print("Embedding all verses via OpenAI (text-embedding-3-large)...")
    texts = [f"{v['english_translation']}\n{v['meaning']}" for v in verses]
    vectors: list[list[float]] = []
    batch_size = 100
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        vectors.extend(await embed_batch(texts[i:i + batch_size]))

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump({str(v["id"]): vec for v, vec in zip(verses, vectors)}, f)
    print(f"Cached embeddings to {CACHE_PATH}")

    return np.array(vectors, dtype=np.float32)


def cluster_node(embeddings: np.ndarray, indices: list[int], node_id: str, depth: int) -> dict:
    if len(indices) <= LEAF_MAX or depth >= MAX_DEPTH:
        return {"id": node_id, "verse_indices": indices, "children": None}

    k = TOP_K if depth == 0 else max(2, round(len(indices) / 12))
    k = min(k, len(indices))
    if k < 2:
        return {"id": node_id, "verse_indices": indices, "children": None}

    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(embeddings[indices])

    children = []
    for c in range(k):
        child_indices = [indices[i] for i in range(len(indices)) if km.labels_[i] == c]
        if not child_indices:
            continue
        child_id = f"t{c}" if node_id == "root" else f"{node_id}-{c}"
        children.append(cluster_node(embeddings, child_indices, child_id, depth + 1))

    return {"id": node_id, "verse_indices": indices, "children": children}


def representative_texts(embeddings: np.ndarray, verses: list[dict], indices: list[int]) -> list[str]:
    node_vecs = embeddings[indices]
    centroid = node_vecs.mean(axis=0)
    norm_vecs = node_vecs / np.linalg.norm(node_vecs, axis=1, keepdims=True)
    norm_centroid = centroid / np.linalg.norm(centroid)
    similarities = norm_vecs @ norm_centroid
    top_local = np.argsort(-similarities)[:REPRESENTATIVES_PER_CLUSTER]
    return [verses[indices[i]]["english_translation"][:200] for i in top_local]


LABEL_PROMPT = """These are groups of verses from Mankutimmana Kagga, a Kannada philosophical
text. For each group, give a short label (max 3 words, Title Case, no punctuation) and a
one-sentence description capturing what's distinctive about that group specifically,
relative to the other groups shown. Labels must be mutually distinct from each other.
{parent_constraint}

Respond as JSON: {{"labels": [{{"index": 0, "label": "...", "description": "..."}}, ...]}}

{groups}"""


async def label_children(embeddings: np.ndarray, verses: list[dict], children: list[dict], parent_label: str | None) -> None:
    groups_text = "\n\n".join(
        f"Group {i} ({len(child['verse_indices'])} verses):\n"
        + "\n".join(f"- {t}" for t in representative_texts(embeddings, verses, child["verse_indices"]))
        for i, child in enumerate(children)
    )
    parent_constraint = (
        f'These groups are all subdivisions of a broader category already labeled '
        f'"{parent_label}". Do not reuse "{parent_label}" as a label for any group below — '
        f"each child needs a label distinct from its own parent's."
        if parent_label else ""
    )
    resp = await openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": LABEL_PROMPT.format(groups=groups_text, parent_constraint=parent_constraint)}],
        temperature=0.3,
        max_tokens=1000,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    parsed = json.loads(raw)
    by_index = {item["index"]: item for item in parsed.get("labels", [])}
    for i, child in enumerate(children):
        entry = by_index.get(i, {})
        child["label"] = entry.get("label", f"Topic {child['id']}")
        child["description"] = entry.get("description", "")


async def label_tree(embeddings: np.ndarray, verses: list[dict], root: dict) -> None:
    # Label level-by-level (breadth-first), fully awaiting each level before
    # descending, so a node's own label is always assigned before it's passed
    # as the "don't reuse this" constraint for labeling its children.
    level = [root]
    total_calls = 0
    while level:
        parents = [n for n in level if n["children"]]
        if not parents:
            break
        await asyncio.gather(*[
            label_children(embeddings, verses, n["children"], n.get("label"))
            for n in parents
        ])
        total_calls += len(parents)
        level = [c for n in parents for c in n["children"]]
    print(f"Labeled tree via {total_calls} calls (level-by-level, siblings batched together)")


def serialize(node: dict, verses: list[dict]) -> dict:
    out = {
        "id": node["id"],
        "label": node.get("label", "All Topics"),
        "description": node.get("description", ""),
        "verse_count": len(node["verse_indices"]),
    }
    if node["children"]:
        out["children"] = [serialize(c, verses) for c in node["children"]]
    else:
        out["verse_numbers"] = sorted(verses[i]["id"] for i in node["verse_indices"])
    return out


def validate(tree: dict) -> None:
    leaf_sizes = []
    all_numbers = []

    def walk(node: dict) -> None:
        if "children" in node:
            for c in node["children"]:
                walk(c)
        else:
            leaf_sizes.append(len(node["verse_numbers"]))
            all_numbers.extend(node["verse_numbers"])

    walk(tree)

    assert len(all_numbers) == 945, f"expected 945 verse numbers total, got {len(all_numbers)}"
    assert len(set(all_numbers)) == 945, "duplicate verse numbers across leaves"
    assert set(all_numbers) == set(range(1, 946)), "verse numbers don't cover 1-945 exactly"

    print(f"\nValidation passed: {len(leaf_sizes)} leaves, all 945 verses covered exactly once.")
    print(f"Leaf sizes: min={min(leaf_sizes)}, max={max(leaf_sizes)}, avg={sum(leaf_sizes)/len(leaf_sizes):.1f}")
    oversized = [s for s in leaf_sizes if s > LEAF_MAX]
    if oversized:
        print(f"  Note: {len(oversized)} leaves exceeded LEAF_MAX={LEAF_MAX} (hit MAX_DEPTH): {oversized}")


def print_sample_leaves(tree: dict, verses: list[dict], n: int = 10) -> None:
    by_id = {v["id"]: v for v in verses}
    leaves = []

    def walk(node: dict, path: list[str]) -> None:
        if "children" in node:
            for c in node["children"]:
                walk(c, path + [node["label"]])
        else:
            leaves.append((path + [node["label"]], node))

    walk(tree, [])
    print(f"\n=== Sample of {min(n, len(leaves))} leaves (of {len(leaves)} total) ===")
    for path, leaf in leaves[:n]:
        print(f"\n{' > '.join(path)}  ({leaf['verse_count']} verses)")
        print(f"  {leaf['description']}")
        for num in leaf["verse_numbers"][:3]:
            print(f"  [{num}] {by_id[num]['english_translation'][:100]}")


async def main() -> None:
    verses = load_verses()
    embeddings = await get_embeddings(verses)

    print(f"\nClustering {len(verses)} verses (TOP_K={TOP_K}, LEAF_MAX={LEAF_MAX}, MAX_DEPTH={MAX_DEPTH})...")
    raw_tree = cluster_node(embeddings, list(range(len(verses))), "root", depth=0)

    await label_tree(embeddings, verses, raw_tree)

    tree = serialize(raw_tree, verses)
    validate(tree)
    print_sample_leaves(tree, verses)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)
    print(f"\nWrote topic tree to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
