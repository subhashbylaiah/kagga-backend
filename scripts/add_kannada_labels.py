"""
One-time pass over the EXISTING backend/data/topic_tree.json: translates each
node's already-generated English label/description into natural Kannada
(label_kn/description_kn), without touching clustering or the English text at
all. Run after generate_topic_tree.py, or any time to backfill Kannada onto
an existing tree.

Usage:
    OPENAI_API_KEY=sk-... python3 scripts/add_kannada_labels.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from openai import AsyncOpenAI

TREE_PATH = Path(__file__).parent.parent / "data" / "topic_tree.json"
BATCH_SIZE = 20

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    print("ERROR: OPENAI_API_KEY not set")
    sys.exit(1)

openai = AsyncOpenAI(api_key=OPENAI_KEY)

TRANSLATE_PROMPT = """Translate each of these English topic labels and descriptions (from a
philosophical text explorer) into natural, idiomatic Kannada — the way a Kannada speaker
would actually name and describe the theme, not a literal word-for-word translation.

Respond as JSON: {{"translations": [{{"id": "...", "label_kn": "...", "description_kn": "..."}}, ...]}}

{items}"""


def collect_nodes(node: dict, out: list[dict]) -> None:
    out.append(node)
    for child in node.get("children") or []:
        collect_nodes(child, out)


async def translate_batch(nodes: list[dict]) -> None:
    items_text = "\n\n".join(
        f'id: {n["id"]}\nlabel: {n["label"]}\ndescription: {n["description"]}'
        for n in nodes
    )
    resp = await openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": TRANSLATE_PROMPT.format(items=items_text)}],
        temperature=0.3,
        max_tokens=2000,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    parsed = json.loads(raw)
    by_id = {item["id"]: item for item in parsed.get("translations", [])}
    for n in nodes:
        entry = by_id.get(n["id"], {})
        n["label_kn"] = entry.get("label_kn", "")
        n["description_kn"] = entry.get("description_kn", "")


async def main() -> None:
    with open(TREE_PATH) as f:
        tree = json.load(f)

    all_nodes: list[dict] = []
    collect_nodes(tree, all_nodes)
    print(f"Found {len(all_nodes)} nodes (including root) to translate.")

    batches = [all_nodes[i:i + BATCH_SIZE] for i in range(0, len(all_nodes), BATCH_SIZE)]
    print(f"Translating in {len(batches)} batches of up to {BATCH_SIZE}...")
    await asyncio.gather(*(translate_batch(b) for b in batches))

    missing = [n["id"] for n in all_nodes if not n.get("label_kn")]
    if missing:
        print(f"WARNING: {len(missing)} nodes missing label_kn: {missing}")
    else:
        print("All nodes have label_kn.")

    print("\n=== Sample ===")
    for n in all_nodes[:8]:
        print(f"[{n['id']}] {n['label']} -> {n['label_kn']}")
        print(f"       {n['description']}")
        print(f"       {n['description_kn']}")

    with open(TREE_PATH, "w") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)
    print(f"\nWrote updated tree (with Kannada labels) to {TREE_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
