"""Generate a synthetic corpus of customer-support tickets.

Realistic-looking tickets across known archetypes give the clustering
pipeline something interesting to find. Each ticket carries a hidden
ground-truth `archetype` label so we can later evaluate cluster quality.

Usage:
    python -m ticket_miner.generate --n 500 --out data/tickets.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

from ticket_miner.archetypes import ARCHETYPES, total_base_count

# ----------------------------------------------------------------------
# Sampling parameters
# ----------------------------------------------------------------------

TIERS = ["free", "plus", "team", "enterprise"]
TIER_WEIGHTS = [0.30, 0.35, 0.20, 0.15]

TONES = ["frustrated", "polite", "terse", "verbose", "confused"]
LENGTHS = ["short", "medium", "long"]
LENGTH_WEIGHTS = [0.30, 0.50, 0.20]
LENGTH_HINTS = {
    "short": "2-3 sentences",
    "medium": "4-7 sentences",
    "long": "8-15 sentences",
}

WINDOW_DAYS = 30  # spread ticket timestamps across last N days
RECENT_DAYS = 7  # the "growing" window for weight_recent archetypes
RECENT_FRACTION = 0.6  # share of growing-archetype tickets in the recent window

CODE_PROBABILITY_TECHNICAL = 0.65
CODE_PROBABILITY_NONTECH = 0.15

# ----------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------


class TicketContent(BaseModel):
    subject: str
    body: str


@dataclass
class Ticket:
    id: str
    archetype: str  # ground truth — hidden from clustering pipeline
    created_at: str
    customer_tier: str
    tone: str
    length: str
    has_code: bool
    subject: str
    body: str


# ----------------------------------------------------------------------
# Prompt construction
# ----------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are generating realistic synthetic customer-support tickets for "
    "the OpenAI developer platform. Tickets must look like a real developer "
    "or end-user wrote them: concrete API error messages, real model names "
    "(gpt-4o, gpt-4.1, gpt-4o-mini, gpt-4.1-mini, text-embedding-3-small, "
    "etc.), real HTTP status codes, and short Python or Node code snippets "
    "where realistic. Do NOT mention that the ticket is synthetic. Output "
    "ONLY the customer's incoming message — no support-agent reply."
)


def build_user_prompt(
    archetype: str, tone: str, length: str, has_code: bool, tier: str
) -> str:
    cfg = ARCHETYPES[archetype]
    if has_code and cfg["code_pattern"] != "n/a":
        code_line = (
            f"Include a short (3-10 line) Python code snippet illustrating: "
            f"{cfg['code_pattern']}."
        )
    else:
        code_line = "Do NOT include a code snippet."

    return (
        f"Write one customer-support ticket with these characteristics:\n\n"
        f"- Scenario: {cfg['scenario']}\n"
        f"- Customer tier: {tier}\n"
        f"- Tone: {tone}\n"
        f"- Length: {length} ({LENGTH_HINTS[length]})\n"
        f"- {code_line}\n\n"
        f"Return JSON with exactly two keys:\n"
        f'  "subject": short subject line (under 80 chars)\n'
        f'  "body":    the ticket body\n'
    )


# ----------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------


def generate_ticket(
    client: OpenAI,
    model: str,
    ticket_id: str,
    archetype: str,
    created_at: datetime,
    tier: str,
    tone: str,
    length: str,
    has_code: bool,
) -> Ticket:
    user = build_user_prompt(archetype, tone, length, has_code, tier)
    completion = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        response_format=TicketContent,
        temperature=1.0,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("Model returned no parsed content")
    return Ticket(
        id=ticket_id,
        archetype=archetype,
        created_at=created_at.isoformat(),
        customer_tier=tier,
        tone=tone,
        length=length,
        has_code=has_code,
        subject=parsed.subject.strip(),
        body=parsed.body.strip(),
    )


# ----------------------------------------------------------------------
# Planning — decide what to generate before any LLM calls
# ----------------------------------------------------------------------


def plan_tickets(total: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    base = total_base_count()
    scale = total / base
    now = datetime.now(timezone.utc)

    plans: list[dict] = []
    for archetype, cfg in ARCHETYPES.items():
        n = max(1, round(cfg["count"] * scale))
        for _ in range(n):
            if cfg.get("weight_recent"):
                # Bias toward the last RECENT_DAYS — gives severity scorer
                # something to detect.
                if rng.random() < RECENT_FRACTION:
                    days_ago = rng.uniform(0, RECENT_DAYS)
                else:
                    days_ago = rng.uniform(RECENT_DAYS, WINDOW_DAYS)
            else:
                days_ago = rng.uniform(0, WINDOW_DAYS)

            ts = now - timedelta(days=days_ago)
            tier = rng.choices(TIERS, weights=TIER_WEIGHTS)[0]
            tone = rng.choice(TONES)
            length = rng.choices(LENGTHS, weights=LENGTH_WEIGHTS)[0]

            if archetype in ("noise", "billing_quota", "auth_error"):
                code_p = CODE_PROBABILITY_NONTECH
            else:
                code_p = CODE_PROBABILITY_TECHNICAL
            has_code = rng.random() < code_p

            plans.append(
                {
                    "archetype": archetype,
                    "created_at": ts,
                    "tier": tier,
                    "tone": tone,
                    "length": length,
                    "has_code": has_code,
                }
            )

    rng.shuffle(plans)
    plans = plans[:total]
    for i, p in enumerate(plans):
        p["id"] = f"TKT-{i + 1:04d}"
    return plans


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic support tickets.")
    parser.add_argument("--n", type=int, default=500, help="Total tickets to generate")
    parser.add_argument(
        "--out",
        dest="out",
        type=Path,
        default=Path("data/tickets.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY not set. Copy .env.example to .env and add your key.",
            file=sys.stderr,
        )
        return 1

    client = OpenAI()
    plans = plan_tickets(args.n, args.seed)
    print(f"Planned {len(plans)} tickets across {len(ARCHETYPES)} archetypes.")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    tickets: list[Ticket] = []
    failures: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                generate_ticket,
                client,
                args.model,
                p["id"],
                p["archetype"],
                p["created_at"],
                p["tier"],
                p["tone"],
                p["length"],
                p["has_code"],
            ): p
            for p in plans
        }
        with tqdm(total=len(futures), desc="generating", unit="tkt") as pbar:
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    tickets.append(fut.result())
                except Exception as e:  # noqa: BLE001
                    failures.append((p["id"], f"{type(e).__name__}: {e}"))
                pbar.update(1)

    tickets.sort(key=lambda t: t.id)
    with args.out.open("w") as f:
        for t in tickets:
            f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")

    print(f"\nWrote {len(tickets)} tickets → {args.out}")
    if failures:
        print(f"  ({len(failures)} failures)")
        for tid, err in failures[:5]:
            print(f"  {tid}: {err}")

    dist = Counter(t.archetype for t in tickets)
    print("\nArchetype distribution:")
    for arch, cnt in sorted(dist.items(), key=lambda kv: -kv[1]):
        print(f"  {arch:30s} {cnt:4d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
