"""Order reconciliation: does the chart contain everything the note commits to?

Two Jev requests per check.

Stage 1 asks two atomic questions per orderable item over the note alone -- is it
mentioned, and is it committed to. Keeping those separate is what lets the UI tell
"never came up" apart from "considered and declined", which is the difference
between a useful nudge and a dangerous one.

Stage 2 only runs for items that came back committed but absent from the chart, and
asks which sentence carries the commitment. Its options are the note's own
sentences, so the highlight is selected from the text rather than generated.

The diff, the thresholds and every piece of copy live here in code. The model
supplies judgments about prose; it never decides what happens.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, SecretStr
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, TypeSafeError

load_dotenv()

HERE = Path(__file__).parent
app = FastAPI(title="Order reconciliation")

# --- thresholds -------------------------------------------------------------
# Starting points, not findings. Missing an order is worse than a spurious
# nudge, so COMMITTED_FIRM sits low and there is an explicit uncertain band
# rather than a single cutoff.
COMMITTED_FIRM = 0.70
COMMITTED_MAYBE = 0.45
MENTIONED = 0.60

# Sent to the page so a reader can see where each probability fell, rather than
# being told a verdict and having to trust it.
THRESHOLDS = {
    "committed_firm": COMMITTED_FIRM,
    "committed_maybe": COMMITTED_MAYBE,
    "mentioned": MENTIONED,
}

# --- the orderable catalogue ------------------------------------------------
# Code owns this list. The model never invents an order; it only judges the
# note against entries that already exist here.
CATALOG = [
    ("MB-BC2", "Blood cultures, two sets", "peripheral blood cultures collected from two separate sites"),
    ("RX-CTX", "Ceftriaxone 1 g IV", "intravenous ceftriaxone"),
    ("RX-AZI", "Azithromycin 500 mg IV", "azithromycin, by any route"),
    ("RX-VAN", "Vancomycin IV", "intravenous vancomycin"),
    ("IM-CXR", "Chest X-ray, portable", "a plain chest radiograph"),
    ("IM-CTC", "CT chest with contrast", "a CT scan of the chest"),
    ("CH-LAC", "Lactate, serum", "a serum lactate level"),
    ("HM-CBC", "Full blood count", "a full blood count, also called a CBC"),
    ("CH-BMP", "Renal panel", "a renal or basic metabolic panel, including urea and creatinine"),
    ("CH-CRP", "C-reactive protein", "a C-reactive protein level"),
    ("CH-TRP", "Troponin I", "a cardiac troponin level"),
    ("IV-CRY", "Crystalloid bolus, 30 mL/kg", "an intravenous fluid bolus of crystalloid, saline or Hartmann's"),
    ("RT-O2", "Oxygen, titrated to SpO2 > 92%", "supplemental oxygen"),
    ("MB-RVP", "Respiratory viral panel", "a respiratory viral panel or influenza swab"),
    ("CD-ECG", "ECG, 12-lead", "a twelve-lead electrocardiogram"),
    ("MB-URI", "Urinalysis with culture", "a urine dipstick, urinalysis or urine culture"),
]
BY_CODE = {code: (label, desc) for code, label, desc in CATALOG}

# --- the seeded case --------------------------------------------------------
SEED_NOTE = """\
62M, three days of productive cough with rusty sputum, fever to 38.9, now increasingly short of breath at rest. Ex-smoker, no regular medications.

On examination he looks unwell and is using accessory muscles. RR 26, SpO2 90% on room air, HR 112, BP 104/62. Coarse crackles and dullness at the right base.

Impression: community-acquired pneumonia, right lower lobe, with sepsis physiology.

Plan: blood cultures from two sites before any antibiotics, then ceftriaxone and azithromycin. Portable chest film at the bedside. Send a lactate and repeat it in two hours, along with a full blood count and renal panel. Start 30 mL/kg crystalloid now and reassess his perfusion. Titrate oxygen to keep saturations above 92 percent. Send a respiratory viral panel including influenza. CT chest is not indicated at this stage. Holding vancomycin unless he deteriorates or cultures suggest MRSA. Admit under medicine."""

SEED_PLACED = ["IM-CXR", "HM-CBC", "CH-TRP"]

PATIENT = {
    "name": "OKONKWO, Daniel",
    "mrn": "4417092",
    "age_sex": "62 y  M",
    "dob": "14 Mar 1964",
    "location": "ED Bay 4",
    "arrived": "03:12",
    "weight": "84 kg",
    "allergies": "Penicillin — rash (childhood)",
}

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_note(note: str) -> list[list[dict]]:
    """Note -> paragraphs of identified sentences. One source of truth, so the
    ids the model chooses from are the ids the browser renders."""
    paragraphs, n = [], 0
    for block in note.split("\n"):
        if not block.strip():
            continue
        sentences = []
        for piece in SENTENCE_SPLIT.split(block.strip()):
            if piece.strip():
                n += 1
                sentences.append({"id": f"s{n}", "text": piece.strip()})
        paragraphs.append(sentences)
    return paragraphs


def committed_question(desc: str) -> Noul:
    """Is this order actually being asked for? Deliberately narrow: everything
    that looks like a mention but is not a commitment goes in the false case."""
    return Noul(
        instructions=f"Does the plan in `note` commit to ordering {desc}?",
        criteria={
            "true": "The note states an intention, plan or instruction to order, send, start or give it.",
            "false": {
                "what": "The note does not commit to it.",
                "includes": [
                    "it is not mentioned at all",
                    "it is explicitly declined, not indicated, or withheld",
                    "it is deferred to a condition that has not happened",
                    "it is described as already done elsewhere",
                ],
            },
        },
    )


def mentioned_question(desc: str) -> Noul:
    """Does the note refer to it at all? Paired with the question above, this is
    what separates "never came up" from "came up and was ruled out"."""
    return Noul(
        instructions=f"Does `note` refer to {desc} anywhere at all, in any way?",
        criteria={
            "true": "It is named or clearly described, including to rule it out, decline it or defer it.",
            "false": "It does not appear in the note in any form.",
        },
    )


def locate_question(desc: str, options: dict[str, str]) -> Choice:
    """Which sentence carries the commitment? The options are the note's own
    sentences, so the answer is selected from the text and never written."""
    return Choice(
        instructions=f"Which sentence commits to ordering {desc}?",
        criteria=options,
    )


class CheckRequest(BaseModel):
    note: str = SEED_NOTE
    placed: list[str] = SEED_PLACED
    # Let whoever is using the page supply their own key, so a deployment does
    # not have to hold one. SecretStr keeps it out of reprs and tracebacks; it is
    # used for the request and never written down.
    api_key: SecretStr | None = None


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/case")
def case():
    return {
        "patient": PATIENT,
        "note": SEED_NOTE,
        "placed": SEED_PLACED,
        "catalog": [{"code": c, "label": lbl} for c, lbl, _ in CATALOG],
        "needs_key": not os.environ.get("TYPESAFE_API_KEY"),
    }


@app.post("/api/check")
async def check(req: CheckRequest):
    paragraphs = split_note(req.note)
    sentences = [s for para in paragraphs for s in para]
    if not sentences:
        return {"error": "The note is empty. Write a plan, then check it."}

    placed = set(req.placed)
    t_start = time.perf_counter()

    # api_key=None falls back to TYPESAFE_API_KEY in the environment.
    key = req.api_key.get_secret_value() if req.api_key else None
    try:
        async with AsyncTypeSafeClient(api_key=key) as client:
            # ---- stage 1: two atomic judgments per catalogue item ----------
            questions = {}
            for i, (code, label, desc) in enumerate(CATALOG):
                questions[f"c{i}_committed"] = committed_question(desc)
                questions[f"c{i}_mentioned"] = mentioned_question(desc)

            t0 = time.perf_counter()
            s1 = await client.system_one(state={"note": req.note}, questions=questions)
            stage1_ms = round((time.perf_counter() - t0) * 1000)

            # ---- sort into buckets, in code -------------------------------
            # "silent" is everything the note neither commits to nor mentions.
            # It is the majority and the UI says nothing about it, but it is
            # returned so a reader can see the whole catalogue was considered.
            buckets: dict[str, list[dict]] = {
                name: []
                for name in ("missing", "uncertain", "matched", "undocumented", "declined", "silent")
            }
            for i, (code, label, _) in enumerate(CATALOG):
                committed = s1.answers[f"c{i}_committed"].noul
                mentioned = s1.answers[f"c{i}_mentioned"].noul
                row = {
                    "code": code,
                    "label": label,
                    "committed": round(committed, 2),
                    "mentioned": round(mentioned, 2),
                }
                on_chart = code in placed
                row["on_chart"] = on_chart

                if committed >= COMMITTED_FIRM:
                    bucket = "matched" if on_chart else "missing"
                elif committed >= COMMITTED_MAYBE:
                    bucket = "matched" if on_chart else "uncertain"
                elif on_chart:
                    bucket = "undocumented"
                elif mentioned >= MENTIONED:
                    bucket = "declined"
                else:
                    bucket = "silent"  # not committed, not mentioned, not ordered

                row["bucket"] = bucket
                buckets[bucket].append(row)

            missing = buckets["missing"]

            # ---- stage 2: which sentence carries each commitment? ---------
            # Options are the note's own sentences plus a no-match escape, so a
            # highlight is always something the physician actually wrote.
            stage2_ms, s2_tokens = 0, 0
            if missing:
                options = {s["id"]: s["text"] for s in sentences}
                options["none"] = "No sentence in the note commits to this order."
                locate = {
                    f"loc{i}": locate_question(BY_CODE[row["code"]][1], options)
                    for i, row in enumerate(missing)
                }
                t0 = time.perf_counter()
                s2 = await client.system_one(state={"note": req.note}, questions=locate)
                stage2_ms = round((time.perf_counter() - t0) * 1000)
                s2_tokens = s2.usage.input_tokens or 0
                for i, row in enumerate(missing):
                    a = s2.answers[f"loc{i}"]
                    row["sentence_id"] = None if a.choice == "none" else a.choice
                    row["locate_confidence"] = round(a.confidence, 2)
                    # The runners-up are the interesting part: they show whether
                    # the pick was decisive or a close call between two sentences.
                    ranked = sorted(a.probabilities.items(), key=lambda kv: -kv[1])
                    row["locate_top"] = [[sid, round(p, 3)] for sid, p in ranked[:3]]

    except TypeSafeError as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    missing_ranked = sorted(missing, key=lambda r: -r["committed"])

    # What was actually sent, built by the same functions that sent it, so this
    # cannot drift into a flattering paraphrase of the real prompt. Sampled on
    # the top row so it matches what a reader is looking at.
    sample = missing_ranked[0] if missing_ranked else {"code": CATALOG[0][0], "label": CATALOG[0][1]}
    sample_desc = BY_CODE[sample["code"]][1]
    asked = {
        "order": {"code": sample["code"], "label": sample["label"], "described_as": sample_desc},
        "state": {"note": "<the note, verbatim>"},
        "committed": committed_question(sample_desc).model_dump(exclude_none=True),
        "mentioned": mentioned_question(sample_desc).model_dump(exclude_none=True),
        "locate": {
            **locate_question(sample_desc, {}).model_dump(exclude_none=True),
            "criteria": f"<every sentence of the note, keyed s1..s{len(sentences)}, plus 'none'>",
        },
    }

    return {
        "paragraphs": paragraphs,
        "missing": missing_ranked,
        "uncertain": sorted(buckets["uncertain"], key=lambda r: -r["committed"]),
        "matched": sorted(buckets["matched"], key=lambda r: -r["committed"]),
        "undocumented": buckets["undocumented"],
        "declined": buckets["declined"],
        "silent": sorted(buckets["silent"], key=lambda r: -r["mentioned"]),
        "thresholds": THRESHOLDS,
        "asked": asked,
        "meta": {
            "model": s1.model,
            "questions": len(questions) + len(missing),
            "stage1_questions": len(questions),
            "stage2_questions": len(missing),
            "stage1_ms": stage1_ms,
            "stage2_ms": stage2_ms,
            "total_ms": round((time.perf_counter() - t_start) * 1000),
            "input_tokens": (s1.usage.input_tokens or 0) + s2_tokens,
        },
    }
