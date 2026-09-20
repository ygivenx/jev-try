# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: jev-try
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Alfred: clinical triage with Jev (TypeSafe System One)
#
# A worked example that follows the shape the TypeSafe docs recommend in
# [how to build with System One](https://docs.typesafe.ai/concepts/how-to-build-with-system-one.md),
# applied to triage notes instead of the docs' support tickets.
#
# The seven steps from that page, and where each one shows up below:
#
# | Doc step | Where |
# |---|---|
# | 1. Use code when possible | `vitals_flags()` — every number is compared in code |
# | 2. Decompose input state | `build_state()` sends only the fields the questions need |
# | 3. Structure the input state | nested JSON, referenced with backticked paths |
# | 4. Decompose questions | eight narrow questions, one property each |
# | 5. Ask many questions in parallel | a single `system_one()` call |
# | 6. Combine outputs in code | `compose()` — weights and rules live here |
# | 7. Route on uncertainty | `route()` — confidence thresholds, no model involved |
#
# > **This is an API demo, not a medical device.** The notes below are synthetic,
# > the acuity levels are loosely ESI-shaped but are not a validated triage
# > instrument, and nothing here is safe to point at a real patient. The point is
# > the *software pattern*: which judgments you hand to a model, which you keep in
# > code, and how you refuse to act when the model is unsure.

# %% [markdown]
# ## Setup
#
# `TypeSafeClient()` reads `TYPESAFE_API_KEY` from the environment and defaults to
# `jev-latest`. The key lives in `.env` and is loaded with `python-dotenv` so it
# never appears in the notebook.

# %%
from __future__ import annotations

import os
import time

import pandas as pd
from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

load_dotenv()
assert os.environ.get("TYPESAFE_API_KEY"), "Put TYPESAFE_API_KEY in .env"

client = TypeSafeClient()
print("available models:", [m.name for m in client.models.list().models])

# %% [markdown]
# ## Step 1 — code owns the numbers
#
# `jev-1.13` is explicitly [not reliable at arithmetic, counting, or numeric
# comparison](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md), and the
# documented workaround is to *"use semantic representations instead of numeric
# values"*. So vitals never reach the model as numbers. Code applies the
# thresholds and hands over words.
#
# This is also the cheapest possible win: a threshold comparison does not need a
# model, and `hr > 120` will never be wrong in a way you have to evaluate.

# %%
VITALS_RULES = {
    "hypoxia": lambda v: v["spo2"] < 92,
    "hypotension": lambda v: v["sbp"] < 90,
    "tachycardia": lambda v: v["hr"] > 120,
    "bradycardia": lambda v: v["hr"] < 50,
    "tachypnoea": lambda v: v["rr"] > 22,
    "fever": lambda v: v["temp_c"] >= 38.0,
    "hypothermia": lambda v: v["temp_c"] < 36.0,
}


def vitals_flags(vitals: dict) -> list[str]:
    """Numeric vitals -> semantic labels. Pure code, no model."""
    return [name for name, rule in VITALS_RULES.items() if rule(vitals)]


def unstable_vitals(flags: list[str]) -> bool:
    """A deterministic tripwire that does not get a vote from the model."""
    return any(f in flags for f in ("hypoxia", "hypotension", "bradycardia"))


# %% [markdown]
# ## The cases
#
# Six synthetic presentations, chosen to span the range: two that should escalate
# hard, two that should not, and two that are genuinely ambiguous — because a
# triage demo where every answer is obvious tells you nothing about calibration.

# %%
CASES = [
    {
        "id": "C1",
        "patient": {"age": 61, "sex": "M"},
        "note": (
            "Crushing substernal chest pressure that started 25 minutes ago while "
            "walking to the car, radiating into the left jaw and arm. Diaphoretic "
            "and nauseated. Says it feels like an elephant on his chest."
        ),
        "vitals": {"hr": 104, "sbp": 138, "rr": 20, "temp_c": 36.7, "spo2": 96},
        "history": ["hypertension", "type 2 diabetes", "smoker 30 pack-years"],
    },
    {
        "id": "C2",
        "patient": {"age": 74, "sex": "F"},
        "note": (
            "Husband reports she was fine at breakfast. Since about 40 minutes ago "
            "her right arm is limp, the right side of her face is drooping and her "
            "speech is slurred and hard to follow. She is awake and looking around."
        ),
        "vitals": {"hr": 88, "sbp": 172, "rr": 18, "temp_c": 36.9, "spo2": 97},
        "history": ["atrial fibrillation", "on apixaban"],
    },
    {
        "id": "C3",
        "patient": {"age": 23, "sex": "F"},
        "note": (
            "Sore throat and blocked nose for three days. Coughing at night, "
            "otherwise eating and drinking normally. Took paracetamol this morning. "
            "Asking whether she needs antibiotics before a flight on Friday."
        ),
        "vitals": {"hr": 78, "sbp": 118, "rr": 16, "temp_c": 37.3, "spo2": 99},
        "history": [],
    },
    {
        "id": "C4",
        "patient": {"age": 68, "sex": "M"},
        "note": (
            "Two days of burning on urination, now shaking with chills and confused "
            "about the date according to his daughter. Warm and flushed, breathing "
            "fast, very drowsy but rousable."
        ),
        "vitals": {"hr": 126, "sbp": 84, "rr": 26, "temp_c": 38.9, "spo2": 93},
        "history": ["benign prostatic hyperplasia", "indwelling catheter"],
    },
    {
        "id": "C5",
        "patient": {"age": 34, "sex": "F"},
        "note": (
            "Cramping lower abdominal pain since last night, comes and goes. Ate "
            "takeaway the evening before. One episode of loose stool, no vomiting. "
            "Pain is bearable when she sits still. Last period about six weeks ago."
        ),
        "vitals": {"hr": 96, "sbp": 112, "rr": 18, "temp_c": 37.1, "spo2": 98},
        "history": [],
    },
    {
        "id": "C6",
        "patient": {"age": 8, "sex": "M"},
        "note": (
            "Ate a biscuit at a party, now lips and eyelids are swollen, widespread "
            "hives, and he is coughing with a hoarse voice and says his throat feels "
            "tight. Mother has an unused adrenaline autoinjector in her bag."
        ),
        "vitals": {"hr": 148, "sbp": 82, "rr": 32, "temp_c": 36.8, "spo2": 90},
        "history": ["known peanut allergy", "asthma"],
    },
]

# %% [markdown]
# ## Steps 2 & 3 — decompose and structure the state
#
# Two rules from the docs are doing the work here. *"Include only the context
# relevant to the current questions"* — accuracy drops as irrelevant state grows,
# so the raw vitals dict is dropped once code has turned it into flags. And
# *"reference specific values with backticked dot-and-index paths"* — so the state
# is nested JSON with stable field names the instructions can point at, like
# `` `note` `` and `` `vitals_flags` ``.

# %%
def build_state(case: dict) -> dict:
    return {
        "note": case["note"],
        "patient": case["patient"],
        "history": case["history"] or ["none reported"],
        "vitals_flags": vitals_flags(case["vitals"]) or ["all vitals within normal range"],
    }


print(build_state(CASES[3]))

# %% [markdown]
# ## Steps 4 & 5 — eight narrow questions, one request
#
# Each question evaluates exactly one property, which is what makes the answers
# inspectable and recomposable. All eight go in a single `system_one()` call: they
# run in parallel, they cannot see each other's answers, and per the
# [parallel questions cookbook](https://docs.typesafe.ai/cookbooks/parallel_questions.md)
# batching is dramatically cheaper than one call each (measured at the end).
#
# Choices made deliberately:
#
# - **`Score` levels describe situations, not degrees.** The docs are blunt that
#   *"the model doesn't see a level's number or its neighbours"* and that numbered
#   or comparative wording lowers confidence, so each level stands alone.
# - **One `Noul` per red flag**, not one multi-label question. Several can be true
#   at once, and a Noul is the probability of yes for a single condition.
# - **`onset` is a `Choice`, not arithmetic.** Jev *"reads dates as text, not as
#   ordered quantities"*, so it bins the onset semantically and code decides what a
#   bin means.
# - **`imaging_likely` is speculative.** Its premise is stated inside the
#   instructions, and code only reads it when the disposition is an ED one — the
#   [fan-out pattern](https://docs.typesafe.ai/patterns/fan-out.md).

# %%
ACUITY_LEVELS = [
    "A minor, self-limiting problem, or an administrative request such as a "
    "medication refill or a note for work.",
    "Uncomfortable but stable, and safe to wait a day or two: an earache, a "
    "sprained ankle, a slowly spreading rash.",
    "Should be assessed today because it could get worse without treatment: "
    "persistent vomiting, a wound that needs closing, a fever with a bad cough.",
    "Should be assessed within minutes because it may deteriorate quickly: severe "
    "pain, breathlessness while sitting still, a deformed limb, heavy bleeding "
    "that has been controlled.",
    "Needs resuscitation started now: the airway is threatened, the person is "
    "unresponsive, in shock, seizing, or a heart attack or stroke appears to be in "
    "progress.",
]

DISPOSITIONS = {
    "resuscitation_now": "Move to a resuscitation bay immediately; a clinician is needed within seconds.",
    "ed_rapid": "Emergency department, seen ahead of the queue.",
    "ed_standard": "Emergency department, safe to wait in the normal queue.",
    "urgent_care": "An urgent care or same-day primary care appointment is enough.",
    "self_care": "Advice and self-care at home, with instructions on what would change that.",
}

QUESTIONS = {
    "acuity": Score(
        instructions="Judge how quickly this person needs to be seen, based on `note`, `vitals_flags` and `history`.",
        criteria=ACUITY_LEVELS,
    ),
    "disposition": Choice(
        instructions="Where should this person be sent right now, given `note` and `vitals_flags`?",
        criteria=DISPOSITIONS,
    ),
    "onset": Choice(
        instructions="According to `note`, how long ago did the current problem begin? Use what the note says rather than calculating.",
        criteria={
            "under_1h": "The note describes onset within roughly the last hour, including phrasings like 'just now' or a count of minutes.",
            "1h_to_6h": "The note places onset earlier today, several hours ago.",
            "6h_to_24h": "The note places onset yesterday or overnight.",
            "over_24h": "The note describes a problem lasting more than a day, such as several days or weeks.",
            "unclear": "The note does not say when the problem began.",
        },
    ),
    "flag_acs": Noul(
        instructions="Does `note` describe chest discomfort with features typical of a heart attack?",
        criteria={
            "true": "Chest pain or pressure with features such as radiation to the arm, jaw or back, sweating, nausea, or brought on by exertion.",
            "false": "No chest discomfort, or chest discomfort clearly explained by something else such as a cough, a rash, or an injury to the chest wall.",
        },
    ),
    "flag_stroke": Noul(
        instructions="Does `note` describe a sudden loss of brain function affecting one side of the body?",
        criteria={
            "true": "Sudden weakness, drooping or numbness on one side, slurred or lost speech, or sudden loss of vision on one side.",
            "false": "No such deficit, or the symptoms came on gradually over weeks, or have been present for years.",
        },
    ),
    "flag_sepsis": Noul(
        instructions="Does `note` describe an infection together with signs that the whole body is deteriorating?",
        criteria={
            "true": "A likely source of infection plus systemic decline such as new confusion, drowsiness, rigors, or mottled skin.",
            "false": "No infection, or an infection where the person is otherwise alert and behaving normally.",
        },
    ),
    "flag_airway": Noul(
        instructions="Does `note` describe a threat to breathing or to the airway itself?",
        criteria={
            "true": "Swelling of the lips, tongue or throat, a hoarse or lost voice with breathing difficulty, stridor, choking, or a sense of the throat closing.",
            "false": "Breathing is described as comfortable, or the only respiratory symptom is a cough or a blocked nose.",
        },
    ),
    "note_completeness": Score(
        instructions="How much of what a triage clinician would need is actually written down in `note`?",
        criteria=[
            "Barely anything: a few words with no description of the problem.",
            "The main complaint is named but key details are missing, such as when it started or how severe it is.",
            "Enough to act on: the complaint, roughly when it started, and how the person looks or is coping.",
        ],
    ),
    # Speculative: only read when the disposition is an ED one.
    "imaging_likely": Noul(
        instructions=(
            "Assume this person is going to the emergency department. Would the "
            "problem described in `note` usually be investigated with a scan or "
            "an X-ray early in that visit?"
        ),
    ),
}

print(len(QUESTIONS), "questions in one request")

# %% [markdown]
# ## One call, and what comes back
#
# A `Score` answer carries `score`, `probabilities` and `confidence`; a `Choice`
# carries `choice`, `probabilities` and `confidence`; a `Noul` carries just `noul`,
# the probability of yes, with no separate confidence.
#
# The fractional score is a probability-weighted mean, and the docs warn that the
# same number can come from very different distributions — `1.0` might be all the
# mass on level 1, or split between 0 and 2. So print the distribution, not only
# the mean.

# %%
case = CASES[3]  # the urosepsis one
response = client.system_one(state=build_state(case), questions=QUESTIONS)

print(f"model={response.model}  usage={response.usage}\n")

a = response.answers
print(f"acuity            {a['acuity'].score:.2f}  conf={a['acuity'].confidence:.2f}")
print(f"  distribution    {dict(a['acuity'].probabilities)}")
print(f"disposition       {a['disposition'].choice}  conf={a['disposition'].confidence:.2f}")
print(f"  distribution    {dict(a['disposition'].probabilities)}")
print(f"onset             {a['onset'].choice}  conf={a['onset'].confidence:.2f}")
print(f"note_completeness {a['note_completeness'].score:.2f}")
print()
for flag in ("flag_acs", "flag_stroke", "flag_sepsis", "flag_airway"):
    print(f"{flag:14} P(yes)={a[flag].noul:.2f}")

# %% [markdown]
# ## Step 6 — combine in code
#
# The model produced signals. Turning them into a decision is policy, and policy
# belongs in code where it can be read, versioned and changed without re-running
# inference. Two things worth noticing:
#
# - The escalation rule is **not** a weighted average. A weighted score suits
#   compensating preferences, but "any one red flag is enough" is a disjunction —
#   averaging would let a strong airway signal be diluted by a calm-looking note.
#   So the red flags are an `any()`, and the weighted score is only used for
#   ordering the queue.
# - `unstable_vitals` sits alongside the model's opinion with the power to escalate
#   on its own. Code that already knows the answer should not have to ask.

# %%
RED_FLAGS = ("flag_acs", "flag_stroke", "flag_sepsis", "flag_airway")
FLAG_THRESHOLD = 0.70


def normalized(answer) -> float:
    """Score -> 0..1, dividing by the top level index."""
    return answer.score / (len(answer.legend) - 1)


def compose(case: dict, response) -> dict:
    a = response.answers
    flags = vitals_flags(case["vitals"])
    fired = [f for f in RED_FLAGS if a[f].noul >= FLAG_THRESHOLD]

    acuity_n = normalized(a["acuity"])
    # Ordering signal only. 0.7 acuity / 0.3 worst red flag is a starting point
    # to evaluate on real data, not a validated weighting.
    priority = 0.7 * acuity_n + 0.3 * max((a[f].noul for f in RED_FLAGS), default=0.0)

    return {
        "id": case["id"],
        "acuity": round(a["acuity"].score, 2),
        "acuity_conf": round(a["acuity"].confidence, 2),
        "disposition": a["disposition"].choice,
        "disp_conf": round(a["disposition"].confidence, 2),
        "onset": a["onset"].choice,
        "red_flags": fired,
        "vitals_flags": flags,
        "priority": round(priority, 3),
        "escalate": bool(fired) or acuity_n >= 0.75 or unstable_vitals(flags),
        "thin_note": normalized(a["note_completeness"]) < 0.5,
        # Speculative answer, read only where its premise holds.
        "imaging_likely": (
            a["imaging_likely"].noul >= 0.5
            if a["disposition"].choice in ("resuscitation_now", "ed_rapid", "ed_standard")
            else None
        ),
    }


# %% [markdown]
# ## Step 7 — route on uncertainty
#
# [Confidence-gated routing](https://docs.typesafe.ai/patterns/confidence-routing.md)
# gives a universal floor of `0.6` and a higher bar for consequential actions. The
# thresholds below are deliberately asymmetric, which is the whole point of the
# pattern: thresholds scale with what going wrong costs.
#
# Sending someone home is the expensive mistake in triage, so `self_care` needs
# `0.85` and is blocked outright if any red flag fired or the note is too thin to
# judge. Escalating unnecessarily costs a clinician a few minutes, so it needs no
# confidence at all — an uncertain "this might be a stroke" should still move.

# %%
FLOOR = 0.60
DISCHARGE_BAR = 0.85


def route(d: dict) -> tuple[str, str]:
    """-> (action, why). Pure code; the model gets no vote here."""
    if d["escalate"]:
        why = ", ".join(d["red_flags"] + d["vitals_flags"]) or f"acuity {d['acuity']}"
        return "ESCALATE_NOW", f"escalated on {why}"

    if d["disp_conf"] < FLOOR:
        return "CLINICIAN_REVIEW", f"disposition confidence {d['disp_conf']} below floor {FLOOR}"

    if d["thin_note"]:
        return "REQUEST_MORE_INFO", "note too incomplete to triage"

    if d["disposition"] == "self_care":
        if d["disp_conf"] < DISCHARGE_BAR:
            return "CLINICIAN_REVIEW", f"self-care needs {DISCHARGE_BAR}, got {d['disp_conf']}"
        return "SELF_CARE_ADVICE", "confidently low acuity"

    return "AUTO_ROUTE", f"routed to {d['disposition']}"


# %% [markdown]
# ## Run the cohort
#
# One request per case, then the queue ordered by the composed priority.

# %%
rows = []
for c in CASES:
    r = client.system_one(state=build_state(c), questions=QUESTIONS)
    d = compose(c, r)
    d["action"], d["why"] = route(d)
    rows.append(d)

df = pd.DataFrame(rows).sort_values("priority", ascending=False)
pd.set_option("display.width", 200, "display.max_colwidth", 44)
print(df[["id", "acuity", "disposition", "disp_conf", "red_flags", "priority", "action"]].to_string(index=False))
print()
for r in rows:
    print(f"{r['id']}: {r['action']:18} {r['why']}")

# %% [markdown]
# ## Why one request and not eight
#
# The docs claim batching is both cheaper and faster. Worth confirming rather than
# taking on faith — this compares the batched call against one call per question
# on the same state.

# %%
state = build_state(CASES[0])

t0 = time.perf_counter()
batched = client.system_one(state=state, questions=QUESTIONS)
batched_s = time.perf_counter() - t0

t0 = time.perf_counter()
split = [client.system_one(state=state, questions={k: v}) for k, v in QUESTIONS.items()]
split_s = time.perf_counter() - t0

bt = batched.usage.input_tokens
st = sum(r.usage.input_tokens for r in split)
print(f"batched:  {bt:>6} input tokens   {batched_s:5.2f}s   (1 request)")
print(f"separate: {st:>6} input tokens   {split_s:5.2f}s   ({len(QUESTIONS)} requests)")
print(f"\nbatching is {st / bt:.1f}x cheaper and {split_s / batched_s:.1f}x faster")

# Same state, same questions - so the answers should also agree.
disagreements = [
    k
    for k, r in zip(QUESTIONS, split)
    for b, s in [(batched.answers[k], r.answers[k])]
    if getattr(b, "choice", None) != getattr(s, "choice", None)
]
print("choice disagreements between batched and separate:", disagreements or "none")

# %% [markdown]
# ## Notes for turning this into Alfred
#
# What this notebook establishes, and what it deliberately does not:
#
# - **The decomposition is the design.** Eight named signals, each independently
#   inspectable, with policy in `compose()` and `route()`. Changing the discharge
#   bar or a weight is a code change that needs no new inference, because the
#   evidence and the question meanings have not moved.
# - **The thresholds are placeholders.** `0.70` for a red flag and `0.85` for
#   discharge are the docs' starting points, not findings. They need to be set on
#   labelled cases from the population Alfred will actually see, with the cost of
#   each error type made explicit.
# - **Typed output guarantees the interface, not the truth.** A `ScoreAnswer` is
#   always well-formed. Whether `acuity` tracks real acuity is an empirical
#   question about this domain, and the one that matters most here.
# - **Jaggedness to respect**: no arithmetic and no date comparison in questions
#   (both are in code above); keep state tight, since accuracy falls as irrelevant
#   context grows; and note that a patient-supplied free-text field is
#   [adversarial input](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md) that
#   can steer a judgment, which is worth a screening question of its own.
# - **Next step**: assemble 50–100 labelled notes, then sweep `FLAG_THRESHOLD` and
#   `DISCHARGE_BAR` and look at the two error rates separately. The
#   [self-consistency cookbook](https://docs.typesafe.ai/cookbooks/consistency_noul_cookbook.md)
#   covers how to trade automation rate against agreement.
