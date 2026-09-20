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
# # Jev / TypeSafe feature tour
#
# Every feature of the TypeSafe Python SDK, one small cell at a time. Each cell
# stands alone and makes at most one API call, so you can jump anywhere.
#
# | | Section |
# |---|---|
# | 1 | Setup, env vars, client options |
# | 2 | Models |
# | 3 | State: string, object, list |
# | 4 | `Noul` — probability of yes |
# | 5 | `Choice` — one of a set |
# | 6 | `Score` — position on a scale |
# | 7 | Structured (JSON) instructions and criteria |
# | 8 | Many questions in one request |
# | 9 | Confidence, and where it comes from |
# | 10 | The response object |
# | 11 | Model selection, timeouts, headers |
# | 12 | Retries |
# | 13 | Errors |
# | 14 | Logging |
# | 15 | Typed responses |
# | 16 | Async |
# | 17 | Limits, cost, and known rough edges |

# %% [markdown]
# ## 1. Setup
#
# Four environment variables configure the SDK. Only the first is required.

# %%
import asyncio
import concurrent.futures
import logging
import os

from dotenv import load_dotenv
from typesafe_sdk import constants

load_dotenv()

for name in ("API_KEY_ENV", "BASE_URL_ENV", "DEFAULT_MODEL_ENV", "LOG_LEVEL_ENV"):
    var = getattr(constants, name)
    print(f"{var:26} set={'yes' if os.environ.get(var) else 'no'}")

print()
print("DEFAULT_BASE_URL", constants.DEFAULT_BASE_URL)
print("DEFAULT_MODEL   ", constants.DEFAULT_MODEL)
print("DEFAULT_TIMEOUT ", constants.DEFAULT_TIMEOUT, "seconds")

# %% [markdown]
# `TypeSafeClient()` with no arguments reads `TYPESAFE_API_KEY` from the
# environment. Every constructor argument is keyword-only, and every one of them
# can also be overridden per call on `system_one()`.

# %%
from typesafe_sdk import Choice, Noul, RetryPolicy, Score, TypeSafeClient

client = TypeSafeClient()  # everything below uses this

# The full set of knobs, for reference:
#   TypeSafeClient(api_key=..., model=..., retry=RetryPolicy(...), timeout=...,
#                  headers={...}, transport=..., http_client=..., base_url=...)

# A short clinical message reused throughout, to keep every call cheap.
NOTE = (
    "Sore throat and a mild fever for two days. Still eating and drinking "
    "normally. Asking whether she needs antibiotics before a flight on Friday."
)
print("ready")

# %% [markdown]
# ## 2. Models
#
# `client.models.list()` returns `ModelMetadata` entries. The aliases are what you
# pin against; `response.model` tells you which concrete version answered.

# %%
for m in client.models.list().models:
    print(f"{m.name:14} {m.release_date[:10]}  {m.description}")

# %% [markdown]
# ## 3. State
#
# State is the context the model reads. It can be a **string**, a **JSON object**,
# or a **list of strings** — text only, no images or audio. The docs recommend an
# object for most requests so each part has a descriptive name.
#
# With an object you can point instructions at a specific field using a backticked
# path, including indexes: `` `messages[0].text` ``.

# %%
# (a) a bare string
r = client.system_one(state=NOTE, questions={"q": Noul(instructions="Is a fever mentioned?")})
print("string state  ->", r.answers["q"].noul)

# (b) a list of strings
r = client.system_one(
    state=["Sore throat for two days.", "Mild fever.", "Flying on Friday."],
    questions={"q": Noul(instructions="Is a fever mentioned?")},
)
print("list state    ->", r.answers["q"].noul)

# (c) a nested object, with the question naming an exact path
r = client.system_one(
    state={
        "patient": {"age": 23},
        "messages": [{"from": "patient", "text": NOTE}],
        "clinic_policy": "Antibiotics are not indicated for viral sore throat.",
    },
    questions={
        "q": Noul(instructions="Does `messages[0].text` mention a fever?"),
        "policy": Noul(instructions="Would `clinic_policy` discourage antibiotics for `messages[0].text`?"),
    },
)
print("object state  ->", r.answers["q"].noul, "| policy:", r.answers["policy"].noul)

# %% [markdown]
# ## 4. `Noul` — the probability of yes
#
# A Noul asks one yes/no question and returns `noul`, the probability of yes,
# between 0 and 1. There is **no separate confidence** on a Noul: the number is
# both the answer and the uncertainty. `0.5` means genuinely torn, not "medium".
#
# Use one Noul per condition when several conditions could be true at once.

# %%
r = client.system_one(
    state=NOTE,
    questions={
        "urgent": Noul(instructions="This person needs to be seen urgently."),
        "infection": Noul(instructions="This describes symptoms of an infection."),
        "wants_antibiotics": Noul(instructions="This person is asking about antibiotics."),
    },
)
for k, a in r.nouls.items():
    print(f"{k:18} P(yes)={a.noul:.2f}")

# %% [markdown]
# Optional `criteria` pins down the boundary between yes and no with `true` and
# `false` descriptions. This is the fix for the docs' first known rough edge —
# Jev "answers the question you wrote, not the one you meant", so write the edges
# down rather than hoping they're inferred.

# %%
vague = Noul(instructions="Is this serious?")
pinned = Noul(
    instructions="Does this need a same-day appointment?",
    criteria={
        "true": "Symptoms that could worsen within hours without treatment.",
        "false": "A mild, self-limiting illness where the person is eating, drinking and behaving normally.",
    },
)
r = client.system_one(state=NOTE, questions={"vague": vague, "pinned": pinned})
print(f"vague  'is this serious?'        -> {r.answers['vague'].noul:.2f}")
print(f"pinned 'same-day appointment?'   -> {r.answers['pinned'].noul:.2f}")
print("\nSame underlying situation. The second question is answerable; the first isn't.")

# %% [markdown]
# ## 5. `Choice` — exactly one of a set
#
# `criteria` is a mapping of option name to description. Descriptions are optional
# (`None` works) but they carry most of the accuracy. You get back `choice`, a
# `probabilities` distribution over **every** option, and `confidence`.

# %%
r = client.system_one(
    state=NOTE,
    questions={
        "route": Choice(
            instructions="Where should this message be sent?",
            criteria={
                "emergency": "Needs emergency care now.",
                "same_day_gp": "Needs a doctor today.",
                "routine_gp": "A routine appointment in the next week or two.",
                "pharmacist": "A pharmacist can resolve this without a doctor.",
                "self_care": "Advice to manage at home is enough.",
            },
        )
    },
)
a = r.answers["route"]
print("choice     ", a.choice)
print("confidence ", round(a.confidence, 3))
for opt, p in sorted(a.probabilities.items(), key=lambda kv: -kv[1]):
    print(f"  {opt:14} {p:5.2f} {'#' * int(p * 40)}")

# %% [markdown]
# Always include a way to say "none of these". The model must pick one of your
# options, so if reality isn't in the list you get a confidently wrong answer
# rather than an admission. Note what happens to the distribution when the real
# answer is missing:

# %%
r = client.system_one(
    state="My laptop will not turn on.",
    questions={
        "no_escape": Choice(
            instructions="What is this about?",
            criteria={"sore_throat": None, "rash": None, "headache": None},
        ),
        "with_escape": Choice(
            instructions="What is this about?",
            criteria={
                "sore_throat": None,
                "rash": None,
                "headache": None,
                "not_medical": "Not a medical problem at all.",
            },
        ),
    },
)
print("no escape hatch:", r.answers["no_escape"].choice, "conf", round(r.answers["no_escape"].confidence, 2))
print("with escape:    ", r.answers["with_escape"].choice, "conf", round(r.answers["with_escape"].confidence, 2))

# %% [markdown]
# ## 6. `Score` — position on an ordered scale
#
# `criteria` is an ordered list of 2 to 10 levels. Two rules from the docs matter
# more than anything else here:
#
# - **Describe situations, not degrees.** "Broken feature, workaround exists" beats
#   "moderately severe".
# - **Each level must stand alone**, because "the model doesn't see a level's
#   number or its neighbours". Don't write comparatives, and don't number them —
#   numbering measurably lowers confidence.
#
# `score` is a probability-weighted mean, so it can land between levels. `legend`
# and `probabilities` are keyed by level index starting at 0.

# %%
r = client.system_one(
    state=NOTE,
    questions={
        "severity": Score(
            instructions="How quickly does this person need to be seen?",
            criteria=[
                "A minor complaint that will settle on its own.",
                "Uncomfortable but safe to wait a few days.",
                "Should be looked at today.",
                "Needs attention within the hour.",
            ],
        )
    },
)
a = r.answers["severity"]
print(f"score      {a.score:.2f}   (0 .. {len(a.legend) - 1})")
print(f"confidence {a.confidence:.2f}\n")
for i, text in a.legend.items():
    print(f"  [{i}] p={a.probabilities[i]:.2f}  {text}")

# %% [markdown]
# **The mean hides the distribution.** A score of `1.0` can mean all the mass on
# level 1, or an even split between levels 0 and 2 — the same number describing
# confident-middling and wildly-torn. So read `probabilities` whenever the decision
# matters, or read `confidence`, which is what collapses the shape into one number.

# %%
peaked = {0: 0.0, 1: 1.0, 2: 0.0}
split = {0: 0.5, 1: 0.0, 2: 0.5}
for name, d in (("peaked", peaked), ("split", split)):
    mean = sum(i * p for i, p in d.items())
    print(f"{name:7} distribution={d}  mean={mean:.1f}")

# %% [markdown]
# ### Choosing between the three
#
# | Need | Use | Returns |
# |---|---|---|
# | Is this condition true? | `Noul` | `noul` (probability of yes) |
# | Which one of these? | `Choice` | `choice`, `probabilities`, `confidence` |
# | How much, along one dimension? | `Score` | `score`, `legend`, `probabilities`, `confidence` |
#
# One dimension per question. If you catch yourself writing "how severe *and*
# urgent", that's two Scores and a weighted sum in code.

# %% [markdown]
# ## 7. Structured instructions and criteria
#
# `instructions` and every criteria entry accept a **string, object, array, or
# null** — not just strings. Labelled keys beat a long sentence when a question has
# several parts, and you can drop existing data straight in instead of serializing
# it into prose.
#
# Useful keys by convention: `what`, `not_for`, `examples` for Choice options;
# `summary`, `signals` for Score levels.

# %%
r = client.system_one(
    state=NOTE,
    questions={
        "topic": Choice(
            instructions={"question": "What is this message asking for?", "read": "`state`"},
            criteria={
                "medication_question": {
                    "what": "Asking whether to take, start or stop a medicine.",
                    "not_for": "Reporting a side effect that has already happened.",
                    "examples": ["do I need antibiotics", "can I stop my statin"],
                },
                "symptom_report": {
                    "what": "Describing symptoms without asking about a medicine.",
                    "examples": ["my throat hurts"],
                },
                "admin": {"what": "Appointments, letters, referrals, paperwork."},
            },
        ),
        "fit_to_fly": Noul(
            instructions={
                "field": {"name": "fit_to_fly", "description": "Whether commercial air travel is reasonable."},
                "question": "Based on `state`, does this person appear fit to fly?",
            },
            criteria={
                "true": {"what": "Mild, stable illness with no breathing difficulty."},
                "false": {"what": "Unstable, infectious with systemic illness, or short of breath."},
            },
        ),
    },
)
print("topic      ", r.answers["topic"].choice, "conf", round(r.answers["topic"].confidence, 2))
print("fit_to_fly ", round(r.answers["fit_to_fly"].noul, 2))

# %% [markdown]
# ## 8. Many questions in one request
#
# Questions in a single call **run in parallel and cannot see each other's
# answers**. Batching is much cheaper than one call per question, because the state
# is sent once instead of N times.
#
# This also licenses *speculative* questions: ask branch-specific things up front,
# state each premise explicitly in the instructions, and let code read only the
# answers whose branch actually applies.

# %%
r = client.system_one(
    state=NOTE,
    questions={
        "route": Choice(
            instructions="Where should this go?",
            criteria={"pharmacist": None, "same_day_gp": None, "self_care": None},
        ),
        "infection": Noul(instructions="This describes an infection."),
        "severity": Score(
            instructions="How unwell does this person sound?",
            criteria=["Barely unwell.", "Moderately unwell.", "Severely unwell."],
        ),
        # Speculative: premise stated in the instructions, read only if routed to a pharmacist.
        "otc_sufficient": Noul(
            instructions="Assume a pharmacist sees this person. Would over-the-counter treatment be enough?"
        ),
    },
)

# Three convenience views, filtered by type, plus the full dict.
print("answers :", sorted(r.answers))
print("nouls   :", sorted(r.nouls))
print("choices :", sorted(r.choices))
print("scores  :", sorted(r.scores))

# Code decides whether the speculative answer is relevant.
route = r.answers["route"].choice
print(f"\nroute={route}")
if route == "pharmacist":
    print("otc_sufficient is relevant ->", round(r.answers["otc_sufficient"].noul, 2))
else:
    print("otc_sufficient ignored (its premise does not hold)")

# %% [markdown]
# ## 9. Confidence, and where it comes from
#
# Confidence is **not a second opinion from the model** — it is a statistic
# computed from `probabilities`. It measures how peaked the distribution is. The
# docs give the three-option formula as `(3 × pmax − 1) / 2`, which generalises to
# `(n × pmax − 1) / (n − 1)`. Worth confirming rather than believing — and the check
# below bounds how much the 2-decimal rounding of `probabilities` could account for:

# %%
r = client.system_one(
    state=NOTE,
    questions={
        "three": Choice(
            instructions="How urgent is this?",
            criteria={"low": None, "medium": None, "high": None},
        ),
        "five": Choice(
            instructions="Which service fits best?",
            criteria={"emergency": None, "same_day_gp": None, "routine_gp": None, "pharmacist": None, "self_care": None},
        ),
    },
)
for key in ("three", "five"):
    a = r.answers[key]
    n = len(a.probabilities)
    pmax = max(a.probabilities.values())
    predicted = (n * pmax - 1) / (n - 1)
    # probabilities arrive rounded to 2dp, so bound what rounding alone could explain
    lo = ((pmax - 0.005) * n - 1) / (n - 1)
    hi = ((pmax + 0.005) * n - 1) / (n - 1)
    explained = lo <= a.confidence <= hi
    print(
        f"{key:6} n={n} pmax={pmax:.2f}  reported={a.confidence:.3f}  "
        f"formula={predicted:.3f}  rounding band=[{lo:.3f}, {hi:.3f}]  "
        f"{'within' if explained else 'OUTSIDE'}"
    )

# %% [markdown]
# The formula tracks the reported value closely, and usually lands inside the band
# that 2-decimal rounding can account for. Not always, though — re-run this cell a
# few times and you will see the occasional `OUTSIDE`, most often when the top two
# options are nearly tied, which suggests confidence is computed from full-precision
# probabilities before they are rounded for the response.
#
# So treat `(n × pmax − 1) / (n − 1)` as a good mental model of what confidence
# *means*, not as a reimplementation of it. Read `a.confidence`; don't recompute it.
#
# Because it is derived, confidence tells you *whether to act*, never *what is
# true*. The documented thresholds, which you are expected to re-tune on your own
# data:
#
# | Confidence | Meaning |
# |---|---|
# | below `0.6` | universal floor — route to a human |
# | `0.6`–`0.85` | act, but confirm first |
# | above `0.85`–`0.9` | safe to act automatically on high-stakes calls |
#
# Two traps. Low confidence on a harmless choice doesn't matter — several equally
# fine options legitimately split the mass. And confidence on an unused branch is
# noise, so ignore it.

# %%
a = r.answers["five"]
FLOOR, HIGH = 0.60, 0.85
if a.confidence < FLOOR:
    print(f"{a.confidence:.2f} -> human review")
elif a.confidence < HIGH:
    print(f"{a.confidence:.2f} -> act on '{a.choice}' after confirmation")
else:
    print(f"{a.confidence:.2f} -> act on '{a.choice}' automatically")

# %% [markdown]
# ## 10. The response object
#
# `SystemOneResponse` carries the model, token usage, the answers, and an escape
# hatch to the raw HTTP response.

# %%
r = client.system_one(state=NOTE, questions={"q": Noul(instructions="Is a fever mentioned?")})

print("model            ", r.model)
print("usage            ", r.usage, f"(input={r.usage.input_tokens}, output={r.usage.output_tokens})")
print("request_id       ", r.request_id)
print("answers          ", dict(r.answers))
print("answer type tag  ", r.answers["q"].type)
print("raw HTTP status  ", r.raw_http_response.status_code)
print("raw JSON         ", r.raw_http_response.json())

# %% [markdown]
# `raw_http_response` is the forward-compatibility hatch: if the API starts
# returning an answer kind this SDK version doesn't model, it still reaches you
# there. `request_id` is the thing to quote in a bug report.

# %% [markdown]
# ## 11. Model selection, timeouts, headers
#
# All three can be set on the client or overridden per call. `jev-preview` is
# described as "should be better in most ways" — but check `response.model`, because
# right now both aliases resolve to the same concrete version, so the small
# differences below are ordinary run-to-run variation rather than two models
# disagreeing. Pin the concrete version when you need reproducibility.

# %%
q = {"severity": Score(
    instructions="How unwell does this person sound?",
    criteria=["Barely unwell.", "Moderately unwell.", "Severely unwell."],
)}

for model in ("jev-latest", "jev-preview"):
    r = client.system_one(state=NOTE, questions=q, model=model, timeout=20.0)
    a = r.answers["severity"]
    print(f"{model:12} -> {r.model:12} score={a.score:.2f} conf={a.confidence:.2f}")

# Per-call extras: extra_headers merges into the request, extra_body into the payload.
r = client.system_one(
    state=NOTE,
    questions={"q": Noul(instructions="Is a fever mentioned?")},
    extra_headers={"X-Trace": "feature-tour"},
)
print("\nwith extra header ->", r.answers["q"].noul)

# %% [markdown]
# ## 12. Retries
#
# `RetryPolicy` is a dataclass with sensible defaults: 2 retries, exponential
# backoff from 0.5s capped at 5s with 25% jitter, retrying `408`, `429` and all
# `5xx`, honouring `Retry-After`, under a 30s total budget per call.
#
# `timeout` here is the budget for *all* attempts together, not per attempt.

# %%
d = RetryPolicy()
print("max_retries        ", d.max_retries)
print("backoff_initial    ", d.backoff_initial, "s, doubling each attempt")
print("backoff_max        ", d.backoff_max, "s")
print("backoff_jitter     ", d.backoff_jitter, "(random fraction subtracted)")
print("respect_retry_after", d.respect_retry_after)
print("timeout            ", d.timeout, "s total budget for all attempts")
print("http_statuses      ", f"{{408, 429}} + all of 500-599  ({len(d.http_statuses)} codes)")

careful = RetryPolicy(max_retries=5, backoff_initial=0.25, backoff_max=8.0, timeout=60.0)
none_at_all = RetryPolicy(max_retries=0)

# Settable per client or per call; the per-call value wins.
r = client.system_one(
    state=NOTE, questions={"q": Noul(instructions="Is a fever mentioned?")}, retry=careful
)
print("\nsurvived with retry policy ->", r.answers["q"].noul)

# `predicate` takes over the decision entirely when the flags aren't enough.
custom = RetryPolicy(predicate=lambda exc: "temporarily" in str(exc).lower())
print("custom predicate policy built:", custom.max_retries, "retries")

# %% [markdown]
# ## 13. Errors
#
# ```
# TypeSafeError
# ├── TypeSafeAPIError                     .status .body .headers .endpoint .request_id
# │   ├── TypeSafeBadRequestError          400
# │   ├── TypeSafeAuthenticationError      401
# │   ├── TypeSafePermissionDeniedError    403
# │   ├── TypeSafeNotFoundError            404
# │   ├── TypeSafeUnprocessableEntityError 422
# │   ├── TypeSafeRateLimitError           429   + .retry_after_ms
# │   ├── TypeSafeInternalServerError      5xx
# │   └── TypeSafeAPIResponseValidationError     + .field_path
# └── TypeSafeAPIConnectionError                 (also a builtin ConnectionError)
#     └── TypeSafeAPITimeoutError                + .timeout
# ```
#
# Catching `TypeSafeAPIError` covers everything the server answered; catching
# `TypeSafeError` covers that plus connection failures. Two real ones:

# %%
from typesafe_sdk import (
    TypeSafeAPIError,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeError,
)

# (a) a bad key -> 401
try:
    TypeSafeClient(api_key="apikey_not_a_real_key").system_one(
        state="hi", questions={"q": Noul(instructions="Is this a greeting?")}
    )
except TypeSafeAuthenticationError as e:
    print(f"401  status={e.status} endpoint={e.endpoint}")
    print(f"     request_id={e.request_id}")
    print(f"     body={e.body}")

# (b) an impossible timeout, with retries off so it fails fast
try:
    client.system_one(
        state=NOTE,
        questions={"q": Noul(instructions="Is a fever mentioned?")},
        timeout=0.001,
        retry=RetryPolicy(max_retries=0),
    )
except TypeSafeAPITimeoutError as e:
    print(f"\ntimeout  timeout={e.timeout}  is ConnectionError={isinstance(e, ConnectionError)}")
except TypeSafeError as e:
    print(f"\nother TypeSafeError: {type(e).__name__}: {e}")

# %% [markdown]
# A malformed question fails before any tokens are spent. The docs say a `Score`
# takes 2 to 10 levels — but only one half of that is enforced, which is worth
# knowing:

# %%
from typesafe_sdk import TypeSafeBadRequestError, TypeSafeUnprocessableEntityError

for n, label in ((11, "11 levels (above the documented max)"), (1, "1 level (below the documented min)")):
    try:
        r = client.system_one(
            state=NOTE,
            questions={"q": Score(instructions="How bad?", criteria=[f"situation {i}" for i in range(n)])},
        )
        print(f"{label:38} ACCEPTED -> score={r.answers['q'].score:.2f}")
    except (TypeSafeBadRequestError, TypeSafeUnprocessableEntityError) as e:
        print(f"{label:38} {type(e).__name__} {e.status}: {e.body['detail']}")

# %% [markdown]
# So the **maximum is enforced and the minimum is not**. A one-level Score is
# accepted and returns `0.00` — a number that looks like an answer and means
# nothing, since there is nowhere else on the scale for the mass to go. Validate
# your own question shapes; don't rely on the API to catch this one.

# %% [markdown]
# ## 14. Logging
#
# The SDK logs on the `typesafe_sdk` logger, or set `TYPESAFE_LOG_LEVEL`. Secret
# headers are redacted for you — `authorization`, API keys, cookies, and any header
# whose name contains `token` or `secret` — so debug logs are safe to paste into an
# issue. Verify rather than trust:

# %%
records: list[str] = []


class Capture(logging.Handler):
    def emit(self, record):
        records.append(self.format(record))


log = logging.getLogger("typesafe_sdk")
handler = Capture()
handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
log.addHandler(handler)
log.setLevel(logging.DEBUG)

client.system_one(state="hi", questions={"q": Noul(instructions="Is this a greeting?")})

log.setLevel(logging.WARNING)
log.removeHandler(handler)

print(f"{len(records)} log records\n")
for line in records[:4]:
    print(line[:200])

leaked = [r for r in records if os.environ["TYPESAFE_API_KEY"] in r]
print(f"\nrecords containing the raw API key: {len(leaked)}")

# %% [markdown]
# ## 15. Typed responses
#
# `response_model` swaps in your own class so downstream code gets real attributes
# instead of dictionary lookups. Subclass `SystemOneResponse` and the SDK validates
# into it.

# %%
from typesafe_sdk import ChoiceAnswer, NoulAnswer, SystemOneResponse


class TriageResponse(SystemOneResponse):
    @property
    def route(self) -> ChoiceAnswer:
        return self.choices["route"]

    @property
    def infection(self) -> NoulAnswer:
        return self.nouls["infection"]

    @property
    def needs_human(self) -> bool:
        return self.route.confidence < 0.6


typed = client.system_one(
    state=NOTE,
    questions={
        "route": Choice(
            instructions="Where should this go?",
            criteria={"pharmacist": None, "same_day_gp": None, "self_care": None},
        ),
        "infection": Noul(instructions="This describes an infection."),
    },
    response_model=TriageResponse,
)
print(type(typed).__name__)
print("route       ", typed.route.choice, round(typed.route.confidence, 2))
print("infection   ", round(typed.infection.noul, 2))
print("needs_human ", typed.needs_human)

# %% [markdown]
# ## 16. Async
#
# `AsyncTypeSafeClient` has the same surface with `await`. It is the right client
# when you are fanning out over many *different* states — many questions about
# *one* state should be a single batched call instead.
#
# (The helper runs the coroutine on its own thread so this cell works both in a
# notebook, which already has an event loop, and as a plain script.)

# %%
from typesafe_sdk import AsyncTypeSafeClient

MESSAGES = [
    NOTE,
    "Crushing chest pain for twenty minutes, sweating, pain going into my left arm.",
    "I need a repeat prescription for my blood pressure tablets.",
]


def run_async(coro):
    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        return ex.submit(asyncio.run, coro).result()


async def triage_all(messages: list[str]):
    async with AsyncTypeSafeClient() as ac:
        tasks = [
            ac.system_one(
                state=m,
                questions={"urgent": Noul(instructions="This person needs to be seen urgently.")},
            )
            for m in messages
        ]
        return await asyncio.gather(*tasks)


for msg, res in zip(MESSAGES, run_async(triage_all(MESSAGES))):
    print(f"P(urgent)={res.answers['urgent'].noul:.2f}  {msg[:58]}")

# %% [markdown]
# ## 17. Limits, cost, and known rough edges
#
# **Budgets:** 64k tokens per request in total, and 32k for the state plus the
# longest single question. **Cost:** $0.042 per million input tokens; output tokens
# are free. State is text only, and English is the strongest language.
#
# **The rough edges for `jev-1.13`**, from the
# [jaggedness page](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md) — the
# first three are the ones that bite in practice:
#
# | Edge | What to do |
# |---|---|
# | Literal reading | Write the boundary conditions into `criteria` |
# | Math and counting | Do it in code; pass words, not numbers |
# | Date comparison | Reads dates as text; extract parts, compare in code |
# | Indirection, double negatives | Phrase directly, name the field |
# | Irrelevant context | Send only fields the questions need |
# | Adversarial content | Injected instructions can steer it; screen inputs |
# | Contradictory instructions | Keep `instructions` and `criteria` aligned |
# | Structural invariants | `P(yes) + P(no)` is not guaranteed to be 1 |
# | Text generation | Not trained for it; use bounded options |
#
# The arithmetic one, demonstrated — note this is *not* a bug to work around with
# better prompting, it's a capability boundary, and the fix is to not ask:

# %%
DOSES = {"amoxicillin_mg": 500, "doses_per_day": 3, "days": 7}

r = client.system_one(
    state={"prescription": DOSES},
    questions={
        "over_10g": Noul(
            instructions="Does the total amount of amoxicillin across the whole course exceed 10000 mg?"
        ),
    },
)
total = DOSES["amoxicillin_mg"] * DOSES["doses_per_day"] * DOSES["days"]
print(f"model says P(total > 10000mg) = {r.answers['over_10g'].noul:.2f}")
print(f"code says  total = {total} mg -> {total > 10000}")
print("\nDo the arithmetic in code. Ask the model only what code can't compute.")

# %% [markdown]
# ## Where to go next
#
# - `alfred_triage.ipynb` in this folder puts all of the above together into one
#   workflow, following the docs' seven-step build guide.
# - [Cookbooks](https://docs.typesafe.ai/llms.txt) are the best source of
#   decompositions — reranking, hierarchical classification, extraction cascades,
#   guardrails, citation checks.
# - The two patterns worth internalising first:
#   [fan-out](https://docs.typesafe.ai/patterns/fan-out.md) (ask everything at
#   once, including speculatively) and
#   [confidence routing](https://docs.typesafe.ai/patterns/confidence-routing.md)
#   (thresholds proportional to what being wrong costs).
