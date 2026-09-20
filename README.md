# Order reconciliation

A physician writes a plan in prose, then places orders in a different part of the
chart. The two drift apart. This is a single-page app that reads the note and tells
you what the note commits to that the chart does not yet carry out.

It is built on [TypeSafe](https://typesafe.ai)'s `jev` model, which returns typed
judgments with calibrated probabilities rather than text. Every threshold, every
diff and every line of copy lives in Python; the model only answers questions about
prose.

## The one design decision worth reading

A naive version asks "which orders are missing?" and nudges for anything the note
names. On the seeded case that produces two dangerous suggestions: the note says
*"CT chest is not indicated at this stage"* and *"Holding vancomycin unless he
deteriorates."* Both are named; neither should be ordered.

So each catalogue item gets **two atomic questions** instead of one:

| | mentioned | committed |
| --- | --- | --- |
| Blood cultures | 0.99 | 0.99 | → missing from chart, nudge |
| CT chest | 0.99 | 0.03 | → declined, say so, do not nudge |
| Vancomycin | 0.97 | 0.04 | → held, say so, do not nudge |
| Urinalysis | 0.02 | 0.02 | → never came up, stay silent |

Separating the two is what lets the UI distinguish *never came up* from
*considered and ruled out*. A nudge to order a CT the physician just ruled out is
the kind that gets the whole feature switched off.

A second request then asks, for each genuinely missing order, **which sentence**
commits to it — with the note's own sentences as the options. The highlight is
therefore always a span the physician actually wrote, selected rather than
generated.

Per check: 32 questions, then one per missing order. Around 450–560 ms and ~8,000
input tokens, about $0.0003.

## Run it

Needs [uv](https://docs.astral.sh/uv/) and a TypeSafe API key.

```sh
cp .env.example .env      # then put your key in it
uv sync
uv run uvicorn app:app --port 8100 --reload
```

Open http://127.0.0.1:8100. The key stays server-side — the browser only ever
talks to `/api/check`. With no key set, the UI reports that instead of failing
silently.

### Docker

```sh
docker build -t order-reconciliation .
docker run --rm -p 8100:8100 --env-file .env order-reconciliation
```

The image installs from `uv.lock` with `--no-dev`, so the notebook toolchain stays
out of it, and runs as a non-root user. It reads `$PORT` if the host sets one,
which covers Render, Railway, Fly and Cloud Run without changes — set
`TYPESAFE_API_KEY` as a secret there rather than baking it in.

## Notebooks

Two, both executed with outputs committed. Each is paired with a `.py` file in
jupytext `py:percent` format — edit either side and `uv run jupytext --sync` keeps
them together.

- **`jev_feature_tour.ipynb`** — the whole SDK surface in 17 sections: the three
  primitives, batching, confidence, retries, the exception hierarchy, `response_model`,
  async, and where the model is unreliable.
- **`alfred_triage.ipynb`** — a clinical triage workflow following the docs' seven
  build steps, with code owning every numeric comparison and uncertainty routed to
  a human.

```sh
uv run jupyter lab
```

## Layout

```
app.py          FastAPI backend — catalogue, thresholds, the two Jev stages, the diff
index.html      the whole frontend, no build step
*.ipynb / *.py  paired notebooks
Dockerfile      self-contained image
```

## Not for clinical use

Synthetic patient, synthetic note, illustrative order catalogue. This is a
demonstration of a judgment API, not a medical device, and nothing here has been
validated for patient care.
