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

| | mentioned | committed | |
| --- | --- | --- | --- |
| Blood cultures | 0.98 | 0.98 | missing from the chart — nudge |
| CT chest | 0.99 | 0.03 | declined — say so, do not nudge |
| Vancomycin | 0.97 | 0.04 | held — say so, do not nudge |
| Urinalysis | 0.02 | 0.02 | never came up — stay silent |

Separating the two is what lets the UI distinguish *never came up* from
*considered and ruled out*. A nudge to order a CT the physician just ruled out is
the kind that gets the whole feature switched off.

A second request then asks, for each genuinely missing order, **which sentence**
commits to it — with the note's own sentences as the options. The highlight is
therefore always a span the physician actually wrote, selected rather than
generated.

Per check: 32 questions, then one per missing order. Around 450 ms and ~8,000
input tokens, about $0.0003.

## Run it in Codespaces

[**Open in a Codespace**](https://codespaces.new/ygivenx/jev-try) — GitHub runs the
container, so there is nothing to install and nothing to host.

One-time setup: add a Codespaces secret named `TYPESAFE_API_KEY` at
[github.com/settings/codespaces](https://github.com/settings/codespaces), scoped to
this repository. The devcontainer declares it, so the codespace picks it up as an
environment variable.

On attach it installs dependencies and starts the server on port 8100, then opens a
preview. To share the URL, make the port public:

```sh
gh codespace ports visibility 8100:public -c $CODESPACE_NAME
```

Idle codespaces stop after 30 minutes, which is the right shape for a demo and the
wrong shape for anything permanent.

## Run it locally

Needs [uv](https://docs.astral.sh/uv/) and a TypeSafe API key.

```sh
cp .env.example .env      # then put your key in it
uv sync
uv run uvicorn app:app --port 8100 --reload
```

### Docker

```sh
docker build -t order-reconciliation .
docker run --rm -p 8100:8100 --env-file .env order-reconciliation
```

The image installs from `uv.lock` with `--no-dev` and runs as a non-root user. It
reads `$PORT` if the host sets one, which covers Render, Railway, Fly and Cloud Run
unchanged — set `TYPESAFE_API_KEY` as a secret there rather than baking it in.

## Why there is a server at all

GitHub Pages would be simpler, and it does not work here. The TypeSafe API
authenticates with a bearer token and offers no publishable or domain-scoped key,
so a static page would have to ship the secret to the browser — where, in a public
repo, it gets scraped and billed to you. The key stays server-side and the browser
only ever talks to `/api/check`. With no key set, the UI reports that rather than
failing silently.

## Layout

```
app.py                      catalogue, thresholds, the two Jev stages, the diff
index.html                  the whole frontend, no build step
Dockerfile                  self-contained image
.devcontainer/              Codespaces: secret wiring and auto-start
.github/workflows/build.yml builds the image, checks it serves and shuts down
```

## Not for clinical use

Synthetic patient, synthetic note, illustrative order catalogue. This is a
demonstration of a judgment API, not a medical device, and nothing here has been
validated for patient care.
