# SIGWAY docs — build brief (for the scheduled authoring run)

You are authoring the **preliminary** SIGWAY documentation on the `docs` branch of the
**public** repo (`jonaselgammal/SIGWAY`). Commit your work **directly to `docs`**.
Do **not** touch other branches; do **not** delete any branches.

## Context
SIGWAY is a scientific Python package for **scalar-induced gravitational waves (SIGWs)**.
This branch was forked from `main` (the official/public code). The unreleased
Poltergeist (C++ backend) and constant-w extensions are **not** on this branch —
do not document them beyond the placeholder under `Extensions`.

## Hard constraints
- **Document only what exists on this branch** (the official package). Read the source
  to ground everything you write; do not invent APIs. If unsure whether something
  exists, leave a clearly marked `TODO` rather than guessing.
- **No example notebooks yet.** Example *pages* (simple/advanced) may contain code
  blocks, but they will be executed/verified locally later — mark them
  `<!-- needs local verification -->`. Do not try to `pip install sigway` or run jax.
- **Do not finalize the API reference.** mkdocstrings imports the package (needs jax),
  which won't work here. Leave `docs/api/index.md` as the wired-but-stub page and keep
  the mkdocstrings/jupyter plugins commented in `mkdocs.yml`.
- Build must succeed with only `pip install -r requirements-docs.txt` minus the
  commented-out (local-only) plugins — i.e. `mkdocs-material` + `pymdown-extensions`.
  Run `mkdocs build --strict` before committing and fix warnings.

## Design — already wired, keep it
- Assets live in `docs/assets/images/`, styles in `docs/stylesheets/extra.css`
  (palette, dark mode, Righteous headings, watermark background). The full design
  system is documented in `brand/DESIGN.md`. **Do not restyle**; reuse these.
- Palette: indigo `#20305f`, cream `#f2ead6`, cyan `#3a8cc4`, red `#e0455c`.
- Use admonitions (`!!! note`, `!!! warning`) for asides; cyan=note, red=warning are
  already themed.

## Tone & length
Pedagogical but **concise**. Aim for the sweet spot between "50 pages nobody reads" and
"one notebook is the docs." Explain *which knob means what physically*; prefer short
sections, worked snippets, and math where it clarifies (LaTeX via `\( \)` / `\[ \]`).

## Structure to deliver (flesh out the existing stubs; keep the nav in `mkdocs.yml`)
1. **Home** (`index.md`) — short pitch + where to go next.
2. **Getting started**
   - **Installation** — pip install, Python/OS support, optional extras.
   - **Simple example** — the *minimum* to get one result (import → set up → compute one
     spectrum → plot). Explain each knob briefly.
   - **Advanced example** — more knobs / sophistication (alternative kernels, parameter
     sweeps, performance notes).
3. **Theory & modelling** — how SIGWAY models SIGWs: the physics, the kernel/integral,
   key assumptions, and the parameter ↔ physics mapping. This is the most important
   "scientific package" section after the examples.
4. **API reference** — leave as the stub described above.
5. **Extensions** — keep the placeholder; note Poltergeist + constant-w arrive with the
   paper.

## When done
- `mkdocs build --strict` clean.
- Commit to `docs` with a clear message; push to `origin`.
- In the commit body, list what still needs the **local** session: execute/verify the
  example pages, build the notebooks, and turn on the mkdocstrings API reference.
