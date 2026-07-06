# A multi-agent architecture for demand + supply QA in transportation planning

*How the pieces built here (three engines, seven gates, the calibrators, the adversarial
review) compose into one deployable, model-agnostic, learning-oriented system — and how the
same system extends from 4-step to activity-based (ABM / CT-RAMP) demand.*

---

## 1. Why agents

The work so far is not a monolith model; it is a set of small, single-responsibility programs
that **produce**, **tune**, **check**, and **criticize** a transport demand estimate, stage by
stage. That is already a multi-agent system in everything but name. Naming it buys three things:

1. **Pluggability** — any stage can host more than one solver (we already run three), and any
   solver can be swapped without touching the checkers.
2. **Model-agnostic verification** — the checkers validate *outputs against controls*, never a
   model's internals, so the same checkers work on 4-step gravity, TRMG2 nested-logit, or an
   ABM tour list.
3. **A learning surface** — a student swaps an agent, re-runs the gates, and *sees* the
   consequence. The architecture teaches the discipline, not just the answer.

The governing rule is one line: **every checker leads with magnitude / control-total bias,
then pattern; a red gate blocks interpreting the stages below it.**

---

## 2. The six agent types

| Agent type | Job | Current instances (files) |
|---|---|---|
| **Reference** | serve the ground truth for a stage (control totals, targets, a reference distribution) | survey `eda_scheme6` (controls), ITRE published %RMSE/VMT (targets), Engine A (shape reference) |
| **Engine** (solver) | produce demand at a stage | A nested-logit · B power gravity · C Grid2Demand gamma (`compare_engines.py`, `make_od_4period.py`) |
| **Calibrator** | tune an engine's parameters to a control | deterrence calibration; C0→C1 2-param gamma fit (`compare_engines.py`, `gen_engine_registry.py`) |
| **Gate** (checker) | validate a stage magnitude-first, emit RAG + verdict | Gates 0–5 + Gate ④ (`stage_audit.py`, `demand_checks.py`, `od_validation.py`, `vmt_vht_gate.py`) |
| **Critic** (adversarial) | attack the claims: what's green-by-construction, what's circular, what's unbenchmarked | the simulated review (`private/REVIEW*_*.md`) |
| **Orchestrator** | sequence the ladder, enforce blocking rules, hold the registry | `model_controls.yml` + `stage_audit.py` |

The **critic** is the agent most systems omit and the reason this one is honest: it is what
caught that Gates 1–3 pass by construction and that "C matches A" is circular.

---

## 3. The layered architecture

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │  ORCHESTRATOR   model_controls.yml  ·  ladder + blocking rules       │
 └─────────────────────────────────────────────────────────────────────┘
        │ sequences, gates, blocks-downstream-on-red
        ▼
 DATA / REFERENCE          SE data · survey controls · network · ITRE targets
        │
        ▼
 ┌──────────────┐   STAGE 1 generation   ┌──────────────┐
 │  ENGINE      │   STAGE 2 distribution │  CALIBRATOR  │  tune params → control
 │  agents      │◄──STAGE 3 mode ───────►│  agents      │
 │ A · B · C    │                        └──────────────┘
 └──────┬───────┘
        │ emits trip list / OD  (market × purpose × period × mode)
        ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  GATE agents   (magnitude-first)                                     │
 │  0 seed · 1 gen · 2 dist · 3 mode · 4 pre-OD  ── DEMAND side         │
 │  5 counts · ④ VMT/VHT                           ── SUPPLY-loaded     │
 └──────┬──────────────────────────────────────────────────────────────┘
        │ verdicts (RAG) + measured review/*
        ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  CRITIC agent   construction-vs-validation, circularity, null models │
 └─────────────────────────────────────────────────────────────────────┘
```

Two gate classes, kept distinct (a fix the critic forced):
- **Construction gates** (identity checks — did we preserve what we forced): generation total,
  distribution mean, mode share. These *cannot* fail as built; they confirm conservation.
- **Validation gates** (independent data): counts (Gate 5), VMT/VHT (Gate ④), and — when ITRE
  sends it — the official OD. These *can* fail and carry the real signal.

---

## 4. The invariant contract (what makes it model-agnostic)

Every engine, whatever its internals, emits the same object:

```
trip list / OD  keyed by  (market, purpose, period, mode)  →  loaded network  →  VMT / VHT
```

The gate agents only ever read that object and the reference controls. They never look inside
the engine. **That interface is the whole trick**: it is why a 4-step gravity and an ABM tour
model can be checked by the identical gate stack. The engine changes; the contract does not.

Design implication: adding a new engine or a new market means implementing the contract, not
editing the checkers. Adding a new check means one more gate agent reading the same contract.

---

## 5. Extending to activity-based demand (ABM / CT-RAMP)

The datasets and the *decision structure* differ — a tour-based ABM produces **tours**
(home→work→shop→home) that decompose into trips, with joint/intra-household constraints a
4-step model never represents. But the **contract is invariant**: an ABM still emits a trip
list that aggregates to an OD by market/purpose/period/mode. So:

| Concern | 4-step (here) | ABM / CT-RAMP | Gate impact |
|---|---|---|---|
| Unit of demand | purpose trips | tours → trips | same OD after decomposition |
| Generation control | survey `wTrips` | activity/tour rates | same Gate 1 (magnitude-first) |
| Distribution | gravity / DC | destination choice per tour leg | same Gate 2 (TLFD vs control) |
| Mode | shares / nested logit | tour-mode + trip-mode consistency | Gate 3 gains a **tour-mode consistency** check |
| Markets | resident + special | + intra-HH, joint, at-work subtours | more market labels, same Gate 4 inventory |
| Supply-loaded | counts, VMT/VHT | identical | Gates 5 / ④ unchanged |

New agents ABM needs (drop-in, same contract): a **tour→trip decomposition** engine, a
**tour-mode/trip-mode consistency** gate, and reference agents for the ABM's own control
totals. Everything downstream of the OD is reused verbatim. This is why the framework is worth
naming as an architecture rather than a script: **the marginal cost of a new demand paradigm is
a few agents, not a rewrite.**

---

## 6. Demand **+** supply consistency (closing the loop)

The gates already straddle both sides — Gates 0–4 are demand, Gates 5/④ are supply-loaded. The
system becomes a *consistent* demand+supply framework when the loop closes:

```
 demand engines ──OD──► assignment (supply) ──skims──► back to distribution/mode
                                    │
                                    └──► Flow-Through-Tensor snapshot (Δ, A, Π)
                                         reload → verify → converge (no re-run)
```

- The **skim-feedback** agent iterates until skim %RMSE ≤ 0.1 (MSA), making demand and supply
  mutually consistent rather than one-shot.
- The **Flow-Through-Tensor** snapshot (`tensor_ftt/`) is the supply-side artifact that lets an
  assignment be *reloaded and verified* (R²=1.0 on a consistent snapshot) instead of re-run —
  the supply analogue of the demand engine registry.
- Net: one orchestrator, demand engines + calibrators on the left, assignment + FTT + skim
  feedback on the right, gate agents spanning the middle. That is the "demand plus supply"
  learning framework, end to end.

---

## 7. As a learning framework

Each engine is a **teaching lens**, run on identical data so the difference is the lesson:
- **A nested-logit** — behavioral structure (what sophistication buys),
- **B power gravity** — the classic textbook form (and its failure modes, e.g. over-peaked
  short trips),
- **C Grid2Demand gamma** — a flexible 2-knob form that, tuned C0→C1, can *represent* A's
  distribution cheaply (the check-and-balance).

The gates teach **magnitude-first**; the critic teaches **construction ≠ validation**; the
demand+supply loop teaches **consistency**. A learner adds an agent, re-runs `stage_audit.py`,
and reads the consequence in the RAG verdicts — the framework is the curriculum.

---

## 8. Deployment & how to add an agent

- **Run the ladder:** `python compare_engines.py` (engines+calibrators) → `python
  od_validation.py` + `python vmt_vht_gate.py` (supply gates) → `python stage_audit.py`
  (orchestrator emits all seven gate tables) → read `review/`.
- **Add an engine:** implement the OD contract, register its coefficients
  (`gen_engine_registry.py`); no checker changes.
- **Add a gate:** read the contract + a reference control, emit RAG + verdict, register it in
  `model_controls.yml`; `stage_audit.py` picks it up.
- **Add a market (or an ABM):** emit labeled OD for the new market; the inventory gate (4) and
  the supply gates absorb it unchanged.

---

## 9. Roadmap

- **P0 (from the critic):** split construction vs validation gates in the emitted tables; get a
  survey TLFD as external shape ground truth; relabel C0→C1 as representational flexibility.
- **P1:** engine-neutral trip-length anchor; null-spread baseline from ITRE's by-facility fit;
  per-period VMT/VHT/peak-speed (done).
- **Next paradigm:** the tour→trip and tour-mode-consistency agents for a CT-RAMP-style ABM,
  proving the contract on a genuinely different demand model.
- **Deployment targets:** SCAG, ARC, NVTA, MAG — same gate stack, different reference agents,
  demand checked *before* it reaches a DOT assignment.
