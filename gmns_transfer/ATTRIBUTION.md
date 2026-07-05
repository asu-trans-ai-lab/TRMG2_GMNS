# Attribution & provenance

This folder is an **ASU companion branch** of the official Triangle Regional
Model, Generation 2 (TRMG2):

- Upstream: https://github.com/Triangle-Modeling-and-Analytics/TRMG2
- Copyright (c) 2022 Institute for Transportation Research and Education (ITRE),
  released under the MIT License (see `../LICENSE`, which applies to and is
  retained by this branch).

## What this folder is — and is not

**Is:** a GMNS interoperability *transfer* — an open-format companion that
expresses the TRMG2 travel model (network, demand steps, assignment) in GMNS/CSV
form, with a tensor computational-graph formulation for research on calibration
and computational efficiency. All model coefficients, parameter tables, and
behavioral structures are **TRMG2's own**, used verbatim from `../master/`; this
branch reproduces, it does not re-estimate or invent.

**Is not:** part of the official TRMG2 model, and not endorsed by ITRE or the
Triangle MPOs. Official releases and documentation are at the upstream links
above. Any deviation of this reproduction from official model results is a
property of this branch, not of TRMG2.

## Respectful-use notes
- Fidelity gaps of the reproduction are documented openly (`doc/FIDELITY_STATUS.md`)
  and framed as *our* reproduction status, never as defects of the original model.
- The raw household survey used to estimate TRMG2 is private to ITRE and is
  neither included nor needed here; only published/derived tables are used.
- Third-party publications about TRMG2 are not redistributed in this branch.

## Contact
ASU branch: Xuesong (Simon) Zhou, Arizona State University.
Upstream model questions belong with ITRE / Triangle Modeling & Analytics.
