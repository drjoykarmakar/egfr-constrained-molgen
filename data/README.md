# Data provenance

Project 3 does not issue a new ChEMBL query for EGFR training data.

`scripts/prepare_data.py` reads three artifacts from a local checkout of
Project 2 (`egfr-virtual-screen`, release `v1.0.0`):

- `data/processed/egfr_activity_classification.csv`
- `results/tables/scaffold_split_assignments.csv`
- `data/raw/egfr_chembl37_activities.manifest.json`

It freezes three Project 3 files locally:

- scaffold-training actives for generator training,
- scaffold-test actives for held-out comparison only,
- the complete Project 2 benchmark SMILES for exact-match rejection.

A JSON manifest records source paths, Project 2 release tag, ChEMBL release/date,
activity cutoff, counts, and SHA-256 checksums. Large CSV snapshots are ignored
by git; the manifest is intended to be committed.
