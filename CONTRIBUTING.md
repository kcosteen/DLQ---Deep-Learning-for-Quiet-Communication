# Contributing & team workflow

3-person team, ~8 weeks. Work happens on this shared repo via pull requests.

## Roles

| Person | Area | Primary files |
|---|---|---|
| **Person A** | Data & Augmentation | `src/data.py`, `src/augment.py`, `notebooks/`, `tests/` |
| **Person B** | Modeling & Training | `src/model.py`, `src/train.py`, `src/export.py` |
| **Person C** | Demo & Evaluation | `src/crop.py`, `src/evaluate.py`, `app/`, `docs/` |

Issues are labelled `person-A` / `person-B` / `person-C` and attached to weekly
milestones. Put your GitHub username on the issues assigned to you.

## The two contracts everyone depends on — lock them in Week 1

1. **Crop + preprocessing contract (`src/crop.py`).** The MediaPipe hand-crop and
   the resize/normalize must be *identical* in training and in the app. All of it
   lives in `src/crop.py` and nowhere else. Changing a constant there changes both
   sides at once — on purpose. **Guarded by `tests/test_preprocessing_parity.py`.**
   Owner of the file: Person C, but a change needs sign-off from B (training) too.

2. **Leakage-safe split (`src/data.py::frame_range_split`).** Never introduce a
   random split, and never let augmented copies of a train frame cross into val.
   **Guarded by `tests/test_split_leakage.py`.** Owner: Person A.

If either test fails, the PR does not merge. These are the project's integrity
guards, not optional niceties.

## Sync points

- **Twice-weekly 30-minute syncs.**
- **Shared W&B project** (`asl-fingerspelling`) — everyone logs runs there so we
  compare on the same axes.
- **Shared webcam test set** (`data/webcam_testset/`) — coordinated by Person A;
  all three record every sign in varied conditions (Week 5). **This data never
  enters training and is never used for model selection.** Capture protocol,
  layout, and the quarantine agreement:
  [`docs/WEBCAM_TESTSET_PROTOCOL.md`](docs/WEBCAM_TESTSET_PROTOCOL.md) (#15).
  Record with `python scripts/record_webcam_testset.py --help`; verify your
  coverage with `python scripts/check_webcam_coverage.py` before opening a PR
  that adds captures.

## Pull-request checklist

- [ ] `pytest` passes locally (at least the split-leakage + parity tests).
- [ ] New CLIs have `--help` (argparse) and a docstring.
- [ ] You did not add a random split or diverge the preprocessing contract.
- [ ] If you report an accuracy number, it is the **leakage-safe** or **webcam**
      number — never the naïve random-split number — and you said which.
- [ ] Big data / checkpoints are gitignored, not committed.

## Honest-metrics rule

The only accuracy we *report* is the webcam-test-set number. Dev/val numbers are
fine internally but must be labelled as such. Do not put a random-split number in
any report, slide, or README.

That number is only honest while `data/webcam_testset/` stays quarantined: it
never trains, and it never picks a checkpoint, epoch, threshold, or augmentation
setting. Tune on the dev val split, then score the webcam set once — see the
[quarantine rule](docs/WEBCAM_TESTSET_PROTOCOL.md#-quarantine--read-this-before-you-record-anything).
