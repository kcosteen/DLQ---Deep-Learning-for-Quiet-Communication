# Shared webcam test set — capture protocol

**Issue #15 · Person A coordinates · Week 5 · scored by #16**

`data/webcam_testset/` is the **only accuracy number this project reports**. The
87k Kaggle images are one signer in one room, so even the leakage-safe
frame-range val split is a *dev* number (same signer, same room, same camera).
The honest question — *does this work on a stranger's hand, on a different webcam,
in a different room?* — is answered by this set and nothing else.

Target: **≥ 90% overall, no class below 85%** (`docs/PROJECT_PLAN.md`).

---

## 🚫 QUARANTINE — read this before you record anything

> **This data never enters training, and is never used for model selection.**
>
> Concretely, that means it is **never**:
> - passed to `python -m src.train` (as `--root`, `--manifest`, `--backgrounds`,
>   or anything else);
> - included in `data/split_manifest.json` or any train/val split;
> - used to pick a checkpoint, an epoch, a learning rate, a threshold, an
>   augmentation setting, or an architecture;
> - looked at per-class to decide what to tune next, before the final run;
> - scored more than necessary. Every extra peek is a little bit of manual
>   overfitting, even when no gradient is involved.
>
> It is touched by exactly one command:
>
> ```bash
> python -m src.evaluate --checkpoint checkpoints/<model>.pt \
>     --webcam data/webcam_testset --figures docs/figures
> ```
>
> **Why this rule is absolute:** the moment we tune against these images, the
> number stops measuring generalisation to new people and starts measuring how
> well we fitted three specific people — and we lose the one honest metric we
> have, permanently. There is no way to un-see a test set. If the webcam number
> disappoints, fix it with *training-side* changes (more augmentation, background
> replacement, more data) evaluated on the **dev val split**, then re-score the
> webcam set once.

### Agreement — tick your line when you have read the above

Each teammate ticks their own box in a PR (or in the issue #15 thread) before
recording. This is the "written down and agreed" half of the #15 DoD.

- [ ] **Person A — @qdoan1** — I agree the webcam test set never trains and is
      never used for model selection.
- [ ] **Person B — @<github-id>** — I agree the webcam test set never trains and
      is never used for model selection.
- [ ] **Person C — @<github-id>** — I agree the webcam test set never trains and
      is never used for model selection.

---

## Who records what

**All three of us record all 29 classes** — A–Z plus `space`, `delete`,
`nothing` (`src.data.CLASSES`, the only source of truth). Not a class each: the
whole point is three different hands, so every class needs all three people.

Fill in your ids before the first session — the id becomes the filename prefix,
and the coverage checker reports per person:

| Role | GitHub | `--person` id | Sessions |
|---|---|---|---|
| Person A · Data | @qdoan1 | `qdoan` | ≥ 2, different conditions |
| Person B · Modeling | @`<tbd>` | `<tbd>` | ≥ 2, different conditions |
| Person C · Demo & Eval | @`<tbd>` | `<tbd>` | ≥ 2, different conditions |

Person ids are lowercase letters/digits/`-`, **no underscore** (`_` separates the
fields of the filename).

**Volume:** aim for **10 images per class per person** (≈ 870 images total). The
floor the checker enforces is 1 per person per class; 10 gives per-class accuracy
that means something. One pass of all 29 classes takes about 15–20 minutes.

---

## Varied conditions — the entire point

Three people recording at the same desk, under the same lamp, against the same
wall is *not* a test set. It measures the same thing the Kaggle data does and it
will report a number we cannot trust. **Same-room, same-lighting captures defeat
the purpose.**

Each person records **at least two sessions** with a *different* `--condition`
tag, and between them covers:

- [ ] **≥ 2 lighting setups** — e.g. daylight from a window *and* a warm room
      lamp at night. Vary direction too: light in front of you vs. behind you
      (backlit is the hard case).
- [ ] **≥ 2 backgrounds** — e.g. a plain wall *and* a cluttered one (bookshelf,
      kitchen, window). At least one background that is *not* plain.
- [ ] **Different hand sizes / distances** — some shots close to the camera
      (hand fills the frame), some at arm's length. The crop is normalised to
      224², so this varies the resolution and blur the model sees.
- [ ] **Small pose variation** — rotate the wrist a few degrees between shots,
      shift the hand off-centre. Do not hold a rigid pose for ten identical
      frames; ten near-duplicates are worth about one image.
- [ ] **Whatever else differs about you** — skin tone, hand size, sleeves,
      jewellery, left- vs right-handed signing. Do not normalise these away.

Across the team, also try to differ in **camera** (laptop vs external webcam) —
sensor and lens differences are exactly the shift we want to measure.

Tag each session so the difference is recorded and checkable:

```bash
--condition desklamp-plainwall      # session 1
--condition window-daylight-kitchen # session 2
```

The coverage checker **fails** a person with fewer than two distinct condition
tags.

---

## Layout and filenames — do not improvise this

```
data/webcam_testset/
├── A/
│   ├── qdoan_desklamp-plainwall_001.jpg      <- 224x224 BGR crop
│   ├── qdoan_window-daylight-kitchen_001.jpg
│   ├── kcosteen_overhead-bookshelf_001.jpg
│   └── ...
├── B/
├── ...
├── space/
├── delete/
└── nothing/
```

**Folder per class; the person goes in the FILENAME:**

```
data/webcam_testset/<CLASS>/<person>_<condition>_<seq>.jpg
```

Concrete example: `data/webcam_testset/A/qdoan_desklamp-plainwall_001.jpg`.

### Why not a folder per person?

Because the scorer would not see it. `src/evaluate.py::_webcam_samples(root)`
loops over `CLASSES` and calls `data.list_class_files(root, c)`, which lists files
**directly** under `root/<CLASS>/` and **does not recurse**. So:

| Layout | What `--webcam` scores |
|---|---|
| `data/webcam_testset/A/qdoan_001.jpg` | ✅ scored |
| `data/webcam_testset/qdoan/A/001.jpg` | ❌ **silently 0 images** |
| `data/webcam_testset/A/qdoan/001.jpg` | ❌ **silently 0 images** |

The failure is silent: the eval still prints an accuracy, just over a subset (or
over nothing). `scripts/check_webcam_coverage.py` treats any images in a shadowed
directory as a hard error for this reason.

The 29 folder names must match `src.data.CLASSES` exactly — including
`delete` (**not** Kaggle's `del`), `space` and `nothing`.

---

## How to record

Everything goes through `src.crop.HandCropper`, the frozen preprocessing contract
(`docs/PREPROCESSING_CONTRACT.md`). **Do not** drop in phone photos, screenshots,
or frames cropped by hand: test framing must equal training framing, or the number
measures our cropping, not the model. The helper does this for you:

```bash
# one session = one condition; walk all 29 classes
python scripts/record_webcam_testset.py \
    --person qdoan --all-classes --count 10 \
    --condition desklamp-plainwall

# or just re-do a few classes
python scripts/record_webcam_testset.py \
    --person qdoan --class M N S T --count 10 \
    --condition window-daylight-kitchen
```

Two windows open: the camera view (with the detected hand box and a HUD) and the
**crop the model actually sees**. Keys:

| Key | Action |
|---|---|
| `SPACE` / `c` | capture the current crop |
| `a` | toggle auto-capture (one every `--auto-delay` seconds) |
| `u` | undo — delete the last capture of this session |
| `n` | next class |
| `q` / `ESC` | quit |

What the script guarantees:

- What lands on disk is exactly `HandCropper.crop(frame).crop_bgr` — a 224² BGR
  crop, identical in framing to `src.cache_crops` (training) and
  `app/webcam_speller.py` (serving).
- The frame is **mirrored before cropping**, because `app/webcam_speller.py`
  mirrors before cropping. (`--no-mirror` exists, but only use it if the app
  changes.) Keeping these the same is what makes the benchmark predict demo-day.
- **No-hand frames are refused.** If MediaPipe finds no hand the crop is a centre
  square, which is a picture of your room, not of a sign. The HUD turns red and
  nothing is written. If it will not detect your hand: more light, plainer
  background, hand more fully in frame.
- **`nothing` inverts that rule** — it is the idle gesture, so keep your hands out
  of frame and the empty-frame crop is the correct sample. (Same convention the
  training cache records for `nothing`.)
- Sequence numbers continue from what is already in the folder, so a second
  session appends instead of overwriting.

### Recording tips

- Sign the letter, then move slightly between shots (rotate, shift, change
  distance). Ten frames of a frozen hand ≈ one image of information.
- Do the confusable groups deliberately and carefully: **M/N/S/T**, **A/E/S**,
  **K/V**, **X**, **I/Y** — the measured failures in `docs/RESULTS.md`. Sloppy
  captures here produce a scary number that is really a labelling problem.
- If a take is bad, press `u` immediately; deleting files afterwards is fine too.

---

## Verify coverage

```bash
python scripts/check_webcam_coverage.py --root data/webcam_testset
```

It prints a (class × person) count matrix and checks the two #15 DoD conditions:

- **(a) Coverage** — all 29 classes, from all three people, each with ≥ 2 distinct
  condition tags.
- **(b) Layout** — the files are where `src.evaluate --webcam` looks. It walks with
  `data.list_class_files` (the scorer's own function), cross-checks the count
  against `evaluate._webcam_samples` when torch is installed, flags any shadowed
  per-person directory, and verifies every image is 224×224 (i.e. came through
  `HandCropper`).

**Exit code 0 only when both hold**, so it can gate CI later. Run it after every
session; run it before opening the PR that closes #15.

```bash
python scripts/check_webcam_coverage.py --people qdoan kcosteen <third-id>
```

Passing `--people` pins the expected roster, so a typo'd `--person` shows up as a
missing teammate instead of quietly looking like a fourth contributor.

---

## Storage and git

`data/webcam_testset/.gitignore` ignores everything except `.gitkeep`,
`.gitignore`, `README.md` and `samples/`. **Do not commit the captures** — keep
them in the team's shared drive and each of us syncs a local copy. Nothing under
`data/webcam_testset/<CLASS>/` should ever appear in a PR diff.

Do **not** drop synthetic or placeholder images into a class folder to "test the
pipeline" (unlike `data/backgrounds/`, which does ship tiny samples). Everything
under a class folder is scored as the reported benchmark; a fake image there is a
corrupted metric. Use `--out /tmp/webcam_smoke` if you want to try the recorder.

---

## Definition of Done (#15)

| # | Condition | Status |
|---|---|---|
| 1 | Protocol written down (this file) | ✅ done |
| 2 | Recorder that enforces the crop contract + layout | ✅ `scripts/record_webcam_testset.py` |
| 3 | Coverage/quarantine checker, non-zero exit when incomplete | ✅ `scripts/check_webcam_coverage.py` |
| 4 | "Never trains on this" written down | ✅ this file, `README.md`, `CONTRIBUTING.md`, `data/webcam_testset/README.md` |
| 5 | "Never trains on this" **agreed** | ⬜ three ticks in the checklist above — **humans** |
| 6 | 29 classes × 3 people × ≥ 2 conditions, on disk | ⬜ **humans must record it** — verify with the checker |

Rows 5 and 6 cannot be produced by tooling: they need three people in three rooms
with three webcams. Everything else is in place so that those two are the only
work left, and row 6 is machine-checkable the moment the captures exist.

Once the checker passes, #16 scores it — and that number is the one that goes in
the report.
