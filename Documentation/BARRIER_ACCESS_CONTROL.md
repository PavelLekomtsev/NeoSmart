# Barrier Access Control System

## Overview

The Barrier Access Control system is an entry-only automated checkpoint for the parking lot. It combines **two cameras**, a **license-plate whitelist**, and a **state machine** that drives an animated barrier in Unreal Engine 5. A car that matches a whitelisted plate is granted entry automatically; the barrier opens, waits for the car to pass through a separately-monitored safety area, then closes.

There is no exit barrier — cars leave the lot without any automated interaction. Exit is handled by the parking-session bookkeeping (duration, billing hooks) rather than a physical gate.

---

## Why Two Cameras?

A single camera cannot reliably do both plate recognition and safety monitoring. The constraints are in direct conflict:

- **Plate recognition** needs a camera close to the vehicle, slightly off-axis, with a tight field of view so the license plate occupies enough pixels to be readable by the OCR model.
- **Safety monitoring** needs a wide-angle view of the entire area under and immediately past the barrier arm so a car is never mistaken as "clear" when it's partially under the arm.

We split the two concerns onto two dedicated cameras:

- **Plate camera (camera5)** — mounted before the barrier on the approach side. Sees cars approaching, does license-plate OCR, and decides whether to open. Defines two zones:
  - **Approach zone** — a larger polygon. A car entering here triggers the state machine out of idle.
  - **Reading zone** — a smaller polygon closer to the barrier, where the plate is large enough to OCR reliably.
- **Safety camera (camera6)** — mounted past the barrier looking down at the drive-through area. Defines one zone:
  - **Safety zone** — the polygon under and immediately past the barrier arm. The barrier must never close while a vehicle intersects this zone.

The two cameras share a single controller object. The plate camera drives the *forward* part of the cycle (detect car → read plate → decide → open). The safety camera drives the *backward* part (watch for the car to clear → close).

---

## The State Machine

At any given moment the barrier is in exactly one of eight states:

1. **IDLE** — default, waiting for a car.
2. **CAR_APPROACHING** — a car is in the approach zone; we're watching for it to reach the reading zone.
3. **READING_PLATE** — a car is in the reading zone; OCR runs every frame.
4. **ACCESS_GRANTED** — a plate was recognized and is in the whitelist. One-tick transition into opening.
5. **ACCESS_DENIED** — the recognized plate is not whitelisted (or consensus concluded "denied"). The system displays the denial for five seconds, then returns to idle.
6. **BARRIER_OPENING** — the `"open"` command has been issued to Unreal Engine 5; we wait for the animation-complete acknowledgment.
7. **CAR_PASSING** — the barrier is fully open. The safety camera watches for the car to clear.
8. **BARRIER_CLOSING** — the `"close"` command has been issued; we wait for its acknowledgment.

### Transitions

A successful entry walks through the states linearly: `IDLE → CAR_APPROACHING → READING_PLATE → ACCESS_GRANTED → BARRIER_OPENING → CAR_PASSING → BARRIER_CLOSING → IDLE`.

A denial diverges: `IDLE → CAR_APPROACHING → READING_PLATE → ACCESS_DENIED → IDLE`.

Manual override from the dashboard short-circuits straight to `BARRIER_OPENING` or `BARRIER_CLOSING`, skipping recognition.

---

## Plate Recognition Pipeline

Plate recognition is a two-stage process loaded once at application startup:

1. **Plate detection** — a YOLO model (trained specifically for license plates) runs on the full frame and returns bounding boxes for any plates it finds. We bias the pipeline toward high recall (low confidence threshold) because a missed detection on one frame is harmless — we'll try again next frame — but a false negative during the brief reading window would force the driver to back up.
2. **OCR** — each detected plate is cropped and fed to a ParseQ transformer-based scene-text recognizer. ParseQ returns a string and a per-character confidence vector.

### Frame Enhancement

Unreal Engine sometimes produces washed-out frames, especially in overcast or twilight lighting. A white plate with black letters loses a surprising amount of readable contrast when the whole frame's dynamic range compresses.

Before detection we apply a simple HSV boost: multiply saturation and value (brightness) by fixed factors. This widens the dynamic range of the image before anything touches it. The important insight is that the **same enhanced frame** is displayed on the "AI Detection" canvas in the dashboard — what the operator sees is exactly what the AI sees, so debugging is intuitive.

### Match-Training Augmentation

The upstream plate-detector model was trained with a ToGray + CLAHE preprocessing step. We replicate that preprocessing at inference time. Matching train-time and inference-time preprocessing reduces the domain shift between the training distribution and UE5's synthetic frames, which measurably improves recall on borderline plates.

### OCR Preprocessing for the Cropped Plate

Before the cropped plate goes into ParseQ we run CLAHE with a slightly aggressive clip limit, a min-max normalization, and a light unsharp mask. These push the characters' edges apart so ParseQ's attention has cleaner features to latch onto. The trade-off is mild noise amplification in the background, which is acceptable because ParseQ ignores non-character regions.

### Eager Model Loading

Both YOLO and ParseQ are loaded and warmed up (one dummy inference each) at application startup. Without warmup the first real plate pays a ~10-second cold-start penalty, which is far too long for a driver waiting at a barrier. After warmup, steady-state per-frame latency is well under a second on modest hardware.

---

## PlateScanner Integration

[![PlateScanner](https://img.shields.io/badge/upstream-encore--ecosystem%2FPlateScanner-181717?logo=github&logoColor=white)](https://github.com/encore-ecosystem/PlateScanner)
[![Submodule](https://img.shields.io/badge/integration-git%20submodule-2088ff?logo=git&logoColor=white)](https://github.com/encore-ecosystem/PlateScanner)
[![YOLO](https://img.shields.io/badge/detection-Ultralytics%20YOLO-00ffff?logo=python&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![ParseQ](https://img.shields.io/badge/OCR-ParseQ-ff6f00?logo=pytorch&logoColor=white)](https://github.com/baudm/parseq)

License-plate detection and OCR are not built from scratch. We integrate the open-source [**PlateScanner**](https://github.com/encore-ecosystem/PlateScanner) project — a complete two-stage recognizer that pairs a YOLO plate detector with a [ParseQ](https://github.com/baudm/parseq) scene-text transformer fine-tuned for license plates. Using PlateScanner instead of training our own gave us a working recognizer in days rather than months and let the project focus on the parts that *are* unique to NeoSmart: zone geometry, the access state machine, the UE5 handshake, and Russian-plate-aware post-processing.

> **Repository:** [`github.com/encore-ecosystem/PlateScanner`](https://github.com/encore-ecosystem/PlateScanner)
> **Role in NeoSmart:** plate-detector weights + ParseQ recognition pipeline.
> **Integration:** vendored as a git submodule under [`SmartParking/PlateScanner/`](../SmartParking/PlateScanner/).

### How the Integration Is Wired

The integration lives in a single wrapper file, [`SmartParking/web_app/plate_scanner.py`](../SmartParking/web_app/plate_scanner.py), which exposes a `PlateRecognizer` class to the rest of the application. Everything the rest of NeoSmart needs from PlateScanner — load the model, detect a plate in a frame, OCR the crop, normalize the text — happens behind that single class.

```text
                  ┌───────────────────────────────────────────────┐
                  │           BarrierController                   │
                  │  (state machine: IDLE → READING_PLATE → ...)  │
                  └───────────────────┬───────────────────────────┘
                                      │
                                      ▼
                ┌───────────────────────────────────────────┐
                │         PlateRecognizer (wrapper)         │
                │  enhance → detect → crop → OCR → normalize│
                └───────┬─────────────────────────┬─────────┘
                        │                         │
                        ▼                         ▼
   ┌──────────────────────────────┐   ┌──────────────────────────────┐
   │  PlateScanner YOLO weights   │   │     ParseQ via torch.hub     │
   │  (Models/plate_scanner/*.pt) │   │   (baudm/parseq, pretrained) │
   └──────────────────────────────┘   └──────────────────────────────┘
```

### What We Take from Upstream

| Component | Source | How we use it |
|---|---|---|
| Plate-detector weights | PlateScanner release artefact | Loaded directly into Ultralytics YOLO |
| OCR architecture | ParseQ via `torch.hub.load("baudm/parseq", ...)` | Recognition stage |
| Pre-OCR preprocessing recipe | PlateScanner training pipeline | Adapted (CLAHE clip pushed harder) |
| Train-time augmentation | ToGray + CLAHE | Replicated at inference for distribution match |

### What We Built on Top

- **Frame enhancement** — HSV brightness + saturation boost before detection, applied to the same image the operator sees on the dashboard. UE5's renders are sometimes washed out; the boost widens dynamic range so YOLO and ParseQ have crisper contrast to work with.
- **Eager warmup** — one dummy inference per model at application startup, so the first real driver doesn't pay a ten-second cold-start while parseq's torch.hub graph compiles.
- **Russian-plate post-processing** — position-aware digit↔letter swaps and a strict full-plate regex (covered in the next section). PlateScanner returns generic OCR strings; turning them into valid Russian plates is our job.
- **Access policy and consensus buffering** — multi-frame voting with the asymmetric grant/deny rule lives entirely in `BarrierController`, not in the wrapper.

### Why a Local Port of the OCR Pipeline

The wrapper does not import PlateScanner's `platescanner.utils` package directly. Upstream's `__init__` chain transitively pulls in [`cvtk`](https://pypi.org/project/cvtk/) — a separate computer-vision toolkit that we don't need and that adds non-trivial install friction on Windows. Rather than fight the dependency, we vendor a small **local port** of two upstream pieces:

- `_preprocess_license_plate` — gray + CLAHE + min-max stretch + light unsharp mask.
- `_LocalRecognitionModel` — a thin wrapper around `torch.hub` that loads ParseQ exactly as PlateScanner does, then runs the same decode + tokenizer logic.

This keeps the *behavior* identical to upstream while the *dependency surface* stays minimal. The vendored block is clearly marked in the source so anyone reading it knows where the lineage came from.

### Submodule Hygiene

PlateScanner is pinned as a git submodule rather than copy-pasted. Two practical benefits:

1. **Upstream upgrades are explicit.** Pulling in a new PlateScanner release is `git submodule update --remote` followed by a re-test — the change shows up as a single commit, not a sprawling diff.
2. **Provenance is preserved.** Anyone reading the codebase can trace exactly which upstream commit our integration is built against.

Model weights are *not* in the submodule (they're hosted separately on Yandex Disk per the upstream README). They live under [`Models/plate_scanner/`](../Models/plate_scanner/), gitignored, and the wrapper's loader auto-discovers any `.pt` file dropped into that directory. First-time setup is one download and one drop.

---

## Russian Plate Normalization

Russian private-car plates follow a strict format: `L D D D L L D D (D)`, where `L` is a letter and `D` a digit. Letters are restricted to the twelve characters that look identical in Cyrillic and Latin (A, B, E, K, M, H, O, P, C, T, Y, X). Region codes are two or three digits.

Generic OCR models like ParseQ know nothing about this structure. They happily return readings like `M214OA152` (correct), but also `M2I4OA152` (one digit misread as a letter) or `M214OA15Z` (last digit as a letter). Without post-processing, these almost-correct readings would be treated as distinct plates and never match the whitelist.

### Position-Aware Character Swapping

Each character position in a Russian plate is either a letter slot or a digit slot. If the OCR returns a digit in a letter slot — or vice versa — we remap using a visual-similarity table:

- Digits that look like letters: `0→O`, `1→I`, `2→Z`, `3→E`, `4→A`, `5→S`, `6→G`, `7→T`, `8→B`.
- Letters that look like digits: `O→0`, `I→1`, `Z→2`, `E→3`, `A→4`, `S→5`, `G→6`, `T→7`, `B→8`, and similar confusable pairs.

After this pass, any Latin letter in a letter slot that isn't part of the Russian-plate overlap set is remapped to its nearest visual cousin — `F→E`, `R→B`, `N→H`, and so on.

### Full-Plate Validation

After normalization we run a final regex check: the result must match `^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}$`. If it doesn't, the reading is dropped entirely — the OCR output wasn't a plausible plate.

This single check turned out to be surprisingly important. Partial reads like `M214OA` (trailing digits cut off) used to leak into the consensus buffer and, worst case, could be logged as DENIED for a car whose full plate was on the whitelist. Forcing every reading to be a full 8- or 9-character plate before it enters the pipeline eliminates this class of error.

---

## Access Decision: Asymmetric Consensus

OCR is noisy. A single reading is unreliable. The natural response is to take many readings and vote — and that is what we do, with a twist.

### The Symmetric Baseline

The straightforward policy would be: collect readings every frame while the car is in the reading zone; once three readings agree on the same plate, decide. If the plate is whitelisted, open; otherwise, deny. This is **symmetric** — both decisions require the same level of evidence.

### The Asymmetric Insight

The cost of a false **grant** and a false **deny** are not the same:

- A false **grant** requires OCR to misread one real plate *exactly* as some other plate that happens to be on the whitelist. This is vanishingly unlikely — the whitelist is small and the target strings are specific.
- A false **deny** is common whenever OCR is even slightly wrong. One swapped character — `M214OA152` vs. `M2I4OA152` — is enough. A strict consensus rule will reject the wrong-spelling readings one by one until either the correct spelling accumulates three votes, or the car leaves. Meanwhile the driver sits at a closed barrier.

So we use an **asymmetric** policy:

- **Grant**: if *any single reading* matches the whitelist, open the barrier immediately.
- **Deny**: still requires three agreeing readings (or the car leaving the reading zone with at least one reading in the buffer, at which point we decide using the best reading available).

This is enabled by the fact that whitelist checks are exact-match: a single correct reading is as trustworthy as any number of correct readings. The only way to hit a false grant is an exact-target misread, which doesn't happen in practice.

The insight generalizes: in any binary decision where one side is much harder to fake than the other, it's usually right to apply asymmetric evidence thresholds.

---

## UE5 Integration: The Command Handshake

Unreal Engine 5 controls the barrier's physical animation. Python and UE5 communicate through a simple REST interface:

- **Polling** — UE5 polls `GET /api/barrier/entry` roughly once per second. The response includes the current command (`"idle"`, `"open"`, or `"close"`) and metadata.
- **Acknowledgment** — when UE5 finishes an animation, it posts to `/api/barrier/entry/ue5_ack` with `"open_complete"` or `"close_complete"`.

### The Sticky-Command Rule

An early version of the state machine advanced out of `BARRIER_OPENING` after a fixed short timeout (five seconds) even if UE5 hadn't acknowledged yet. This produced a subtle but painful bug:

> The barrier's open animation takes seven to ten seconds. After five seconds the Python state machine would move on to `CAR_PASSING`, then to `BARRIER_CLOSING` once the safety zone cleared, and issue the `"close"` command. UE5, still in the middle of opening, would receive `"close"` mid-animation and simply ignore it — the animation completes as open, the driver drives through, and the barrier stays open indefinitely because the close command was never re-issued.

The fix is the **sticky command** rule: once a command is issued, it stays set on every subsequent tick until the corresponding `ack` arrives. Only the ack advances the state machine. The timeout is now a 30-second failsafe with a warning log rather than a routine advance. Two consequences:

1. The close command is physically impossible to issue before the open animation completes, because the state machine cannot leave `BARRIER_OPENING` without the ack.
2. Every command is re-asserted on every poll cycle, so a transient packet loss or a UE5 poll that arrives a few milliseconds late still gets the current command. The overall design is now effectively a two-entry command queue — open, then close — with strict serialization.

---

## The Second-Car Edge Case

A car that arrives while the barrier is still animating its close cycle used to get stuck.

The sequence: the state machine is in `BARRIER_CLOSING` waiting for UE5's close-complete ack. During this state the controller only re-asserts the close command — it does not check for new cars. Meanwhile the second car has rolled past the approach zone and is now sitting in the reading zone. Eventually UE5 acks close, the state advances to `IDLE`, and `IDLE` used to check *only* the approach zone for new cars. The second car isn't in the approach zone anymore — it's past it — so `IDLE` sees nothing and the car sits there indefinitely.

The fix: in `IDLE`, check the reading zone *first*. If a car is already there, skip `CAR_APPROACHING` and jump straight to `READING_PLATE`. The approach-zone check remains as the secondary fallback for the normal case.

This is a small change with an outsized effect on real-world robustness. Any state that can take a while to complete (barrier cycles, denial timeouts, manual overrides) defers new-car pickup, and the reading-zone fast-path in `IDLE` unblocks that car the moment we return to idle — no matter how far along its approach it got while the controller was busy.

---

## The Data Layer

A small SQLite database holds three tables:

- **`allowed_plates`** — the whitelist. Plate number, owner name, vehicle description. This is the *persistent* part of the system. Operators add, edit, and remove plates through the dashboard; changes are immediately written and survive across restarts.
- **`access_log`** — audit trail of every recognition attempt. Time, barrier, plate, confidence, result. This is *ephemeral* — cleared on every application startup so the dashboard's "Today's Statistics" counters start from zero each run.
- **`parking_sessions`** — active and completed parking sessions. Entry time, exit time, duration. Also ephemeral, for the same reason.

### Seeding

On first run the `allowed_plates` table is seeded from a JSON file shipped with the project. On subsequent runs the seeder checks whether the table is empty and skips seeding if plates are already present — so user-added plates are never overwritten by the seed data.

### Runtime Reset

The "reset on startup" policy is deliberate: the dashboard's statistics panel is a per-run monitoring surface, not a historical record. If we accumulated the log across runs, a week of development runs would make today's activity invisible under a mountain of old events. Whitelist state, by contrast, is the operator's source of truth and must never be touched by a restart.

---

## Dashboard Surface

The dashboard splits barrier functionality across two navigation tabs:

- **Barrier Control** — live operations. Two camera streams (plate camera + safety camera), each with a raw UE5 view on the left and an AI-annotated view on the right. A state panel showing the current state, last recognized plate, OCR confidence, and access result. A manual override with open/close buttons. A Today's Statistics panel with entries, denials, and currently-inside counts.
- **Access Management** — administrative work. The full access log with time, barrier, plate, owner name, confidence, and result; and the whitelist editor, where each plate can be added, edited (owner and vehicle description), or removed.

The split keeps the operations view calm and the administrative view focused. It is too easy to make an administrative mistake when you're watching live video next to an edit button.

### Dual-Canvas Display for the Plate Camera

The plate camera shows two canvases side by side: the raw UE5 frame on the left, and the enhanced-and-annotated frame on the right. The enhancement applied to the right canvas is *exactly* the enhancement applied before detection and OCR. This makes debugging tangible — if the AI is misreading a plate, the right canvas shows the image the AI is working from, not some upstream pristine render.

---

## Summary of Key Insights

- **Two cameras for one checkpoint.** Splitting plate recognition and safety monitoring across dedicated cameras lets each camera be optimally positioned for its job.
- **Asymmetric access policy.** Instant grant on a single whitelist hit; deferred deny until consensus or the car leaves. The decision costs aren't symmetric, so the evidence thresholds shouldn't be either.
- **Domain-aware OCR post-processing.** Russian plates have rigid structure. Using that structure — position-aware character swaps, full-plate regex validation — converts almost-correct OCR into correct plate lookups.
- **Drop partial reads entirely.** A truncated plate is worse than no plate. Reject anything that isn't a full 8- or 9-character match.
- **Sticky commands, not timeouts.** Commands to UE5 stay set on every tick until the corresponding ack arrives. Timeouts exist only as 30-second failsafes.
- **Fast-path new cars in IDLE.** When the controller has been busy and returns to idle, check the reading zone before the approach zone — the car might already be past the approach.
- **Enhance once, display what you enhance.** The same processed frame goes to detection, OCR, and the operator's screen. The operator sees what the AI sees.
- **Persistent whitelist, ephemeral logs.** The policy layer outlives restarts; the monitoring layer resets with them. Reset-scope should match the role of the data, not blanket every table.
