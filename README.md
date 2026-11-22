# Argus: onboard-AI wildfire early warning from CubeSats

> *Argus Panoptes, the hundred-eyed watchman who never slept.*

**Track:** SDG 9, Industry, Innovation & Infrastructure. Co-benefits: SDG 13 Climate Action, SDG 11 Sustainable Cities (wildland-urban interface, WUI), and SDG 15 Life on Land.

**The problem.** Today's fire satellites (VIIRS, MODIS, Sentinel-2, Landsat) pass over a fire only a few times a day, mostly around **10:30 and 13:30 local time**. Their detections then reach users through ground processing, globally within about **3 h** (FIRMS near-real-time). Fires peak in the late afternoon. Using live data, Argus found that **26 of 27 afternoon VIIRS fires were not yet burning in the same morning's Sentinel-2 image**. Those fires ignite between today's overpasses.

**The idea.** Argus is a constellation of small, low-cost CubeSats in **late-afternoon and evening orbits** (15:30, 18:30 and 21:30 local time). Each one:
- detects fire **onboard** with a 20k-parameter neural network,
- turns a 39 MB image into a **196 kB alert**,
- sends a **340-byte flash alert through a commercial messaging constellation** within minutes,
- and prioritises its limited ground-station downlink by risk to people.

## Run it

```
python argus_cli.py run --live            # mission driven by today's NASA FIRMS fires + live TLEs (~1-2 min)
python argus_cli.py run --live --offline  # same, from data/cache only
python argus_cli.py live                  # today's fire picture: strongest fires, fires threatening people
python argus_cli.py build-dataset         # (re)build the real Sentinel-2 training set (~10 min, ~150 MB cached)
python argus_cli.py dashboard             # Streamlit app
python argus_cli.py --fast                # quick smoke run on the built-in fire-season scenario
python argus_cli.py run --live --deterministic   # bit-reproducible numbers
```

You can run it from any folder. Only **NumPy** is required. Every other dependency has a fallback, and no API keys are needed:

| Missing | Fallback |
|---|---|
| `torch` | NumPy per-pixel spectral MLP (pruning + INT8 + ONNX via `onnx.helper`) |
| `skyfield` | Analytic J2 propagator, including for real satellites parsed from TLE text |
| `onnx` / `onnxruntime` | Export skipped / parity checked with `onnx.reference` |
| network | Most recent copy in `data/cache` |
| `streamlit` | CLI only |

`ARGUS_DISABLE=torch,skyfield` forces the fallbacks.

## Live data (all public, no keys)

| Source | Used for |
|---|---|
| NASA FIRMS VIIRS NOAA-20/21 + MODIS, 24 h and 7-day | Live fire clusters to task; training-set pairing; independent validation |
| Element84 Earth Search STAC + AWS `sentinel-cogs` | Real Sentinel-2 L2A imagery (B04, B8A, B11, B12, SCL) |
| CelesTrak GP data (resource + weather) | Live TLEs of NOAA-20/21, Suomi NPP, Terra, Aqua, Sentinel-2A/B/C, Landsat 8/9 |
| GeoNames cities15000 | Population exposure, giving WUI risk per fire |

`argus/cog.py` is a dependency-free cloud-optimized GeoTIFF reader. It uses HTTP range requests, TIFF/BigTIFF parsing, DEFLATE with predictor, and UTM projection. A 64×64-pixel chip costs a few hundred kB instead of a 100 MB granule. Its georeferencing is verified independently: FIRMS hotspots land 1–10 px (20–200 m) from the fire pixels it extracts.

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| 0 Live | `live.py`, `sentinel2.py`, `cog.py` | FIRMS hotspots are clustered into fire events, with population exposure. The strongest and most-exposed fires are tasked. The real Sentinel-2 chip dataset is built here. |
| 1 Orbit | `orbit.py`, `coverage.py` | Argus-6 (3 planes × 2), propagated with SGP4. Five ground stations with elevation masks and an adaptive-rate link budget; antenna/transmitter contention. Captures day and night. Revisit-gap analysis against real fire satellites, plus a constellation-scaling study. |
| 2 Model | `segmentation.py`, `models_torch.py`, `fire_index.py` | **ArgusNet**: depthwise-separable U-Net with squeeze-excitation attention (19.7k params). Inputs are 4 bands, NBR, the SWIR difference, log(B12/B11) and log(B12/B8A). Loss is focal plus soft Dice, with an ignore band for uncertain pseudo-label pixels. Confidence is temperature-calibrated. It is trained on synthetic tiles **plus real Sentinel-2 chips**, and benchmarked against a **Murphy-2016-style physics threshold** algorithm. |
| 3 Edge | `edge.py` | L1 pruning + fine-tune, INT8 static post-training quantization (the attention blocks quantize too), and ONNX export with a parity check. Jetson Nano (5 W) and Raspberry Pi 4 latency/energy are modelled. |
| 4 Downlink | `scheduler.py` | Per-satellite buffer and adaptive-rate pass slots. Compares FIFO, a multi-criteria urgency heuristic and Q-learning. Adds **relay flash alerts** and **cloud-aware quicklooks**. |

## Method notes

* **Real training data without hand labelling.** `build-dataset` pairs each FIRMS "fire-day" with Sentinel-2 scenes from the same day. A second pass is seeded by MODIS-Terra, which crosses at the same local time as Sentinel-2. The physics test labels each chip, with a 2-pixel ignore band. The train/val/test split is by fire location, so no fire event appears in both training and test.
* **Fair real-image metrics.** On real chips the physics test is the label source, so its IoU there is 1 by construction. The independent checks against NASA FIRMS are:
  * recall of hotspots that were already burning at the Sentinel-2 pass;
  * the share of detections FIRMS confirms (within ±12 h and 400 m);
  * false alarms on chips where FIRMS saw nothing within ±24 h.
* **Orbit design.** A trade study against live TLEs chose 15:30 / 18:30 / 21:30 orbits over morning ones, which duplicate Terra, Sentinel-2 and Landsat.
* **Mission value.** The scheduler's score uses ground truth, and the reinforcement-learning reward is built the same way:
  * a high-priority (WUI) fire alert is worth 10, decaying with latency, with a 3 h alert lifetime;
  * any other fire alert is worth 4;
  * archive scenes are worth 0.3, scaled by how cloud-free they are.

## Results

These are from `python argus_cli.py run --live --deterministic`, run on 22 Nov 2026 at 11:00 UTC with that day's live data:
- 130,026 FIRMS hotspots grouped into 19,798 fire clusters;
- 30 fires tasked, 14 of them near people;
- 905 captures;
- 127 real Sentinel-2 chips from 67 grid tiles and 69 fire locations, 44 with active fire.

**1. Detector: ArgusNet vs. the classical physics threshold** (synthetic test set with exact labels)

| | ArgusNet | Physics baseline |
|---|---|---|
| Pixel IoU | **0.82** | 0.65 |
| Pixel precision | **0.96** | 0.70 |
| Tile false-alarm rate | **0.0%** | 5.2% (glint 9.5%, industry 13%, cloud 8.5%) |
| Burn-area error | **21%** | 90% |

**2. Real Sentinel-2 imagery** (31 held-out chips; these fire events are never seen in training)
- IoU 0.79 against the physics labels.
- It found 4 of 4 FIRMS hotspots that were already burning at the Sentinel-2 pass.
- 57% of its detections are FIRMS-confirmed, the same as the baseline.
- **Finding:** 26 of 27 afternoon VIIRS fires were not burning at that morning's Sentinel-2 pass.

**3. Edge**
- Pruned + INT8 ArgusNet: **33 kB** stored (FP32 is 86 kB), IoU 0.83.
- ONNX output matches PyTorch to 1e-5.
- About 11 J per scene on a Jetson Nano, versus 2,187 J to downlink the compressed scene.

**4. Downlink** (data is 1.8× the downlink capacity)

| | FIFO | Urgency heuristic | Q-learning |
|---|---|---|---|
| WUI alerts delivered within 3 h | 10/60 | **60/60** | **60/60** |
| Mean WUI alert latency | 399 min | **31 min** | 33 min |
| First WUI notification with relay | **2.0 min** | **2.0 min** | **2.0 min** |
| Mission value | 186 | **890** | 871 |

Q-learning learns: on held-out episodes it scores 181 before training and 204 after. It still trails the heuristic (210).

**5. Coverage of today's fires** (live TLEs of 10 operational fire satellites)

| | Mean worst revisit gap | Mean time to first alert |
|---|---|---|
| Today's satellites | 6.1 h | 4.4 h (FIRMS near-real-time) |
| + Argus-6 | **4.9 h** | **2.6 h** |
| + Argus-12 | 3.7 h | 1.9 h |
| + Argus-24 | 3.6 h | 1.8 h |

Most of the gain comes by 12 satellites, so Argus-12 is the operational target.

## Honest limitations

* **Small real test set.** The real held-out set is small (about 30 chips from about 17 scenes). The physics pseudo-labels miss very weak fire pixels, which pulls the model toward the baseline on real data.
* **Modelled hardware numbers.** Jetson Nano and Raspberry Pi 4 figures come from MAC counts and published-class throughput. Host latency, model sizes and IoU are measured.
* **Assumed relay performance.** Relay availability (95%) and latency (20–150 s) are assumptions for an Iridium-SBD-class service. Link rates are those of a power-limited CubeSat.
* **Simplified coverage counting.** A fire counts as observed when it is inside a sensor's swath or Argus's ±450 km pointing reach. Cloud cover isn't modelled in the coverage study.
* **Live numbers change daily.** FIRMS updates every few hours. `--offline` reruns the cached snapshot.
