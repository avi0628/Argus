"""Configuration dataclasses and the default mission scenario.

All physical numbers are representative values for a 3U/6U CubeSat mission;
they are configurable and are *modelled* (not measured) unless stated otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Space / ground segment
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SatelliteSpec:
    """Design elements for one sun-synchronous CubeSat."""

    name: str
    altitude_km: float
    ltan_hours: float  # local time of ascending node -> sets RAAN at epoch
    mean_anomaly_deg: float
    eccentricity: float = 0.0010
    arg_perigee_deg: float = 90.0
    inclination_deg: float | None = None  # None -> computed for a sun-synchronous orbit
    bstar: float = 2.0e-5  # SGP4 drag term [1/earth radii]


@dataclass(frozen=True)
class GroundStation:
    name: str
    lat_deg: float
    lon_deg: float
    alt_m: float = 0.0
    min_elevation_deg: float = 5.0
    band: str = "S"
    max_rate_mbps: float = 2.0  # CubeSat-class, power-limited link at best geometry
    antennas: int = 1
    region: str = "mid-latitude"


@dataclass(frozen=True)
class WatchZone:
    """Wildfire-prone / Wildland-Urban-Interface area the constellation is tasked to watch."""

    name: str
    lat_deg: float
    lon_deg: float
    radius_km: float
    wui_risk: float  # 0..1 exposure of people/assets at the wildland-urban interface
    population_density: float  # people / km^2
    fire_events_per_day: float  # Poisson rate of new fire events (fire-season adjusted)
    cloud_prob: float
    biome: str = "mixed"
    live_area_ha: float = 0.0  # > 0: zone built from a live FIRMS cluster that is burning now


DEFAULT_SATELLITES: tuple[SatelliteSpec, ...] = (
    # Three late-afternoon / evening planes: operational fire sensors cross at ~10:30 and ~13:30 local
    # (and 22:30 / 01:30), while fires peak in the late afternoon. A trade study against live TLEs of
    # VIIRS/MODIS/Sentinel-2/Landsat chose 15:30 / 18:30 / 21:30 LTAN (see README "Orbit design").
    SatelliteSpec("Argus-1", 550.0, 15.5, 0.0),
    SatelliteSpec("Argus-2", 550.0, 15.5, 180.0),
    SatelliteSpec("Argus-3", 575.0, 18.5, 90.0),
    SatelliteSpec("Argus-4", 575.0, 18.5, 270.0),
    SatelliteSpec("Argus-5", 560.0, 21.5, 45.0),
    SatelliteSpec("Argus-6", 560.0, 21.5, 225.0),
)

# Power-limited CubeSat links (~2 W RF): S-band 1.5-2 Mb/s, compact X-band 4 Mb/s at best geometry.
DEFAULT_STATIONS: tuple[GroundStation, ...] = (
    GroundStation("Svalbard", 78.23, 15.41, 500.0, 5.0, "S", 2.0, 1, "high-latitude"),
    GroundStation("Inuvik", 68.32, -133.55, 100.0, 5.0, "S", 1.5, 1, "high-latitude"),
    GroundStation("Matera", 40.65, 16.70, 540.0, 10.0, "X", 4.0, 1, "mid-latitude"),
    GroundStation("Hartebeesthoek", -25.89, 27.69, 1540.0, 10.0, "S", 2.0, 1, "mid-latitude"),
    GroundStation("Singapore", 1.35, 103.82, 20.0, 10.0, "S", 1.5, 1, "equatorial"),
)

# November scenario: southern-hemisphere / tropical burning season + Santa Ana season.
DEFAULT_ZONES: tuple[WatchZone, ...] = (
    WatchZone("Sierra Nevada foothills WUI (US-CA)", 38.9, -121.0, 220.0, 0.90, 120.0, 0.45, 0.15, "mediterranean"),
    WatchZone("Southern California WUI (US-CA)", 34.2, -118.2, 180.0, 0.95, 900.0, 0.55, 0.10, "chaparral"),
    WatchZone("Blue Mountains WUI (AU-NSW)", -33.7, 150.3, 220.0, 0.85, 300.0, 0.60, 0.30, "eucalypt forest"),
    WatchZone("Western Cape WUI (ZA)", -33.9, 18.9, 180.0, 0.80, 250.0, 0.45, 0.20, "fynbos"),
    WatchZone("Attica WUI (GR)", 38.1, 23.8, 150.0, 0.85, 700.0, 0.15, 0.30, "mediterranean"),
    WatchZone("Central Portugal WUI (PT)", 40.0, -8.0, 180.0, 0.70, 110.0, 0.15, 0.45, "pine/eucalypt"),
    WatchZone("Okanagan WUI (CA-BC)", 49.9, -119.5, 180.0, 0.75, 60.0, 0.10, 0.55, "montane conifer"),
    WatchZone("Colorado Front Range WUI (US-CO)", 40.0, -105.3, 180.0, 0.85, 350.0, 0.35, 0.25, "grass/ponderosa"),
    WatchZone("Valparaiso WUI (CL)", -33.05, -71.6, 160.0, 0.90, 500.0, 0.50, 0.25, "matorral/eucalypt"),
    WatchZone("Dandenong-Yarra WUI (AU-VIC)", -37.8, 145.4, 180.0, 0.85, 250.0, 0.45, 0.35, "eucalypt forest"),
    WatchZone("Cerrado savanna (BR)", -12.5, -47.5, 300.0, 0.30, 15.0, 0.70, 0.55, "savanna"),
    WatchZone("Rondonia deforestation arc (BR)", -10.0, -63.0, 300.0, 0.20, 8.0, 0.60, 0.65, "tropical forest"),
    WatchZone("Central Kalimantan peatlands (ID)", -2.2, 113.9, 250.0, 0.35, 20.0, 0.40, 0.70, "peat swamp"),
    WatchZone("Northern Territory savanna (AU)", -14.5, 132.3, 300.0, 0.15, 1.0, 0.90, 0.35, "tropical savanna"),
    WatchZone("Guinea savanna belt (GH/TG)", 9.5, 0.5, 320.0, 0.25, 40.0, 0.90, 0.20, "guinea savanna"),
)


@dataclass(frozen=True)
class HardwareProfile:
    """Flight-computer model used to translate MAC counts into latency / energy (estimates)."""

    name: str
    short_name: str
    active_power_w: float
    idle_power_w: float
    effective_gmacs_fp32: float  # sustained GMAC/s for small depthwise CNNs (not peak)
    effective_gmacs_int8: float
    per_inference_overhead_ms: float
    notes: str = ""


DEFAULT_HARDWARE: tuple[HardwareProfile, ...] = (
    HardwareProfile(
        "NVIDIA Jetson Nano, 5 W mode (GPU FP16, TensorRT)", "JetsonNano-5W",
        5.0, 1.3, 20.0, 20.0, 0.45,
        "Maxwell GPU has no fast INT8 path: INT8 only saves memory/bandwidth",
    ),
    HardwareProfile(
        "Raspberry Pi 4 / CM4 (4x Cortex-A72 @1.5 GHz, ONNX Runtime + XNNPACK)", "RPi4-CM4",
        5.1, 2.7, 3.2, 5.5, 0.12,
        "NEON QS8 kernels give ~1.7x over FP32",
    ),
)


# ---------------------------------------------------------------------------
# Stage configs
# ---------------------------------------------------------------------------
def _default_epoch() -> datetime:
    return datetime(2026, 11, 20, 0, 0, 0, tzinfo=timezone.utc)


@dataclass
class OrbitConfig:
    epoch_utc: datetime = field(default_factory=_default_epoch)
    duration_hours: float = 24.0
    downlink_tail_hours: float = 3.0  # passes simulated after the last capture
    time_step_s: float = 20.0
    backend: str = "auto"  # auto | skyfield | analytic
    tle_file: str | None = None
    satellites: tuple[SatelliteSpec, ...] = DEFAULT_SATELLITES
    stations: tuple[GroundStation, ...] = DEFAULT_STATIONS
    zones: tuple[WatchZone, ...] = DEFAULT_ZONES
    field_of_regard_km: float = 450.0  # off-nadir pointing reach for tasked zone captures
    background_interval_s: float = 240.0  # daytime survey capture cadence per satellite
    background_min_sun_elev_deg: float = 10.0
    zone_revisit_debounce_s: float = 1200.0
    link_efficiency: float = 0.85  # framing / coding / protocol overhead
    reference_range_km: float = 1000.0  # slant range at which the link closes at 0 dB margin
    min_pass_duration_s: float = 60.0
    background_fire_prob: float = 0.03
    background_cloud_prob: float = 0.35
    background_glint_prob: float = 0.12


@dataclass
class ModelConfig:
    tile_size: int = 64
    gsd_m: float = 20.0  # Sentinel-2 SWIR native resolution (B8A/B11/B12)
    n_train: int = 640
    n_val: int = 160
    n_test: int = 240
    p_fire: float = 0.5
    epochs: int = 24  # chosen by a sweep (14/24 epochs x 16/20 channels x 2/4 real oversampling)
    batch_size: int = 32
    learning_rate: float = 4e-3
    base_channels: int = 16
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    dice_weight: float = 1.0
    detection_threshold: float = 0.5
    min_fire_pixels: int = 3
    confidence_topk: int = 8
    use_torch: bool = True
    train_threads: int = 4
    tiles_dir: str | None = None  # optional directory of real .npy/.npz/.tif tiles to score
    use_real_data: bool = True  # add real Sentinel-2 chips (data/real_s2_chips.*) when present
    real_dataset_path: str | None = None
    real_oversample: int = 4
    real_weight: float = 1.0
    calibrate: bool = True  # temperature scaling of confidences


@dataclass
class PayloadConfig:
    scene_px: int = 2560  # full scene edge length in pixels (2560 px * 30 m = 77 km)
    bands: int = 4
    bits_per_sample: int = 12
    lossless_compression_ratio: float = 2.2  # CCSDS-123-class lossless multispectral
    roi_context_px: int = 256  # full-resolution context chip around each detection
    quicklook_px: int = 256  # low-res false-colour quicklook
    quicklook_bytes_per_px: float = 0.25  # ~2 bit/px lossy
    metadata_bytes: int = 640  # alert JSON: time, geolocation, FRP proxy, confidence, area
    tx_power_w: float = 8.0  # DC power of the downlink transmitter while keyed


@dataclass
class EdgeConfig:
    prune_amount: float = 0.3
    finetune_epochs: int = 3
    quantize: bool = True
    export_onnx: bool = True
    max_iou_drop: float = 0.03
    bench_runs: int = 40
    bench_threads: int = 1
    calibration_tiles: int = 64
    hardware: tuple[HardwareProfile, ...] = DEFAULT_HARDWARE


@dataclass
class SchedulerConfig:
    buffer_capacity_mb: float = 768.0  # downlink staging partition of onboard storage
    processing_overhead_s: float = 20.0  # readout + tiling + packaging before products are queued
    alert_ttl_hours: float = 3.0  # an alert older than this is no longer actionable
    value_tau_hours: float = 2.0
    hp_wui_threshold: float = 0.6
    # mission value model (ground truth, used for evaluation and RL reward)
    hp_value: float = 10.0
    fire_value: float = 4.0
    missed_fire_scene_factor: float = 0.4
    false_alarm_value: float = 0.1
    scene_value: float = 0.3
    drop_penalty_factor: float = 0.5
    # multi-criteria urgency heuristic weights (sum of the six urgency weights = 1)
    w_confidence: float = 0.25
    w_size: float = 0.15
    w_growth: float = 0.10
    w_new: float = 0.10
    w_risk: float = 0.25
    w_fresh: float = 0.15
    w_efficiency: float = 0.30
    scene_priority_factor: float = 0.25
    # tabular Q-learning
    rl_episodes: int = 400
    rl_alpha: float = 0.15
    rl_gamma: float = 0.90
    rl_epsilon_start: float = 0.40
    rl_epsilon_end: float = 0.02
    rl_hi_conf: float = 0.75
    rl_validation_episodes: int = 20
    # cloud-aware packaging: no-fire scenes that are mostly cloud are sent as a quicklook
    cloud_quicklook_threshold: float = 0.6
    # Argus relay: 340-byte "flash alerts" through a commercial LEO messaging constellation
    relay_enabled: bool = True
    relay_packet_bytes: int = 340
    relay_latency_s: tuple[float, float] = (20.0, 150.0)
    relay_availability: float = 0.95
    relay_retry_s: float = 300.0
    relay_min_confidence: float = 0.5
    relay_daily_cap: int = 200
    # existing operational path for comparison: FIRMS global near-real-time latency (<= 3 h)
    reference_nrt_latency_h: float = 3.0


@dataclass
class LiveConfig:
    enabled: bool = False  # drive the mission with today's NASA FIRMS fires + live TLEs
    offline: bool = False  # never touch the network; use data/cache only
    n_zones: int = 30  # live fire clusters tasked as watch zones
    coverage_fires: int = 20  # fires used for the revisit-gap analysis
    coverage_step_s: float = 60.0


@dataclass
class SimulationConfig:
    seed: int = 7
    fast: bool = False
    output_dir: str = ""
    orbit: OrbitConfig = field(default_factory=OrbitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    payload: PayloadConfig = field(default_factory=PayloadConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    live: LiveConfig = field(default_factory=LiveConfig)

    def apply_fast_mode(self) -> None:
        """Smaller datasets / fewer epochs for a ~1 minute smoke run."""
        self.fast = True
        self.orbit.time_step_s = 30.0
        self.model.n_train = 320
        self.model.n_val = 96
        self.model.n_test = 160
        self.model.epochs = 12
        self.edge.finetune_epochs = 1
        self.edge.bench_runs = 15
        self.edge.calibration_tiles = 32
        self.scheduler.rl_episodes = 150
