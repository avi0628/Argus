"""Stage 4 - bandwidth-aware downlink prioritization.

Each satellite keeps a finite onboard staging buffer. During every ground pass a
policy decides which queued product to transmit next, subject to the pass's
time-varying (ACM) slot capacity. Three policies are compared:

1. FIFO baseline           - chronological capture order, tail-drop when the buffer is full
2. Urgency heuristic       - multi-criteria score (confidence, fire size, growth, new ignition,
                             WUI/population risk, freshness, payload-size efficiency), lowest-score eviction
3. Q-learning RL scheduler - tabular Q-learning over (queue-class mask, buffer occupancy, pass phase)
                             choosing which product class to serve; trained on domain-randomised
                             episodes, reward = ground-truth mission value of delivered products
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np

from .config import SchedulerConfig
from .orbit import GroundPass

ALERT = "ALERT"
SCENE = "SCENE"
Logger = Callable[[str], None]


@dataclass(frozen=True)
class DownlinkProduct:
    product_id: int
    capture_id: int
    sat_name: str
    capture_time_s: float
    available_time_s: float
    kind: str  # ALERT (metadata-assisted fire alert) | SCENE (lossless compressed scene)
    size_bytes: int
    confidence: float
    est_area_ha: float
    growth_ha_per_h: float
    is_new_detection: bool
    wui_risk: float
    population_density: float
    zone_name: str | None
    cloud_fraction: float = 0.0  # onboard spectral cloud estimate
    quicklook_only: bool = False  # cloudy no-fire scene packaged as a quicklook
    # ground truth - never visible to the policies
    gt_fire: bool = False
    gt_high_priority: bool = False
    fire_event_id: int | None = None
    gt_cloud: float = 0.0


@dataclass(eq=False)
class BufferItem:
    product: DownlinkProduct
    remaining_bytes: float
    admitted_s: float
    tag: int = -1
    evicted: bool = False


@dataclass
class PassContext:
    station: str
    pass_start_s: float
    pass_end_s: float
    pass_capacity_bytes: float
    capacity_remaining_bytes: float
    buffer_used_bytes: float
    buffer_capacity_bytes: float


class MissionValueModel:
    """Ground-truth value of delivering a product with a given capture-to-ground latency."""

    def __init__(self, cfg: SchedulerConfig) -> None:
        self.cfg = cfg
        self.ttl_s = cfg.alert_ttl_hours * 3600.0
        self.tau_s = cfg.value_tau_hours * 3600.0

    def base_value(self, p: DownlinkProduct) -> float:
        c = self.cfg
        if p.gt_fire:
            b = c.hp_value if p.gt_high_priority else c.fire_value
            return b if p.kind == ALERT else b * c.missed_fire_scene_factor
        return c.false_alarm_value if p.kind == ALERT else c.scene_value

    def delivered_value(self, p: DownlinkProduct, latency_s: float) -> float:
        b = self.base_value(p)
        if p.gt_fire:
            return b * math.exp(-latency_s / self.tau_s) if latency_s <= self.ttl_s else 0.05 * b
        if p.kind == SCENE:
            b *= 1.0 - min(0.95, p.gt_cloud)  # a cloud-covered scene carries little surface information
        return b * math.exp(-latency_s / 86400.0)

    def drop_penalty(self, p: DownlinkProduct) -> float:
        return -self.cfg.drop_penalty_factor * self.base_value(p) if p.gt_fire else 0.0


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
class DownlinkPolicy:
    name = "base"

    def reset(self) -> None:
        """Called at the start of every simulation run (must not erase learned state)."""

    def on_admit(self, item: BufferItem) -> None:
        """Hook to tag a newly admitted item (e.g. with an RL class)."""

    def select(self, buffer: list[BufferItem], now_s: float, ctx: PassContext) -> int:
        raise NotImplementedError

    def make_room(self, buffer: list[BufferItem], incoming: BufferItem, free_bytes: float,
                  now_s: float) -> list[int] | None:
        """Indices to evict so ``incoming`` fits, or None to reject it (default: tail drop)."""
        return None

    def on_delivered(self, item: BufferItem, reward: float) -> None:
        pass

    def on_dropped(self, item: BufferItem, penalty: float) -> None:
        pass

    def on_episode_end(self) -> None:
        pass


class FIFOPolicy(DownlinkPolicy):
    name = "FIFO baseline"

    def select(self, buffer: list[BufferItem], now_s: float, ctx: PassContext) -> int:
        return min(range(len(buffer)), key=lambda i: (buffer[i].product.capture_time_s, buffer[i].product.product_id))


class UrgencyHeuristicPolicy(DownlinkPolicy):
    name = "Urgency heuristic"

    def __init__(self, cfg: SchedulerConfig) -> None:
        self.cfg = cfg

    def priority(self, p: DownlinkProduct, now_s: float) -> float:
        c = self.cfg
        age_h = max(0.0, now_s - p.capture_time_s) / 3600.0
        fresh = math.exp(-age_h / c.alert_ttl_hours)
        size_term = min(1.0, math.log1p(max(p.est_area_ha, 0.0)) / math.log1p(100.0))
        growth = min(1.0, max(0.0, p.growth_ha_per_h) / 5.0)
        pop = min(1.0, math.log1p(max(p.population_density, 0.0)) / math.log1p(1000.0))
        risk = 0.7 * p.wui_risk + 0.3 * pop
        urgency = (c.w_confidence * p.confidence + c.w_size * size_term + c.w_growth * growth
                   + c.w_new * (1.0 if p.is_new_detection else 0.0) + c.w_risk * risk + c.w_fresh * fresh)
        efficiency = 1.0 / (1.0 + p.size_bytes / 1e6)  # value per byte: small alerts first
        kind = 1.0 if p.kind == ALERT else c.scene_priority_factor
        return kind * urgency * (1.0 - c.w_efficiency + c.w_efficiency * efficiency)

    def select(self, buffer: list[BufferItem], now_s: float, ctx: PassContext) -> int:
        return max(range(len(buffer)), key=lambda i: self.priority(buffer[i].product, now_s))

    def make_room(self, buffer: list[BufferItem], incoming: BufferItem, free_bytes: float,
                  now_s: float) -> list[int] | None:
        p_in = self.priority(incoming.product, now_s)
        scored = sorted(((self.priority(it.product, now_s), i) for i, it in enumerate(buffer)))
        freed, evict = free_bytes, []
        for score, i in scored:
            if score >= p_in:
                break
            evict.append(i)
            freed += buffer[i].remaining_bytes
            if freed >= incoming.remaining_bytes:
                return evict
        return None


class QLearningPolicy(DownlinkPolicy):
    """Tabular Q-learning over product classes (semi-MDP: one decision per transmitted product)."""

    name = "Q-learning RL"
    N_CLASSES = 6
    CLASS_NAMES = ("alert hi-risk hi-conf", "alert hi-risk lo-conf", "alert lo-risk hi-conf",
                   "alert lo-risk lo-conf", "scene watch-zone", "scene background")
    OCC_BINS = 4
    PHASE_BINS = 3

    def __init__(self, cfg: SchedulerConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.n_states = (1 << self.N_CLASSES) * self.OCC_BINS * self.PHASE_BINS
        self.q = np.zeros((self.n_states, self.N_CLASSES))
        self.visits = np.zeros((self.n_states, self.N_CLASSES), dtype=np.int64)
        self.training = False
        self.epsilon = cfg.rl_epsilon_start
        self.class_value = np.zeros(self.N_CLASSES)  # no hand-coded prior: learned from rewards only
        self.class_count = np.zeros(self.N_CLASSES)
        self._pending: tuple[int, int] | None = None
        self._acc = 0.0

    def classify(self, p: DownlinkProduct) -> int:
        hi_risk = p.wui_risk >= self.cfg.hp_wui_threshold
        if p.kind == ALERT:
            hi_conf = p.confidence >= self.cfg.rl_hi_conf
            if hi_risk:
                return 0 if hi_conf else 1
            return 2 if hi_conf else 3
        return 4 if (p.zone_name is not None or p.wui_risk >= 0.3) else 5

    def reset(self) -> None:
        self._pending = None
        self._acc = 0.0

    def on_admit(self, item: BufferItem) -> None:
        item.tag = self.classify(item.product)

    def _state(self, buffer: list[BufferItem], ctx: PassContext) -> tuple[int, list[int]]:
        bits = 0
        for it in buffer:
            bits |= 1 << it.tag
        occ = min(self.OCC_BINS - 1, int(self.OCC_BINS * ctx.buffer_used_bytes / max(ctx.buffer_capacity_bytes, 1.0)))
        frac = ctx.capacity_remaining_bytes / max(ctx.pass_capacity_bytes, 1.0)
        phase = 0 if frac > 2 / 3 else (1 if frac > 1 / 3 else 2)
        valid = [a for a in range(self.N_CLASSES) if (bits >> a) & 1]
        return (bits * self.OCC_BINS + occ) * self.PHASE_BINS + phase, valid

    def _update(self, next_value: float | None) -> None:
        if self._pending is None:
            return
        s0, a0 = self._pending
        target = self._acc + (self.cfg.rl_gamma * next_value if next_value is not None else 0.0)
        self.q[s0, a0] += self.cfg.rl_alpha * (target - self.q[s0, a0])
        self._pending = None
        self._acc = 0.0

    def _argmax_random_tie(self, values: np.ndarray) -> int:
        best = np.flatnonzero(values >= values.max() - 1e-12)
        return int(best[int(self.rng.integers(len(best)))])

    def select(self, buffer: list[BufferItem], now_s: float, ctx: PassContext) -> int:
        s, valid = self._state(buffer, ctx)
        qv = self.q[s, valid]
        if self.training and self.rng.random() < self.epsilon:
            a = valid[int(self.rng.integers(len(valid)))]
        elif self.visits[s, valid].sum() > 0:
            a = valid[self._argmax_random_tie(qv)]
        else:  # unvisited state: fall back to the learned per-class reward estimates
            a = valid[self._argmax_random_tie(self.effective_class_values()[valid])]
        if self.training:
            self._update(float(qv.max()))
            self._pending = (s, a)
            self.visits[s, a] += 1
        return self._pick(buffer, a, now_s)

    def _pick(self, buffer: list[BufferItem], a: int, now_s: float) -> int:
        """Within a class: freshest alert first (exponential-decay optimal); scenes by suspicion."""
        best_i, best_key = -1, (-math.inf, -math.inf)
        for i, it in enumerate(buffer):
            if it.tag != a:
                continue
            p = it.product
            key = (p.capture_time_s, 0.0) if a < 4 else (p.confidence, p.capture_time_s)
            if key > best_key:
                best_i, best_key = i, key
        return best_i

    def on_delivered(self, item: BufferItem, reward: float) -> None:
        self._acc += reward
        if self.training:
            c = item.tag
            self.class_count[c] += 1
            self.class_value[c] += (reward - self.class_value[c]) / self.class_count[c]

    def on_dropped(self, item: BufferItem, penalty: float) -> None:
        self._acc += penalty

    def on_episode_end(self) -> None:
        if self.training:
            self._update(None)
        self._pending = None
        self._acc = 0.0

    MIN_CLASS_SAMPLES = 3

    def effective_class_values(self) -> np.ndarray:
        """Learned mean reward per class; classes seen too rarely are treated optimistically
        (never evicted first) instead of being mistaken for worthless."""
        vals = self.class_value.copy()
        seen = self.class_count >= self.MIN_CLASS_SAMPLES
        if seen.any():
            vals[~seen] = float(vals[seen].max()) + 1e-3
        return vals

    def make_room(self, buffer: list[BufferItem], incoming: BufferItem, free_bytes: float,
                  now_s: float) -> list[int] | None:
        vals = self.effective_class_values()
        v_in = vals[incoming.tag]
        order = sorted(range(len(buffer)), key=lambda i: (vals[buffer[i].tag], buffer[i].product.capture_time_s))
        freed, evict = free_bytes, []
        for i in order:
            if vals[buffer[i].tag] >= v_in:
                break
            evict.append(i)
            freed += buffer[i].remaining_bytes
            if freed >= incoming.remaining_bytes:
                return evict
        return None

    def class_ranking(self) -> list[tuple[str, float]]:
        """Seen classes by learned mean reward; rarely seen classes last (kept, never evicted first)."""
        seen = self.class_count >= self.MIN_CLASS_SAMPLES
        order = sorted(range(self.N_CLASSES), key=lambda i: (not seen[i], -self.class_value[i]))
        return [(self.CLASS_NAMES[i] + ("" if seen[i] else " [unseen]"), float(self.class_value[i])) for i in order]


class RandomPolicy(DownlinkPolicy):
    """Reference: transmits a uniformly random queued product (used to validate RL learning)."""

    name = "Random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def select(self, buffer: list[BufferItem], now_s: float, ctx: PassContext) -> int:
        return int(self.rng.integers(len(buffer)))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
@dataclass
class SimulationLog:
    policy: str
    deliveries: dict[int, tuple[float, str]] = field(default_factory=dict)
    drops: dict[int, tuple[float, str]] = field(default_factory=dict)
    pending: set[int] = field(default_factory=set)
    capacity_bytes: float = 0.0
    bytes_sent: float = 0.0
    value: float = 0.0
    decisions: int = 0


class DownlinkSimulator:
    """Slot-level simulation of per-satellite buffers and ground-pass transmissions."""

    def __init__(self, cfg: SchedulerConfig, horizon_s: float) -> None:
        self.cfg = cfg
        self.horizon_s = horizon_s
        self.value_model = MissionValueModel(cfg)
        self.buffer_capacity = cfg.buffer_capacity_mb * 1e6

    def run(self, policy: DownlinkPolicy, products: Sequence[DownlinkProduct],
            passes: Sequence[GroundPass]) -> SimulationLog:
        log = SimulationLog(policy.name)
        policy.reset()
        by_sat_p: dict[str, list[DownlinkProduct]] = {}
        for p in products:
            by_sat_p.setdefault(p.sat_name, []).append(p)
        by_sat_g: dict[str, list[GroundPass]] = {}
        for g in passes:
            by_sat_g.setdefault(g.sat_name, []).append(g)
        for sat in sorted(set(by_sat_p) | set(by_sat_g)):
            prods = sorted(by_sat_p.get(sat, []), key=lambda p: (p.available_time_s, p.product_id))
            gps = sorted(by_sat_g.get(sat, []), key=lambda g: g.start_s)
            self._run_satellite(policy, prods, gps, log)
            policy.on_episode_end()
        return log

    def _run_satellite(self, policy: DownlinkPolicy, prods: list[DownlinkProduct], gps: list[GroundPass],
                       log: SimulationLog) -> None:
        cap = self.buffer_capacity
        vm = self.value_model
        buffer: list[BufferItem] = []
        used = 0.0
        k = 0

        def drop(item: BufferItem, t: float, reason: str) -> None:
            log.drops[item.product.product_id] = (t, reason)
            policy.on_dropped(item, vm.drop_penalty(item.product))

        def admit_until(t: float) -> None:
            nonlocal k, used
            while k < len(prods) and prods[k].available_time_s <= t:
                p = prods[k]
                k += 1
                item = BufferItem(p, float(p.size_bytes), p.available_time_s)
                policy.on_admit(item)
                if p.size_bytes > cap:
                    drop(item, p.available_time_s, "oversize")
                    continue
                free = cap - used
                if p.size_bytes > free:
                    evict = policy.make_room(buffer, item, free, p.available_time_s)
                    if evict is None:
                        drop(item, p.available_time_s, "rejected (buffer full)")
                        continue
                    for i in sorted(set(evict), reverse=True):
                        victim = buffer.pop(i)
                        victim.evicted = True
                        used -= victim.remaining_bytes
                        drop(victim, p.available_time_s, "evicted")
                buffer.append(item)
                used += item.remaining_bytes

        for gp in gps:
            if gp.start_s > self.horizon_s:
                break
            slots = [s for s in gp.slots if s.t0_s <= self.horizon_s]
            pass_cap = float(sum(s.capacity_bytes for s in slots))
            remaining_pass = pass_cap
            current: BufferItem | None = None
            for slot in slots:
                admit_until(slot.t0_s)
                log.capacity_bytes += slot.capacity_bytes
                duration = slot.t1_s - slot.t0_s
                if duration <= 0 or slot.capacity_bytes <= 0:
                    continue
                rate = slot.capacity_bytes / duration
                budget = slot.capacity_bytes
                t = slot.t0_s
                while budget > 1e-6 and buffer:
                    if current is None or current.evicted:
                        ctx = PassContext(gp.station_name, gp.start_s, gp.end_s, pass_cap, remaining_pass,
                                          used, cap)
                        current = buffer[policy.select(buffer, t, ctx)]
                        log.decisions += 1
                    send = min(budget, current.remaining_bytes)
                    current.remaining_bytes -= send
                    used -= send
                    budget -= send
                    remaining_pass -= send
                    log.bytes_sent += send
                    t += send / rate
                    if current.remaining_bytes <= 1e-6:
                        buffer.remove(current)
                        p = current.product
                        latency = t - p.capture_time_s
                        reward = vm.delivered_value(p, latency)
                        log.deliveries[p.product_id] = (t, gp.station_name)
                        log.value += reward
                        policy.on_delivered(current, reward)
                        current = None
                remaining_pass -= budget  # unused capacity of this slot is lost
            # partially transmitted products keep their progress (resumable file transfer)
        admit_until(self.horizon_s)
        for it in buffer:
            log.pending.add(it.product.product_id)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class PolicyMetrics:
    policy: str
    products: int
    delivered: int
    dropped: int
    pending: int
    bytes_delivered_mb: float
    capacity_mb: float
    utilization: float
    alerts: int
    alerts_delivered: int
    fire_alerts: int
    fire_alerts_timely: int
    fire_alerts_dropped_or_expired: int
    hp_alerts: int
    hp_timely: int
    hp_late: int
    hp_dropped: int
    hp_expired_onboard: int
    hp_pending: int
    mean_alert_latency_min: float
    median_alert_latency_min: float
    p90_alert_latency_min: float
    hp_mean_latency_min: float
    hp_events: int
    hp_events_alerted: int
    hp_event_first_alert_min: float
    false_alarm_mb_sent: float
    scenes_delivered: int
    mission_value: float
    hp_events_notified: int = 0  # WUI fire events with a first notification (relay or ground)
    hp_event_first_notice_min: float = float("nan")

    def as_row(self) -> dict[str, float | int | str]:
        return dict(self.__dict__)


def compute_metrics(log: SimulationLog, products: Sequence[DownlinkProduct], cfg: SchedulerConfig,
                    horizon_s: float, relay: dict[int, float] | None = None) -> PolicyMetrics:
    ttl = cfg.alert_ttl_hours * 3600.0
    by_id = {p.product_id: p for p in products}
    lat: dict[int, float] = {pid: t - by_id[pid].capture_time_s for pid, (t, _st) in log.deliveries.items()}

    def nanmean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    alerts = [p for p in products if p.kind == ALERT]
    fire_alerts = [p for p in alerts if p.gt_fire]
    hp = [p for p in fire_alerts if p.gt_high_priority]
    fa_lat = [lat[p.product_id] / 60.0 for p in fire_alerts if p.product_id in lat]
    hp_lat = [lat[p.product_id] / 60.0 for p in hp if p.product_id in lat]

    def timely(p: DownlinkProduct) -> bool:
        return p.product_id in lat and lat[p.product_id] <= ttl

    def expired_onboard(p: DownlinkProduct) -> bool:
        return p.product_id in log.pending and horizon_s - p.capture_time_s > ttl

    # event level: first capture of a WUI fire event -> first delivered alert of that event
    first_capture: dict[int, float] = {}
    first_alert: dict[int, float] = {}
    for p in products:
        if not (p.gt_high_priority and p.fire_event_id is not None):
            continue
        ev = p.fire_event_id
        first_capture[ev] = min(first_capture.get(ev, math.inf), p.capture_time_s)
        if p.kind == ALERT and p.product_id in log.deliveries:
            first_alert[ev] = min(first_alert.get(ev, math.inf), log.deliveries[p.product_id][0])
    ev_lat = [(first_alert[e] - first_capture[e]) / 60.0 for e in first_alert]
    first_notice = dict(first_alert)
    for p in products:  # relay flash alerts reach the ground independently of the pass schedule
        if relay and p.product_id in relay and p.gt_high_priority and p.fire_event_id is not None:
            ev = p.fire_event_id
            first_notice[ev] = min(first_notice.get(ev, math.inf), relay[p.product_id])
    notice_lat = [(first_notice[e] - first_capture[e]) / 60.0 for e in first_notice]

    fa_bytes = sum(p.size_bytes for p in alerts if not p.gt_fire and p.product_id in log.deliveries)
    return PolicyMetrics(
        policy=log.policy,
        products=len(products),
        delivered=len(log.deliveries),
        dropped=len(log.drops),
        pending=len(log.pending),
        bytes_delivered_mb=log.bytes_sent / 1e6,
        capacity_mb=log.capacity_bytes / 1e6,
        utilization=log.bytes_sent / log.capacity_bytes if log.capacity_bytes > 0 else 0.0,
        alerts=len(alerts),
        alerts_delivered=sum(p.product_id in log.deliveries for p in alerts),
        fire_alerts=len(fire_alerts),
        fire_alerts_timely=sum(timely(p) for p in fire_alerts),
        fire_alerts_dropped_or_expired=sum((p.product_id in log.drops) or expired_onboard(p)
                                           or (p.product_id in lat and lat[p.product_id] > ttl) for p in fire_alerts),
        hp_alerts=len(hp),
        hp_timely=sum(timely(p) for p in hp),
        hp_late=sum(p.product_id in lat and lat[p.product_id] > ttl for p in hp),
        hp_dropped=sum(p.product_id in log.drops for p in hp),
        hp_expired_onboard=sum(expired_onboard(p) for p in hp),
        hp_pending=sum(p.product_id in log.pending and not expired_onboard(p) for p in hp),
        mean_alert_latency_min=nanmean(fa_lat),
        median_alert_latency_min=float(np.median(fa_lat)) if fa_lat else float("nan"),
        p90_alert_latency_min=float(np.percentile(fa_lat, 90)) if fa_lat else float("nan"),
        hp_mean_latency_min=nanmean(hp_lat),
        hp_events=len(first_capture),
        hp_events_alerted=len(first_alert),
        hp_event_first_alert_min=nanmean(ev_lat),
        false_alarm_mb_sent=fa_bytes / 1e6,
        scenes_delivered=sum(p.kind == SCENE and p.product_id in log.deliveries for p in products),
        mission_value=log.value,
        hp_events_notified=len(first_notice),
        hp_event_first_notice_min=nanmean(notice_lat),
    )


# ---------------------------------------------------------------------------
# Argus relay: flash alerts through a commercial LEO messaging constellation
# ---------------------------------------------------------------------------
@dataclass
class RelayResult:
    deliveries: dict[int, float]  # product_id -> ground arrival time (s)
    sent: int
    attempts: int
    capped: int
    bytes_sent: int
    mean_latency_s: float


def simulate_relay(products: Sequence[DownlinkProduct], cfg: SchedulerConfig, seed: int) -> RelayResult:
    """Each confident ALERT also emits a 340-byte flash packet (Iridium-SBD-class) as soon as it is
    produced: availability-limited, retried every ``relay_retry_s``, capped per satellite per day."""
    rng = np.random.default_rng(seed + 4242)
    deliveries: dict[int, float] = {}
    per_day: dict[tuple[str, int], int] = {}
    attempts = capped = 0
    lat: list[float] = []
    for p in sorted(products, key=lambda q: (q.available_time_s, q.product_id)):
        if p.kind != ALERT or p.confidence < cfg.relay_min_confidence:
            continue
        key = (p.sat_name, int(p.available_time_s // 86400))
        if per_day.get(key, 0) >= cfg.relay_daily_cap:
            capped += 1
            continue
        per_day[key] = per_day.get(key, 0) + 1
        t = p.available_time_s
        for _ in range(12):
            attempts += 1
            if rng.random() < cfg.relay_availability:
                t += float(rng.uniform(*cfg.relay_latency_s))
                deliveries[p.product_id] = t
                lat.append(t - p.capture_time_s)
                break
            t += cfg.relay_retry_s
    return RelayResult(deliveries, len(deliveries), attempts, capped, len(deliveries) * cfg.relay_packet_bytes,
                       float(np.mean(lat)) if lat else float("nan"))


# ---------------------------------------------------------------------------
# RL training on domain-randomised episodes
# ---------------------------------------------------------------------------
class EpisodeSampler:
    """Builds training episodes by resampling product attributes onto a satellite's capture timeline.

    Randomises the alert rate (0.5x-3x), product attributes (drawn from the global
    product pool), capture-time jitter, which satellite's pass schedule is used and the
    link capacity (0.6x-1.4x), so the learned policy does not overfit the evaluation day.
    """

    def __init__(self, products: Sequence[DownlinkProduct], passes: Sequence[GroundPass], seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.products_by_sat: dict[str, list[DownlinkProduct]] = {}
        for p in products:
            self.products_by_sat.setdefault(p.sat_name, []).append(p)
        self.passes_by_sat: dict[str, list[GroundPass]] = {}
        for g in passes:
            self.passes_by_sat.setdefault(g.sat_name, []).append(g)
        self.sats = [s for s in self.products_by_sat if self.passes_by_sat.get(s)]
        self.alerts = [p for p in products if p.kind == ALERT]
        self.scenes = [p for p in products if p.kind == SCENE]
        self.alert_frac = len(self.alerts) / max(1, len(products))
        self._next_id = 10_000_000

    def sample(self) -> tuple[list[DownlinkProduct], list[GroundPass]]:
        rng = self.rng
        sat = self.sats[int(rng.integers(len(self.sats)))]
        p_alert = min(0.8, max(0.02, self.alert_frac) * float(rng.uniform(0.5, 3.0)))
        out: list[DownlinkProduct] = []
        for p in self.products_by_sat[sat]:
            pool = self.alerts if (self.alerts and rng.random() < p_alert) or not self.scenes else self.scenes
            q = pool[int(rng.integers(len(pool)))]
            t = max(0.0, p.capture_time_s + float(rng.uniform(-300.0, 300.0)))
            out.append(replace(q, product_id=self._next_id, sat_name=sat, capture_time_s=t,
                               available_time_s=t + (p.available_time_s - p.capture_time_s)))
            self._next_id += 1
        pass_sat = self.sats[int(rng.integers(len(self.sats)))]
        scale = float(rng.uniform(0.6, 1.4))
        gps = [replace(g.scaled(scale), sat_name=sat) for g in self.passes_by_sat[pass_sat]]
        return out, gps


def train_q_learning(policy: QLearningPolicy, sim: DownlinkSimulator, products: Sequence[DownlinkProduct],
                     passes: Sequence[GroundPass], cfg: SchedulerConfig, seed: int,
                     log: Logger | None = None) -> list[float]:
    sampler = EpisodeSampler(products, passes, seed)
    if not sampler.sats:
        return []
    returns: list[float] = []
    policy.training = True
    n = max(1, cfg.rl_episodes)
    for ep in range(n):
        policy.epsilon = cfg.rl_epsilon_start + (cfg.rl_epsilon_end - cfg.rl_epsilon_start) * ep / max(1, n - 1)
        prods, gps = sampler.sample()
        returns.append(sim.run(policy, prods, gps).value)
        if log and (ep + 1) % max(1, n // 4) == 0:
            window = returns[-max(1, n // 4):]
            log(f"    RL episode {ep + 1:4d}/{n}  eps {policy.epsilon:.2f}  mean return {np.mean(window):7.2f}")
    policy.training = False
    policy.epsilon = 0.0
    return returns


@dataclass
class SchedulingResult:
    metrics: list[PolicyMetrics]
    logs: dict[str, SimulationLog]
    rl_returns: list[float]
    rl_class_ranking: list[tuple[str, float]]
    rl_states_visited: int
    rl_train_time_s: float
    offered_mb: float
    capacity_mb: float
    load_ratio: float
    validation_returns: dict[str, float] = field(default_factory=dict)
    validation_episodes: int = 0
    relay: RelayResult | None = None

    def metric(self, policy: str) -> PolicyMetrics:
        for m in self.metrics:
            if m.policy == policy:
                return m
        raise KeyError(policy)


def run_scheduling_comparison(products: Sequence[DownlinkProduct], passes: Sequence[GroundPass],
                              cfg: SchedulerConfig, horizon_s: float, seed: int,
                              log: Logger = print) -> SchedulingResult:
    sim = DownlinkSimulator(cfg, horizon_s)
    rl = QLearningPolicy(cfg, seed)
    fifo, heuristic = FIFOPolicy(), UrgencyHeuristicPolicy(cfg)

    # held-out randomised episodes (different sampler seed) to measure what the agent actually learned
    val_sampler = EpisodeSampler(products, passes, seed + 9_999)
    n_val = cfg.rl_validation_episodes if val_sampler.sats else 0
    episodes = [val_sampler.sample() for _ in range(n_val)]

    def validate(policy: DownlinkPolicy) -> float:
        return float(np.mean([sim.run(policy, p, g).value for p, g in episodes])) if episodes else float("nan")

    validation: dict[str, float] = {}
    if episodes:
        validation["Q-learning (untrained)"] = validate(QLearningPolicy(cfg, seed))
    t0 = time.perf_counter()
    log(f"  training Q-learning scheduler on {cfg.rl_episodes} domain-randomised episodes")
    returns = train_q_learning(rl, sim, products, passes, cfg, seed, log)
    train_time = time.perf_counter() - t0
    if episodes:
        for policy in (RandomPolicy(seed), fifo, heuristic, rl):
            validation[policy.name] = validate(policy)

    relay = simulate_relay(products, cfg, seed) if cfg.relay_enabled else None
    metrics: list[PolicyMetrics] = []
    logs: dict[str, SimulationLog] = {}
    for policy in (fifo, heuristic, rl):
        lg = sim.run(policy, products, passes)
        logs[policy.name] = lg
        metrics.append(compute_metrics(lg, products, cfg, horizon_s, relay.deliveries if relay else None))
    offered = sum(p.size_bytes for p in products) / 1e6
    capacity = metrics[0].capacity_mb if metrics else 0.0
    return SchedulingResult(
        metrics=metrics, logs=logs, rl_returns=returns, rl_class_ranking=rl.class_ranking(),
        rl_states_visited=int(np.count_nonzero(rl.visits.sum(axis=1))), rl_train_time_s=train_time,
        offered_mb=offered, capacity_mb=capacity, load_ratio=offered / capacity if capacity > 0 else float("inf"),
        validation_returns=validation, validation_episodes=n_val, relay=relay,
    )
