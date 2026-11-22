"""Geodesy, time and solar-geometry helpers (pure NumPy, no external dependencies)."""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

MU_EARTH_KM3_S2 = 398600.4418
R_EARTH_KM = 6378.137  # WGS84 equatorial radius
R_EARTH_MEAN_KM = 6371.0088
J2_EARTH = 1.08262668e-3
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
OMEGA_EARTH_RAD_S = 7.2921150e-5
SECONDS_PER_DAY = 86400.0
JD_J2000 = 2451545.0
JD_UNIX_EPOCH = 2440587.5
JD_SGP4_EPOCH0 = 2433281.5  # 1949-12-31 00:00 UT (SGP4 "days since" reference)
SSO_NODAL_RATE_RAD_S = 2.0 * math.pi / (365.2421897 * SECONDS_PER_DAY)


def datetime_to_jd(dt: datetime) -> float:
    """UTC datetime -> Julian date (UT1 ~ UTC at this fidelity)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() / SECONDS_PER_DAY + JD_UNIX_EPOCH


def gmst_rad(jd_ut1: np.ndarray | float) -> np.ndarray:
    """Greenwich mean sidereal time (IAU-82 polynomial)."""
    jd = np.asarray(jd_ut1, dtype=np.float64)
    d = jd - JD_J2000
    t = d / 36525.0
    gmst_deg = 280.46061837 + 360.98564736629 * d + 0.000387933 * t**2 - t**3 / 38710000.0
    return np.radians(np.mod(gmst_deg, 360.0))


def sun_eci_unit(jd: np.ndarray | float) -> np.ndarray:
    """Low-precision (~0.01 deg) solar direction in the ECI (true-of-date ~ J2000) frame."""
    jd_arr = np.atleast_1d(np.asarray(jd, dtype=np.float64))
    n = jd_arr - JD_J2000
    mean_lon = np.radians(np.mod(280.460 + 0.9856474 * n, 360.0))
    g = np.radians(np.mod(357.528 + 0.9856003 * n, 360.0))
    lam = mean_lon + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2.0 * g)
    eps = np.radians(23.439 - 4.0e-7 * n)
    return np.stack([np.cos(lam), np.cos(eps) * np.sin(lam), np.sin(eps) * np.sin(lam)], axis=-1)


def eci_to_ecef(r_eci: np.ndarray, gmst: np.ndarray) -> np.ndarray:
    """Rotate ECI vectors (..., 3) into the Earth-fixed frame by GMST (polar motion ignored)."""
    c = np.cos(gmst)
    s = np.sin(gmst)
    x = c * r_eci[..., 0] + s * r_eci[..., 1]
    y = -s * r_eci[..., 0] + c * r_eci[..., 1]
    return np.stack([x, y, r_eci[..., 2]], axis=-1)


def sun_ecef_unit(jd: np.ndarray | float) -> np.ndarray:
    jd_arr = np.atleast_1d(np.asarray(jd, dtype=np.float64))
    return eci_to_ecef(sun_eci_unit(jd_arr), gmst_rad(jd_arr))


def sun_right_ascension_deg(jd: float) -> float:
    u = sun_eci_unit(jd)[0]
    return math.degrees(math.atan2(float(u[1]), float(u[0]))) % 360.0


def raan_for_ltan_deg(ltan_hours: float, jd_epoch: float) -> float:
    """RAAN that places the ascending node at the requested local solar time."""
    return (sun_right_ascension_deg(jd_epoch) + 15.0 * (ltan_hours - 12.0)) % 360.0


def sso_inclination_deg(altitude_km: float, eccentricity: float = 0.0) -> float:
    """Inclination giving a J2 nodal precession of 360 deg / tropical year."""
    a = R_EARTH_KM + altitude_km
    n = math.sqrt(MU_EARTH_KM3_S2 / a**3)
    p = a * (1.0 - eccentricity**2)
    cos_i = -SSO_NODAL_RATE_RAD_S / (1.5 * n * J2_EARTH * (R_EARTH_KM / p) ** 2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_i))))


def orbital_period_min(semi_major_axis_km: float) -> float:
    return 2.0 * math.pi * math.sqrt(semi_major_axis_km**3 / MU_EARTH_KM3_S2) / 60.0


def solve_kepler(mean_anomaly: np.ndarray, ecc: float, iterations: int = 8) -> np.ndarray:
    m = np.asarray(mean_anomaly, dtype=np.float64)
    e_anom = m + ecc * np.sin(m)
    for _ in range(iterations):
        e_anom = e_anom - (e_anom - ecc * np.sin(e_anom) - m) / (1.0 - ecc * np.cos(e_anom))
    return e_anom


def geodetic_to_ecef(lat_deg: np.ndarray | float, lon_deg: np.ndarray | float,
                     alt_km: np.ndarray | float) -> np.ndarray:
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    alt = np.asarray(alt_km, dtype=np.float64)
    n = R_EARTH_KM / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    x = (n + alt) * np.cos(lat) * np.cos(lon)
    y = (n + alt) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt) * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def ecef_to_geodetic(r_ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ECEF (..., 3) km -> geodetic latitude [deg], longitude [deg], altitude [km]."""
    r = np.asarray(r_ecef, dtype=np.float64)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    alt = np.zeros_like(p)
    for _ in range(6):
        sin_lat = np.sin(lat)
        n = R_EARTH_KM / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
        alt = p / np.maximum(np.cos(lat), 1e-9) - n
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + alt)))
    return np.degrees(lat), np.degrees(lon), alt


def up_vector(lat_deg: np.ndarray | float, lon_deg: np.ndarray | float) -> np.ndarray:
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def topocentric(sat_ecef: np.ndarray, lat_deg: float, lon_deg: float,
                site_ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Elevation [deg], azimuth [deg] and slant range [km] of satellite positions from a site."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    north = np.array([-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)])
    up = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
    d = np.asarray(sat_ecef, dtype=np.float64) - np.asarray(site_ecef, dtype=np.float64)
    rng = np.linalg.norm(d, axis=-1)
    el = np.degrees(np.arcsin(np.clip((d @ up) / np.maximum(rng, 1e-9), -1.0, 1.0)))
    az = np.degrees(np.arctan2(d @ east, d @ north)) % 360.0
    return el, az, rng


def great_circle_km(lat1: np.ndarray | float, lon1: np.ndarray | float,
                    lat2: np.ndarray | float, lon2: np.ndarray | float) -> np.ndarray:
    """Haversine distance with NumPy broadcasting."""
    p1 = np.radians(np.asarray(lat1, dtype=np.float64))
    p2 = np.radians(np.asarray(lat2, dtype=np.float64))
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2, dtype=np.float64) - np.asarray(lon1, dtype=np.float64))
    h = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * R_EARTH_MEAN_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def slant_range_km(altitude_km: float, elevation_deg: float) -> float:
    re = R_EARTH_MEAN_KM
    el = math.radians(elevation_deg)
    return -re * math.sin(el) + math.sqrt((re * math.sin(el)) ** 2 + 2.0 * re * altitude_km + altitude_km**2)
