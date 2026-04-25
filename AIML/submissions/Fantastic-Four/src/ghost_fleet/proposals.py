from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scene_io import normalize_sar


@dataclass(frozen=True)
class Candidate:
    row: int
    col: int
    proposal_score: float


def generate_candidates(
    image: np.ndarray,
    *,
    max_candidates: int,
    threshold_quantile: float,
    min_distance: int,
    margin: int,
) -> list[Candidate]:
    from scipy.ndimage import maximum_filter

    normalized = normalize_sar(image)
    response = build_vessel_proposal_response(normalized)

    neighborhood = max(3, min_distance * 2 + 1)
    local_max = response == maximum_filter(response, size=neighborhood, mode="nearest")

    mask = np.zeros_like(response, dtype=bool)
    for quantile in _proposal_quantile_schedule(threshold_quantile):
        threshold = float(np.quantile(response, quantile))
        mask = local_max & (response >= threshold)
        if int(mask.sum()) >= max(1, min(max_candidates, 16)):
            break

    if margin > 0:
        mask[:margin, :] = False
        mask[-margin:, :] = False
        mask[:, :margin] = False
        mask[:, -margin:] = False

    rows, cols = np.nonzero(mask)
    scores = response[rows, cols]

    order = np.argsort(scores)[::-1]
    candidates: list[Candidate] = []
    for index in order[:max_candidates]:
        candidates.append(
            Candidate(
                row=int(rows[index]),
                col=int(cols[index]),
                proposal_score=float(scores[index]),
            )
        )
    return candidates


def _robust_unit_scale(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    low, high = np.percentile(finite, [1.0, 99.7])
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    scaled = (values - low) / (high - low)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def _proposal_quantile_schedule(start: float) -> list[float]:
    start = float(np.clip(start, 0.90, 0.9999))
    schedule = [start, 0.999, 0.998, 0.996, 0.993, 0.99, 0.985]
    ordered: list[float] = []
    for value in schedule:
        if value <= start and value not in ordered:
            ordered.append(value)
    return ordered or [start]


def build_vessel_proposal_response(normalized_sar: np.ndarray) -> np.ndarray:
    """Return a SAR-brightness response tuned for small bright ship targets.

    Raw bright pixels alone over-select wind streaks and coast clutter. This
    response mixes absolute brightness, local contrast, and a small-object
    band-pass term before non-maximum suppression.
    """

    from scipy.ndimage import gaussian_filter

    vv = normalized_sar[0].astype(np.float32)
    vh = normalized_sar[1].astype(np.float32)
    brightness = gaussian_filter(np.maximum(vv, vh), sigma=0.8)

    local_mean = gaussian_filter(brightness, sigma=8.0)
    local_var = gaussian_filter((brightness - local_mean) ** 2, sigma=8.0)
    local_contrast = (brightness - local_mean) / np.sqrt(local_var + 1e-4)

    small_object = gaussian_filter(brightness, sigma=0.7) - gaussian_filter(
        brightness,
        sigma=3.0,
    )
    cross_pol_support = gaussian_filter(vh, sigma=1.0)

    response = (
        0.35 * _robust_unit_scale(brightness)
        + 0.35 * _robust_unit_scale(local_contrast)
        + 0.20 * _robust_unit_scale(small_object)
        + 0.10 * _robust_unit_scale(cross_pol_support)
    )
    return gaussian_filter(response.astype(np.float32), sigma=0.6).astype(np.float32)
