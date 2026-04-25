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
    from scipy.ndimage import gaussian_filter, maximum_filter

    normalized = normalize_sar(image)
    response = np.maximum(normalized[0], normalized[1])
    response = gaussian_filter(response, sigma=1.2)

    threshold = float(np.quantile(response, threshold_quantile))
    neighborhood = max(3, min_distance * 2 + 1)
    local_max = response == maximum_filter(response, size=neighborhood, mode="nearest")

    mask = local_max & (response >= threshold)
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
