from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


EARTH_RADIUS_M = 6_371_000.0


@dataclass
class Detection:
    scene_id: str
    detection_id: str
    lat: float
    lon: float
    score: float
    is_vessel_score: float
    is_fishing_score: float
    vessel_length_m: float | None
    row: int | None = None
    col: int | None = None


@dataclass
class AISContact:
    mmsi: str
    lat: float
    lon: float


@dataclass
class Alert:
    scene_id: str
    detection_id: str
    lat: float
    lon: float
    model_score: float
    vessel_score: float
    fishing_score: float
    estimated_length_m: float | None
    matched_ais: bool
    matched_mmsi: str | None
    nearest_ais_distance_m: float | None
    risk_label: str
    human_message: str


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_M * c


def _parse_float(row: dict[str, str], *names: str, required: bool = True) -> float | None:
    for name in names:
        value = row.get(name)
        if value is None or value == "":
            continue
        return float(value)
    if required:
        raise ValueError(f"Missing required numeric column. Tried: {', '.join(names)}")
    return None


def _parse_int(row: dict[str, str], *names: str, required: bool = True) -> int | None:
    value = _parse_float(row, *names, required=required)
    if value is None:
        return None
    return int(round(value))


def _parse_str(row: dict[str, str], *names: str, required: bool = True) -> str | None:
    for name in names:
        value = row.get(name)
        if value is None or value == "":
            continue
        return value
    if required:
        raise ValueError(f"Missing required text column. Tried: {', '.join(names)}")
    return None


def load_detections(path: Path) -> list[Detection]:
    detections: list[Detection] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            detection = Detection(
                scene_id=_parse_str(row, "scene_id"),
                detection_id=_parse_str(row, "detection_id", "detect_id", required=False)
                or f"det-{index:04d}",
                lat=_parse_float(row, "lat", "latitude", "detect_lat"),
                lon=_parse_float(row, "lon", "longitude", "detect_lon"),
                score=_parse_float(row, "score", "confidence", "detect_score"),
                is_vessel_score=_parse_float(
                    row,
                    "is_vessel_score",
                    "vessel_score",
                    "object_score",
                ),
                is_fishing_score=_parse_float(
                    row,
                    "is_fishing_score",
                    "fishing_score",
                    required=False,
                )
                or 0.0,
                vessel_length_m=_parse_float(
                    row,
                    "vessel_length_m",
                    "estimated_length_m",
                    "length_m",
                    required=False,
                ),
                row=_parse_int(row, "row", "detect_scene_row", required=False),
                col=_parse_int(row, "col", "detect_scene_column", required=False),
            )
            detections.append(detection)
    return detections


def write_detections_csv(path: Path, detections: list[Detection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scene_id",
                "detection_id",
                "lat",
                "lon",
                "score",
                "is_vessel_score",
                "is_fishing_score",
                "vessel_length_m",
                "row",
                "col",
            ],
        )
        writer.writeheader()
        for detection in detections:
            writer.writerow(asdict(detection))


def load_ais_cache(path: Path) -> list[AISContact]:
    contacts: list[AISContact] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            contacts.append(
                AISContact(
                    mmsi=_parse_str(row, "mmsi", "imo", "vessel_id"),
                    lat=_parse_float(row, "lat", "latitude"),
                    lon=_parse_float(row, "lon", "longitude"),
                )
            )
    return contacts


def nearest_ais_contact(
    detection: Detection,
    contacts: Iterable[AISContact],
    match_radius_m: float,
) -> tuple[AISContact | None, float | None]:
    best_contact: AISContact | None = None
    best_distance: float | None = None

    for contact in contacts:
        distance = haversine_m(detection.lat, detection.lon, contact.lat, contact.lon)
        if best_distance is None or distance < best_distance:
            best_contact = contact
            best_distance = distance

    if best_distance is None or best_distance > match_radius_m:
        return None, best_distance
    return best_contact, best_distance


def classify_risk(detection: Detection) -> str:
    if detection.is_fishing_score >= 0.7:
        return "possible_dark_fishing_vessel"
    if detection.vessel_length_m is not None and detection.vessel_length_m >= 150:
        return "possible_large_dark_vessel"
    if detection.vessel_length_m is not None and detection.vessel_length_m >= 90:
        return "possible_medium_large_dark_vessel"
    return "possible_dark_vessel"


def build_human_message(alert: Alert) -> str:
    vessel_type = "Likely Fishing Vessel" if alert.fishing_score >= 0.7 else "Likely Vessel"
    length_text = (
        f" Estimated length: {alert.estimated_length_m:.1f} m."
        if alert.estimated_length_m is not None
        else ""
    )
    confidence = max(alert.model_score, alert.vessel_score) * 100.0
    return (
        f"Vessel detected at ({alert.lat:.5f}, {alert.lon:.5f}) with no matching AIS "
        f"signature within the configured radius. Classification: {vessel_type}. "
        f"Confidence: {confidence:.1f}%.{length_text}"
    )


def generate_alerts(
    detections: Iterable[Detection],
    ais_contacts: Iterable[AISContact],
    min_score: float,
    min_vessel_score: float,
    match_radius_m: float,
) -> list[Alert]:
    alerts: list[Alert] = []
    ais_contacts = list(ais_contacts)

    for detection in detections:
        if detection.score < min_score:
            continue
        if detection.is_vessel_score < min_vessel_score:
            continue

        matched_contact, nearest_distance = nearest_ais_contact(
            detection=detection,
            contacts=ais_contacts,
            match_radius_m=match_radius_m,
        )
        if matched_contact is not None:
            continue

        alert = Alert(
            scene_id=detection.scene_id,
            detection_id=detection.detection_id,
            lat=detection.lat,
            lon=detection.lon,
            model_score=detection.score,
            vessel_score=detection.is_vessel_score,
            fishing_score=detection.is_fishing_score,
            estimated_length_m=detection.vessel_length_m,
            matched_ais=False,
            matched_mmsi=None,
            nearest_ais_distance_m=nearest_distance,
            risk_label=classify_risk(detection),
            human_message="",
        )
        alert.human_message = build_human_message(alert)
        alerts.append(alert)

    alerts.sort(key=lambda item: (item.vessel_score, item.model_score), reverse=True)
    return alerts


def write_alerts_json(path: Path, alerts: list[Alert]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(alert) for alert in alerts]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
