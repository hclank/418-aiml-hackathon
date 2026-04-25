from __future__ import annotations

import argparse
from pathlib import Path

from src.ghost_fleet.alerts import (
    Detection,
    generate_alerts,
    load_ais_cache,
    load_detections,
    write_alerts_json,
    write_detections_csv,
)


def detect_scene(args: argparse.Namespace) -> list[Detection]:
    import numpy as np
    import torch

    from src.ghost_fleet.model import length_scaled_to_meters, load_checkpoint
    from src.ghost_fleet.proposals import generate_candidates
    from src.ghost_fleet.scene_io import (
        extract_center_crop_from_paths,
        find_scene_paths,
        pixel_to_latlon_from_metadata,
        read_scene_overview,
    )

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)

    checkpoint_config = checkpoint.get("config", {})
    crop_size = args.crop_size or int(checkpoint_config.get("crop_size", 128))
    channel_names = tuple(
        checkpoint_config.get(
            "channel_names",
            ("vv", "vh", "vv_minus_vh", "depth", "wind_speed", "owi_mask"),
        )
    )
    min_vessel_score = (
        args.min_vessel_score
        if args.min_vessel_score is not None
        else float(checkpoint_config.get("best_vessel_threshold", 0.5))
    )

    try:
        scene_paths = find_scene_paths(args.scene_root, args.scene_id)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\n"
            "Download and extract the xView3 scene archives into data/xview3/full "
            "before using --scene-id mode."
        ) from exc
    overview_image, scene_metadata, row_scale, col_scale = read_scene_overview(
        scene_paths,
        max_dim=args.overview_max_dim,
    )
    candidates = generate_candidates(
        overview_image,
        max_candidates=args.max_candidates,
        threshold_quantile=args.proposal_quantile,
        min_distance=args.proposal_min_distance,
        margin=max(1, int((crop_size // 2) / max(row_scale, col_scale))),
    )

    detections: list[Detection] = []
    for start in range(0, len(candidates), args.batch_size):
        batch_candidates = candidates[start : start + args.batch_size]
        if not batch_candidates:
            continue

        crops = []
        for candidate in batch_candidates:
            full_row = min(
                scene_metadata.height - 1,
                max(0, int(round(candidate.row * row_scale))),
            )
            full_col = min(
                scene_metadata.width - 1,
                max(0, int(round(candidate.col * col_scale))),
            )
            crop = extract_center_crop_from_paths(
                scene_paths,
                full_row,
                full_col,
                crop_size,
                channel_names=channel_names,
            )
            crops.append(crop)

        batch = torch.from_numpy(np.stack(crops)).float().to(device)
        with torch.no_grad():
            outputs = model(batch)
            vessel_scores = torch.sigmoid(outputs["vessel_logits"]).cpu().numpy()
            fishing_scores = torch.sigmoid(outputs["fishing_logits"]).cpu().numpy()
            length_scores_m = length_scaled_to_meters(
                outputs["length_scaled"]
            ).cpu().numpy()

        for candidate, vessel_score, fishing_score, length_m in zip(
            batch_candidates,
            vessel_scores,
            fishing_scores,
            length_scores_m,
        ):
            if float(vessel_score) < min_vessel_score:
                continue
            full_row = min(
                scene_metadata.height - 1,
                max(0, int(round(candidate.row * row_scale))),
            )
            full_col = min(
                scene_metadata.width - 1,
                max(0, int(round(candidate.col * col_scale))),
            )
            lat, lon = pixel_to_latlon_from_metadata(
                scene_metadata,
                full_row,
                full_col,
            )
            detections.append(
                Detection(
                    scene_id=args.scene_id,
                    detection_id=f"{args.scene_id}_{full_row}_{full_col}",
                    lat=lat,
                    lon=lon,
                    score=float(vessel_score),
                    is_vessel_score=float(vessel_score),
                    is_fishing_score=float(fishing_score),
                    vessel_length_m=float(length_m),
                    row=int(full_row),
                    col=int(full_col),
                )
            )

    detections.sort(key=lambda item: item.is_vessel_score, reverse=True)
    return detections


def run_csv_alert_mode(args: argparse.Namespace) -> int:
    if args.ais is None or args.output is None:
        raise SystemExit("--detections mode requires both --ais and --output.")

    detections = load_detections(args.detections)
    ais_contacts = load_ais_cache(args.ais)
    alerts = generate_alerts(
        detections=detections,
        ais_contacts=ais_contacts,
        min_score=args.min_score,
        min_vessel_score=args.min_vessel_score if args.min_vessel_score is not None else 0.5,
        match_radius_m=args.match_radius_m,
    )
    if args.limit > 0:
        alerts = alerts[: args.limit]

    write_alerts_json(args.output, alerts)

    print(f"Loaded {len(detections)} detections and {len(ais_contacts)} AIS contacts.")
    print(f"Wrote {len(alerts)} unmatched-vessel alerts to {args.output}.")
    if alerts:
        print()
        print("Top alert:")
        print(alerts[0].human_message)
    else:
        print("No dark-vessel alerts passed the current thresholds.")
    return 0


def run_scene_mode(args: argparse.Namespace) -> int:
    if args.checkpoint is None:
        raise SystemExit("--scene-id mode requires --checkpoint.")

    detections = detect_scene(args)
    if args.limit > 0:
        detections = detections[: args.limit]

    if args.detections_output is not None:
        write_detections_csv(args.detections_output, detections)
        print(f"Wrote {len(detections)} detections to {args.detections_output}.")

    if args.ais is None:
        if args.detections_output is None:
            raise SystemExit(
                "--scene-id mode without --ais must provide --detections-output."
            )
        return 0

    if args.output is None:
        raise SystemExit("--scene-id mode with --ais requires --output.")

    ais_contacts = load_ais_cache(args.ais)
    alerts = generate_alerts(
        detections=detections,
        ais_contacts=ais_contacts,
        min_score=args.min_score,
        min_vessel_score=args.min_vessel_score if args.min_vessel_score is not None else 0.5,
        match_radius_m=args.match_radius_m,
    )
    if args.limit > 0:
        alerts = alerts[: args.limit]
    write_alerts_json(args.output, alerts)

    print(f"Generated {len(detections)} detections from scene {args.scene_id}.")
    print(f"Wrote {len(alerts)} unmatched-vessel alerts to {args.output}.")
    if alerts:
        print()
        print("Top alert:")
        print(alerts[0].human_message)
    else:
        print("No dark-vessel alerts passed the current thresholds.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ghost Fleet inference entry point. "
        "Use either --detections for alert generation from a CSV of detections, "
        "or --scene-id plus --checkpoint to scan a real SAR scene."
    )
    parser.add_argument("--detections", type=Path, help="CSV of model detections")
    parser.add_argument("--ais", type=Path, help="CSV of AIS contacts")
    parser.add_argument("--output", type=Path, help="Path to alerts JSON output")
    parser.add_argument("--match-radius-m", type=float, default=1500.0)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--min-vessel-score", type=float)
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--scene-id", help="xView3 scene id for full-scene inference")
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=Path("data/xview3/full"),
        help="Root directory containing extracted xView3 SAR scenes",
    )
    parser.add_argument("--checkpoint", type=Path, help="Trained checkpoint from train.py")
    parser.add_argument(
        "--detections-output",
        type=Path,
        help="Optional path to write raw detections CSV in scene mode",
    )
    parser.add_argument("--device", help="Torch device override, for example cuda or cpu")
    parser.add_argument("--crop-size", type=int, help="Crop size override for scene mode")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-candidates", type=int, default=512)
    parser.add_argument("--proposal-quantile", type=float, default=0.999)
    parser.add_argument("--proposal-min-distance", type=int, default=10)
    parser.add_argument("--overview-max-dim", type=int, default=2048)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.detections is not None:
        return run_csv_alert_mode(args)
    if args.scene_id is not None:
        return run_scene_mode(args)

    parser.error("Provide either --detections or --scene-id.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
