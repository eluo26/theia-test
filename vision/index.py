"""Build a catalog from a scan directory.

Public entry point: build_catalog(scan_dir) -> Catalog.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from vision import detector as detector_lib
from vision import grok_client
from vision.config import Settings, load_settings
from vision.detector import GROK_ONLY_CONFIDENCE, refine_box
from vision.geometry import angular_distance_deg, box_center_angles
from vision.grok_client import GrokCallError, redact
from vision.ingest import LoadedFrame, load_scan
from vision.matching import token_ratio
from vision.preprocess import Tile, encode_png, make_tiles, map_norm_box_to_frame, prepare_frame, shift_box
from vision.schemas import Catalog, CatalogObject, Detection, IndexResponse, norm_box_to_pixels
from vision.viz import annotate_detections

logger = logging.getLogger("vision.index")


def build_catalog(
    scan_dir: str | Path,
    *,
    settings: Settings | None = None,
    client_factory=None,
    detector=None,
    sweep: tuple[float, float, float] | None = None,
    angles_csv: str | Path | None = None,
    use_cache: bool = True,
) -> Catalog:
    """Index angle-tagged photos (or one sweep video) and write catalog.json."""
    settings = settings if settings is not None else load_settings()
    source = Path(scan_dir)
    frames = load_scan(
        source,
        out_dir=settings.out_path,
        video_sample_every_s=settings.video_sample_every_s,
        blur_threshold=settings.blur_threshold,
        sweep=sweep,
        angles_csv=angles_csv,
    )
    _log_resolution_once(frames, settings)
    for frame in frames:
        prepare_frame(frame, settings)
    tiles = [tile for frame in frames for tile in make_tiles(frame, settings)]
    chosen = detector_lib.resolve_detector(settings, detector)
    detections = _index_tiles(
        tiles,
        settings,
        client_factory=client_factory,
        detector=chosen,
        use_cache=use_cache,
    )
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    scan_path = source.resolve()
    objects = merge_detections(detections, settings, created_at=created_at)
    catalog = Catalog(objects=objects, created_at=created_at, scan_dir=str(scan_path))
    write_catalog(catalog, settings.out_path / "catalog.json")
    _write_debug(frames, detections, objects, settings.out_path / "debug")
    logger.info("catalog objects=%s path=%s", len(objects), settings.out_path / "catalog.json")
    return catalog


def merge_detections(
    detections: list[Detection],
    settings: Settings,
    *,
    created_at: str,
) -> list[CatalogObject]:
    """Merge observations with similar labels that sit close together in angle."""
    if not detections:
        return []
    parent = list(range(len(detections)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            if _should_merge(detections[i], detections[j], settings):
                union(i, j)

    groups: dict[int, list[Detection]] = {}
    for index, detection in enumerate(detections):
        groups.setdefault(find(index), []).append(detection)

    representatives: list[tuple[Detection, list[Detection]]] = []
    for group in groups.values():
        representatives.append((_representative(group), group))
    representatives.sort(
        key=lambda item: (
            item[0].azimuth_deg if item[0].azimuth_deg is not None else 0.0,
            item[0].elevation_deg if item[0].elevation_deg is not None else 0.0,
            item[0].label,
            item[0].frame_file,
        )
    )
    objects: list[CatalogObject] = []
    for number, (rep, group) in enumerate(representatives, start=1):
        objects.append(_to_object(f"obj_{number:03d}", rep, group, created_at))
    return objects


def write_catalog(catalog: Catalog, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(catalog.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _index_tiles(tiles, settings, *, client_factory, detector, use_cache: bool) -> list[Detection]:
    if not tiles:
        return []

    async def _gather() -> list[list[Detection]]:
        limit = asyncio.Semaphore(settings.max_concurrency)

        async def one(tile: Tile) -> list[Detection]:
            async with limit:
                try:
                    return await asyncio.to_thread(
                        _index_tile,
                        tile,
                        settings,
                        client_factory,
                        detector,
                        use_cache,
                    )
                except GrokCallError as exc:
                    logger.warning(
                        "skipping tile %s offset=%s,%s error=%s",
                        tile.frame.source_file,
                        tile.offset_x,
                        tile.offset_y,
                        redact(str(exc)),
                    )
                    return []

        return await asyncio.gather(*(one(tile) for tile in tiles))

    groups = asyncio.run(_gather())
    return [detection for group in groups for detection in group]


def _index_tile(tile: Tile, settings, client_factory, detector, use_cache: bool) -> list[Detection]:
    encoded = encode_png(tile.image)
    parsed, _cache = grok_client.parse_image_bytes(
        encoded,
        "image/png",
        settings,
        prompt=grok_client.INDEX_PROMPT,
        response_model=IndexResponse,
        model=settings.fast_model,
        client_factory=client_factory,
        use_cache=use_cache,
    )
    frame = tile.frame
    detections: list[Detection] = []
    for obj in parsed.objects:
        grok_tile = list(norm_box_to_pixels(obj.box, tile.width, tile.height))
        grok_full = map_norm_box_to_frame(
            obj.box,
            tile.width,
            tile.height,
            tile.offset_x,
            tile.offset_y,
            frame.width,
            frame.height,
        )
        try:
            hits = detector.detect(tile.image, obj.label)
        except Exception as exc:  # a broken detector must not drop the Grok box
            logger.warning("detector failed on %s: %s", frame.source_file, redact(str(exc)))
            hits = []
        chosen, source, confidence = refine_box(grok_tile, hits, settings.refine_iou)
        if source == "grok":
            confidence = GROK_ONLY_CONFIDENCE
            full = grok_full
        else:
            full = shift_box(chosen, tile.offset_x, tile.offset_y, frame.width, frame.height)
        azimuth, elevation = box_center_angles(
            full, frame.width, frame.height, frame.pan_deg, frame.tilt_deg, settings
        )
        detections.append(
            Detection(
                label=obj.label,
                description=obj.description,
                bbox_px=full,
                grok_bbox_px=grok_full,
                count=obj.count,
                drug_name=obj.drug_name,
                expiry_text=obj.expiry_text,
                confidence=confidence,
                box_source=source,
                frame_file=frame.source_file,
                pan_deg=frame.pan_deg,
                tilt_deg=frame.tilt_deg,
                azimuth_deg=azimuth,
                elevation_deg=elevation,
                image_width=frame.width,
                image_height=frame.height,
                timestamp=frame.timestamp,
            )
        )
    return detections


def _should_merge(left: Detection, right: Detection, settings: Settings) -> bool:
    if left.azimuth_deg is None or right.azimuth_deg is None:
        return False
    if left.elevation_deg is None or right.elevation_deg is None:
        return False
    if token_ratio(left.label, right.label) < settings.label_sim:
        return False
    if not _drugs_compatible(left.drug_name, right.drug_name, settings.label_sim):
        return False
    separation = angular_distance_deg(
        left.azimuth_deg, left.elevation_deg, right.azimuth_deg, right.elevation_deg
    )
    return separation <= settings.merge_deg


def _drugs_compatible(left: str | None, right: str | None, threshold: float) -> bool:
    if not left or not right:
        return True
    return token_ratio(left, right) >= threshold


def _representative(group: list[Detection]) -> Detection:
    return min(group, key=_rep_key)


def _rep_key(detection: Detection) -> tuple:
    return (
        _center_distance(detection),
        -detection.confidence,
        detection.frame_file,
        tuple(detection.bbox_px),
    )


def _center_distance(detection: Detection) -> float:
    x1, y1, x2, y2 = detection.bbox_px
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    dx = (center_x - detection.image_width / 2) / detection.image_width
    dy = (center_y - detection.image_height / 2) / detection.image_height
    return (dx * dx + dy * dy) ** 0.5


def _to_object(object_id: str, rep: Detection, group: list[Detection], created_at: str) -> CatalogObject:
    if rep.azimuth_deg is None or rep.elevation_deg is None:
        raise ValueError(f"detection {rep.label} is missing azimuth or elevation")
    return CatalogObject(
        object_id=object_id,
        label=rep.label,
        description=rep.description,
        count=rep.count,
        drug_name=_prefer_text(rep, group, "drug_name"),
        expiry_text=_prefer_text(rep, group, "expiry_text"),
        azimuth_deg=rep.azimuth_deg,
        elevation_deg=rep.elevation_deg,
        confidence=rep.confidence,
        frame_file=rep.frame_file,
        bbox_px=list(rep.bbox_px),
        grok_bbox_px=list(rep.grok_bbox_px) if rep.grok_bbox_px is not None else None,
        box_source=rep.box_source,
        last_seen=rep.timestamp or created_at,
        pan_deg=rep.pan_deg,
        tilt_deg=rep.tilt_deg,
        image_width=rep.image_width,
        image_height=rep.image_height,
    )


def _prefer_text(rep: Detection, group: list[Detection], field: str) -> str | None:
    value = getattr(rep, field)
    if value:
        return value
    for detection in sorted(group, key=lambda item: item.frame_file):
        other = getattr(detection, field)
        if other:
            return other
    return None


def _write_debug(frames, detections, objects, dest: Path) -> None:
    by_file: dict[str, list[Detection]] = {}
    for detection in detections:
        by_file.setdefault(detection.frame_file, []).append(detection)
    ids: dict[str, dict[tuple[int, ...], str]] = {}
    for obj in objects:
        ids.setdefault(obj.frame_file, {})[tuple(obj.bbox_px)] = obj.object_id
    dest.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        frame_ids = ids.get(frame.source_file, {})
        annotate_detections(
            frame.image,
            by_file.get(frame.source_file, []),
            dest / f"{Path(frame.source_file).stem}.png",
            ids=frame_ids,
        )


def _log_resolution_once(frames: list[LoadedFrame], settings: Settings) -> None:
    if not frames or None not in (settings.fx, settings.fy, settings.cx, settings.cy):
        return
    frame = frames[0]
    if frame.width == settings.image_width and frame.height == settings.image_height:
        return
    logger.info(
        "frame %s is %sx%s; config placeholders are %sx%s. FOV is applied to the actual frame.",
        frame.source_file,
        frame.width,
        frame.height,
        settings.image_width,
        settings.image_height,
    )
