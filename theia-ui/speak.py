"""Turn an aim result into sentences a person can read."""

from __future__ import annotations


def speak(payload: dict) -> str:
    """One or two sentences. No JSON fields."""
    result = payload.get("result") or {}
    status = result.get("status")
    label = (result.get("label") or "that object").strip()
    reason = (result.get("reason") or "").strip()
    aim = payload.get("aim") or {}
    if status == "found" and aim.get("azimuth_deg") is not None and aim.get("elevation_deg") is not None:
        horizontal = _horizontal(float(aim["azimuth_deg"]))
        vertical = _vertical(float(aim["elevation_deg"]))
        return f"The {label} is {horizontal} and {vertical}. The laser would point there."
    if status == "ambiguous":
        names = _candidate_names(result)
        if names:
            return f"More than one object could match: {names}. The laser stays off."
        return "More than one object could match. The laser stays off."
    sentence = "I don't see that, so the laser stays off."
    if reason:
        return f"{sentence} {reason}"
    return sentence


def inventory_sentence(payload: dict, *, limit: int = 8) -> str:
    """A short spoken list of distinct labels, not the raw catalog."""
    labels: list[str] = []
    seen: set[str] = set()
    for item in payload.get("items") or []:
        label = str(item.get("label") or "").strip()
        key = label.lower()
        if not label or key in seen:
            continue
        seen.add(key)
        labels.append(label)
        if len(labels) >= limit:
            break
    if not labels:
        return "Nothing has been identified yet."
    return "Also in view: " + ", ".join(labels) + "."


def _horizontal(azimuth: float) -> str:
    if abs(azimuth) < 5:
        return "straight ahead"
    side = "right" if azimuth > 0 else "left"
    return f"about {abs(azimuth):.0f} degrees to the {side}"


def _vertical(elevation: float) -> str:
    if abs(elevation) < 3:
        return "level"
    direction = "up" if elevation > 0 else "down"
    return f"about {abs(elevation):.0f} degrees {direction}"


def _candidate_names(result: dict) -> str:
    names: list[str] = []
    seen: set[str] = set()
    for candidate in result.get("candidates") or []:
        label = str(candidate.get("label") or "").strip()
        key = label.lower()
        if not label or key in seen:
            continue
        seen.add(key)
        names.append(label)
    return ", ".join(names)
