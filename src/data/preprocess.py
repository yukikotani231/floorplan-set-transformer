"""SVG parsing utilities for FloorPlanCAD dataset.

This module handles parsing SVG files and extracting CAD primitives
with their annotations.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from svgpathtools import Arc, CubicBezier, Line, QuadraticBezier, parse_path

from .features import CADPrimitive


def parse_svg_file(svg_path: Path) -> list[CADPrimitive]:
    """Parse an SVG file and extract CAD primitives.

    Args:
        svg_path: Path to the SVG file.

    Returns:
        List of CADPrimitive objects.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    primitives: list[CADPrimitive] = []

    # Find all path elements
    for elem in root.iter():
        tag = elem.tag
        if "}" in tag:
            tag = tag.split("}")[1]

        if tag == "path":
            d_attr = elem.get("d")
            if d_attr:
                # Extract semantic label and instance from attributes
                semantic_label = _extract_label(elem)
                instance_id = _extract_instance(elem)

                # Parse the path
                path_primitives = _parse_path_element(d_attr, semantic_label, instance_id)
                primitives.extend(path_primitives)

        elif tag == "line":
            primitive = _parse_line_element(elem)
            if primitive:
                primitives.append(primitive)

        elif tag == "circle":
            primitive = _parse_circle_element(elem)
            if primitive:
                primitives.append(primitive)

        elif tag == "polyline" or tag == "polygon":
            poly_primitives = _parse_polyline_element(elem)
            primitives.extend(poly_primitives)

    return primitives


def _extract_label(elem: ET.Element) -> int | None:
    """Extract semantic label from element attributes."""
    # FloorPlanCAD uses 'semantic-id' attribute
    label = (
        elem.get("semantic-id")
        or elem.get("semantic_label")
        or elem.get("class")
        or elem.get("data-label")
    )
    if label is not None:
        try:
            return int(label)
        except ValueError:
            return None
    return None


def _extract_instance(elem: ET.Element) -> int | None:
    """Extract instance ID from element attributes."""
    # FloorPlanCAD uses 'instance-id' attribute
    instance = elem.get("instance-id") or elem.get("instance_id") or elem.get("data-instance")
    if instance is not None:
        try:
            return int(instance)
        except ValueError:
            return None
    return None


def _parse_path_element(
    d_attr: str,
    semantic_label: int | None,
    instance_id: int | None,
) -> list[CADPrimitive]:
    """Parse SVG path 'd' attribute into primitives."""
    primitives: list[CADPrimitive] = []

    try:
        path = parse_path(d_attr)
    except Exception:
        return primitives

    for segment in path:
        if isinstance(segment, Line):
            primitives.append(
                CADPrimitive(
                    primitive_type="line",
                    start_point=(segment.start.real, segment.start.imag),
                    end_point=(segment.end.real, segment.end.imag),
                    semantic_label=semantic_label,
                    instance_id=instance_id,
                )
            )
        elif isinstance(segment, Arc):
            primitives.append(
                CADPrimitive(
                    primitive_type="arc",
                    start_point=(segment.start.real, segment.start.imag),
                    end_point=(segment.end.real, segment.end.imag),
                    center=(segment.center.real, segment.center.imag),
                    radius=segment.radius.real,
                    semantic_label=semantic_label,
                    instance_id=instance_id,
                )
            )
        elif isinstance(segment, (CubicBezier, QuadraticBezier)):
            control_points = []
            if isinstance(segment, CubicBezier):
                control_points = [
                    (segment.control1.real, segment.control1.imag),
                    (segment.control2.real, segment.control2.imag),
                ]
            else:
                control_points = [(segment.control.real, segment.control.imag)]

            primitives.append(
                CADPrimitive(
                    primitive_type="curve",
                    start_point=(segment.start.real, segment.start.imag),
                    end_point=(segment.end.real, segment.end.imag),
                    control_points=control_points,
                    semantic_label=semantic_label,
                    instance_id=instance_id,
                )
            )

    return primitives


def _parse_line_element(elem: ET.Element) -> CADPrimitive | None:
    """Parse SVG line element."""
    try:
        x1 = float(elem.get("x1", 0))
        y1 = float(elem.get("y1", 0))
        x2 = float(elem.get("x2", 0))
        y2 = float(elem.get("y2", 0))
    except ValueError:
        return None

    return CADPrimitive(
        primitive_type="line",
        start_point=(x1, y1),
        end_point=(x2, y2),
        semantic_label=_extract_label(elem),
        instance_id=_extract_instance(elem),
    )


def _parse_circle_element(elem: ET.Element) -> CADPrimitive | None:
    """Parse SVG circle element as an arc."""
    try:
        cx = float(elem.get("cx", 0))
        cy = float(elem.get("cy", 0))
        r = float(elem.get("r", 0))
    except ValueError:
        return None

    return CADPrimitive(
        primitive_type="arc",
        start_point=(cx + r, cy),
        end_point=(cx + r, cy),  # Full circle
        center=(cx, cy),
        radius=r,
        semantic_label=_extract_label(elem),
        instance_id=_extract_instance(elem),
    )


def _parse_polyline_element(elem: ET.Element) -> list[CADPrimitive]:
    """Parse SVG polyline/polygon element into line segments."""
    points_attr = elem.get("points", "")
    semantic_label = _extract_label(elem)
    instance_id = _extract_instance(elem)

    # Parse points: "x1,y1 x2,y2 ..."
    points: list[tuple[float, float]] = []
    for point_str in points_attr.strip().split():
        parts = re.split(r"[,\s]+", point_str.strip())
        if len(parts) >= 2:
            try:
                points.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue

    primitives: list[CADPrimitive] = []
    for i in range(len(points) - 1):
        primitives.append(
            CADPrimitive(
                primitive_type="line",
                start_point=points[i],
                end_point=points[i + 1],
                semantic_label=semantic_label,
                instance_id=instance_id,
            )
        )

    return primitives


def normalize_coordinates(
    primitives: list[CADPrimitive],
    target_range: tuple[float, float] = (0.0, 1.0),
) -> list[CADPrimitive]:
    """Normalize coordinates to a target range.

    Args:
        primitives: List of primitives.
        target_range: Target coordinate range (min, max).

    Returns:
        New list of primitives with normalized coordinates.
    """
    if not primitives:
        return primitives

    # Find bounding box
    all_x = []
    all_y = []
    for p in primitives:
        all_x.extend([p.start_point[0], p.end_point[0]])
        all_y.extend([p.start_point[1], p.end_point[1]])

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Compute scale
    range_x = max_x - min_x if max_x > min_x else 1.0
    range_y = max_y - min_y if max_y > min_y else 1.0
    scale = max(range_x, range_y)

    target_min, target_max = target_range
    target_scale = target_max - target_min

    def normalize_point(x: float, y: float) -> tuple[float, float]:
        nx = ((x - min_x) / scale) * target_scale + target_min
        ny = ((y - min_y) / scale) * target_scale + target_min
        return (nx, ny)

    normalized = []
    for p in primitives:
        new_start = normalize_point(*p.start_point)
        new_end = normalize_point(*p.end_point)
        new_center = None
        new_radius = None
        new_control_points = None

        if p.center:
            new_center = normalize_point(*p.center)
        if p.radius:
            new_radius = (p.radius / scale) * target_scale
        if p.control_points:
            new_control_points = [normalize_point(*cp) for cp in p.control_points]

        normalized.append(
            CADPrimitive(
                primitive_type=p.primitive_type,
                start_point=new_start,
                end_point=new_end,
                control_points=new_control_points,
                center=new_center,
                radius=new_radius,
                semantic_label=p.semantic_label,
                instance_id=p.instance_id,
            )
        )

    return normalized
