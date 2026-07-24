#!/usr/bin/env python3
"""
OpenStudio OSM Geometry Compiler

Conservatively replaces geometry/spaces in approved Proposed and Baseline OSM
models while retaining template HVAC, constructions, schedules, loads, controls,
and simulation settings.

Designed for OpenStudio OSM 3.10-style files. Uses only Python standard library;
Shapely is optional and improves transfer of F-factor exposed perimeter data.

CLI:
  python osm_geometry_compiler.py --baseline baseline.osm --proposed proposed.osm \
      --geometry updated.osm --outdir output_folder

Run without arguments to launch a small Tkinter file-picker GUI.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import traceback
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

UUID_RE = re.compile(r"^\{[0-9a-fA-F-]{36}\}$")

GEOMETRY_TYPES = {
    "OS:Space",
    "OS:Surface",
    "OS:SubSurface",
    "OS:ShadingSurfaceGroup",
    "OS:ShadingSurface",
    "OS:InteriorPartitionSurfaceGroup",
    "OS:InteriorPartitionSurface",
    "OS:DaylightingControl",
    "OS:IlluminanceMap",
}

PRIMARY_GEOMETRY_TYPES = {"OS:Space", "OS:Surface", "OS:SubSurface"}

UNCONDITIONED_KEYWORDS = (
    "mechshaft",
    "mechanicalshaft",
    "elevator",
    "crawlspace",
)

# Exact construction names in the supplied compliant models, with tolerant fallbacks.
CONSTRUCTION_SEARCH = {
    "prm_wall": ["PRM Steel Framed Exterior Wall R-8.06", "PRM Steel Framed Exterior Wall"],
    "prm_roof": ["PRM IEAD Roof R-15.87", "PRM IEAD Roof"],
    "prm_floor": ["PRM Steel Framed Exterior Floor R-19.23", "PRM Steel Framed Exterior Floor"],
    "prm_window": ["U 0.57 SHGC 0.39 VT 0.43 Simple Glazing", "U 0.57 SHGC"],
    "prm_door": ["PRM Typical Insulated Metal Door R-1.43", "PRM Typical Insulated Metal Door"],
    "cfactor_wall": ["PRM Below Grade Wall_", "PRM Below Grade Wall"],
    "proposed_ground_wall": ["F1_WALL_CELLAR_12INCONC_R8_R13_GroundContactWall_Ground", "F1_WALL_CELLAR_12INCONC_R8_R13"],
    "proposed_ground_floor": ["G_FLOOR_CELLAR_SLAB_6IN_GroundContactFloor_Ground", "G_FLOOR_CELLAR_SLAB_6IN"],
    "proposed_roof": ["R1_ROOF_PROPOSED_ExteriorRoof_Outdoors", "R1_ROOF_PROPOSED"],
}


class CompileError(RuntimeError):
    pass


@dataclass
class OSMObject:
    obj_type: str
    fields: List[str]
    source_index: int = 0
    source_path: str = ""

    @property
    def handle(self) -> Optional[str]:
        if self.fields and UUID_RE.fullmatch(self.fields[0]):
            return self.fields[0]
        return None

    @handle.setter
    def handle(self, value: str) -> None:
        if not self.fields:
            self.fields.append(value)
        else:
            self.fields[0] = value

    @property
    def name(self) -> str:
        if self.handle and len(self.fields) > 1:
            return self.fields[1]
        return self.fields[0] if self.fields else ""

    def clone(self) -> "OSMObject":
        return OSMObject(self.obj_type, list(self.fields), self.source_index, self.source_path)


@dataclass
class ModelData:
    path: Path
    objects: List[OSMObject]
    by_handle: Dict[str, OSMObject] = field(init=False)
    by_type: Dict[str, List[OSMObject]] = field(init=False)

    def __post_init__(self) -> None:
        self.reindex()

    def reindex(self) -> None:
        self.by_handle = {}
        self.by_type = defaultdict(list)
        for obj in self.objects:
            self.by_type[obj.obj_type].append(obj)
            if obj.handle:
                if obj.handle in self.by_handle:
                    raise CompileError(f"Duplicate handle {obj.handle} in {self.path}")
                self.by_handle[obj.handle] = obj

    def names(self, obj_type: Optional[str] = None) -> Dict[str, OSMObject]:
        objs = self.objects if obj_type is None else self.by_type.get(obj_type, [])
        return {o.name: o for o in objs}


# -----------------------------------------------------------------------------
# OSM parser/writer
# -----------------------------------------------------------------------------

def parse_osm(path: Path) -> ModelData:
    text = path.read_text(encoding="utf-8-sig", errors="strict")
    objects: List[OSMObject] = []
    tokens: List[str] = []
    buf: List[str] = []
    in_quote = False
    quote_char = ""
    in_comment = False
    source_index = 0

    def flush_token() -> None:
        token = "".join(buf).strip()
        tokens.append(token)
        buf.clear()

    for ch in text:
        if in_comment:
            if ch in "\r\n":
                in_comment = False
                buf.append(" ")
            continue
        if not in_quote and ch == "!":
            in_comment = True
            continue
        if ch in ('"', "'"):
            if not in_quote:
                in_quote = True
                quote_char = ch
            elif quote_char == ch:
                in_quote = False
            buf.append(ch)
            continue
        if not in_quote and ch in ",;":
            flush_token()
            if ch == ";":
                # Remove leading empty tokens caused by whitespace/comments.
                while tokens and tokens[0] == "":
                    tokens.pop(0)
                if tokens:
                    obj_type = tokens[0]
                    fields = tokens[1:]
                    objects.append(OSMObject(obj_type, fields, source_index, str(path)))
                    source_index += 1
                tokens = []
            continue
        buf.append(ch)

    if in_quote:
        raise CompileError(f"Unterminated quoted value in {path}")
    if "".join(buf).strip() or any(t.strip() for t in tokens):
        raise CompileError(f"Trailing unterminated OSM object in {path}")
    if not objects:
        raise CompileError(f"No OSM objects parsed from {path}")
    return ModelData(path, objects)


def write_osm(path: Path, objects: Sequence[OSMObject]) -> None:
    lines: List[str] = []
    for obj in objects:
        lines.append(f"{obj.obj_type},")
        if not obj.fields:
            lines[-1] = f"{obj.obj_type};"
            lines.append("")
            continue
        for i, value in enumerate(obj.fields):
            end = ";" if i == len(obj.fields) - 1 else ","
            lines.append(f"  {value}{end}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def new_handle() -> str:
    return "{" + str(uuid.uuid4()) + "}"


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def _float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def vertices(obj: OSMObject) -> List[Tuple[float, float, float]]:
    if obj.obj_type == "OS:Surface":
        start = 11
    elif obj.obj_type == "OS:SubSurface":
        start = 8
    elif obj.obj_type in {"OS:ShadingSurface", "OS:InteriorPartitionSurface"}:
        # Common OpenStudio layout: handle, name, construction, group/space, number vertices, vertices...
        start = 5
    else:
        return []
    vals: List[float] = []
    for field_value in obj.fields[start:]:
        try:
            vals.append(float(field_value))
        except ValueError:
            continue
    return [tuple(vals[i:i + 3]) for i in range(0, len(vals) - 2, 3)]


def polygon_area_xy(points: Sequence[Tuple[float, float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area2 = 0.0
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        area2 += p[0] * q[1] - q[0] * p[1]
    return abs(area2) * 0.5


def newell_area_and_normal(points: Sequence[Tuple[float, float, float]]) -> Tuple[float, Tuple[float, float, float]]:
    if len(points) < 3:
        return 0.0, (0.0, 0.0, 0.0)
    nx = ny = nz = 0.0
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        nx += (p[1] - q[1]) * (p[2] + q[2])
        ny += (p[2] - q[2]) * (p[0] + q[0])
        nz += (p[0] - q[0]) * (p[1] + q[1])
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm <= 1e-15:
        return 0.0, (0.0, 0.0, 0.0)
    return norm * 0.5, (nx / norm, ny / norm, nz / norm)


def planarity_error(points: Sequence[Tuple[float, float, float]]) -> float:
    if len(points) < 4:
        return 0.0
    _, normal = newell_area_and_normal(points)
    if normal == (0.0, 0.0, 0.0):
        return float("inf")
    p0 = points[0]
    return max(abs((p[0] - p0[0]) * normal[0] + (p[1] - p0[1]) * normal[1] + (p[2] - p0[2]) * normal[2]) for p in points)


def space_bounds(geometry: ModelData) -> Dict[str, Tuple[float, float, float, float, float, float]]:
    coords: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    for surf in geometry.by_type.get("OS:Surface", []):
        if len(surf.fields) > 4:
            coords[surf.fields[4]].extend(vertices(surf))
    result = {}
    for space_handle, pts in coords.items():
        if pts:
            result[space_handle] = (
                min(p[0] for p in pts), max(p[0] for p in pts),
                min(p[1] for p in pts), max(p[1] for p in pts),
                min(p[2] for p in pts), max(p[2] for p in pts),
            )
    return result


def space_volume(space: OSMObject, bounds: Optional[Tuple[float, float, float, float, float, float]] = None) -> float:
    # gbXML-generated OS:Space stores Volume in the final field.
    if space.fields:
        value = _float(space.fields[-1], -1.0)
        if value >= 0:
            return value
    if bounds:
        return max(0.0, (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]) * (bounds[5] - bounds[4]))
    return 0.0


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def is_unconditioned_space(name: str) -> bool:
    n = normalized_name(name)
    return any(k in n for k in UNCONDITIONED_KEYWORDS)


# -----------------------------------------------------------------------------
# Model discovery
# -----------------------------------------------------------------------------

def find_by_name(model: ModelData, candidates: Sequence[str], allowed_types: Optional[Sequence[str]] = None) -> OSMObject:
    types = set(allowed_types) if allowed_types else None
    # Exact first.
    for candidate in candidates:
        for obj in model.objects:
            if (types is None or obj.obj_type in types) and obj.name == candidate:
                return obj
    # Case-insensitive prefix/substring fallback.
    for candidate in candidates:
        lc = candidate.lower()
        for obj in model.objects:
            if types is not None and obj.obj_type not in types:
                continue
            if obj.name.lower().startswith(lc) or lc in obj.name.lower():
                return obj
    raise CompileError(f"Required object not found in {model.path.name}: one of {list(candidates)}")


def discover_zones(template: ModelData) -> Tuple[OSMObject, OSMObject]:
    zones = template.by_type.get("OS:ThermalZone", [])
    if len(zones) < 2:
        raise CompileError(f"Template {template.path.name} needs at least conditioned and unconditioned thermal zones")
    unconditioned = next((z for z in zones if "uncondition" in z.name.lower()), None)
    conditioned = next((z for z in zones if z is not unconditioned and ("residential" in z.name.lower() or "condition" in z.name.lower())), None)
    if unconditioned is None:
        # Fall back to the zone assigned to old elevators/mechanical shafts.
        counts = Counter()
        for space in template.by_type.get("OS:Space", []):
            if is_unconditioned_space(space.name) and len(space.fields) > 10:
                counts[space.fields[10]] += 1
        if counts:
            unconditioned = template.by_handle.get(counts.most_common(1)[0][0])
    if conditioned is None:
        conditioned = next((z for z in zones if z is not unconditioned), None)
    if not conditioned or not unconditioned:
        raise CompileError(f"Could not identify conditioned/unconditioned zones in {template.path.name}")
    return conditioned, unconditioned


def discover_space_defaults(template: ModelData) -> Tuple[str, str, str]:
    old_spaces = template.by_type.get("OS:Space", [])
    if not old_spaces:
        raise CompileError(f"No template spaces in {template.path.name}")
    space_type = Counter(s.fields[2] for s in old_spaces if len(s.fields) > 2 and s.fields[2]).most_common(1)
    construction_set = Counter(s.fields[3] for s in old_spaces if len(s.fields) > 3 and s.fields[3]).most_common(1)
    schedule_set = Counter(s.fields[4] for s in old_spaces if len(s.fields) > 4 and s.fields[4]).most_common(1)
    return (
        space_type[0][0] if space_type else "",
        construction_set[0][0] if construction_set else "",
        schedule_set[0][0] if schedule_set else "",
    )


def discover_story_assignment(template: ModelData, z_value: float) -> str:
    stories = []
    for story in template.by_type.get("OS:BuildingStory", []):
        if len(story.fields) > 2 and story.handle:
            stories.append((_float(story.fields[2]), story.handle))
    if not stories:
        return ""
    return min(stories, key=lambda item: abs(item[0] - z_value))[1]


# -----------------------------------------------------------------------------
# Compile logic
# -----------------------------------------------------------------------------

def geometry_handle_set(model: ModelData) -> set[str]:
    return {obj.handle for obj in model.objects if obj.obj_type in GEOMETRY_TYPES and obj.handle}


def dependent_additional_properties(model: ModelData, handles: set[str]) -> List[OSMObject]:
    return [
        obj for obj in model.by_type.get("OS:AdditionalProperties", [])
        if len(obj.fields) > 1 and obj.fields[1] in handles
    ]


def remap_imported_handles(imported: List[OSMObject], occupied: set[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    seen = set(occupied)
    for obj in imported:
        if not obj.handle:
            continue
        old = obj.handle
        if old in seen:
            mapping[old] = new_handle()
            obj.handle = mapping[old]
        seen.add(obj.handle)
    if mapping:
        for obj in imported:
            obj.fields = [mapping.get(value, value) for value in obj.fields]
    return mapping


def source_construction_kind(geometry: ModelData, surface: OSMObject) -> str:
    if len(surface.fields) <= 3 or not surface.fields[3]:
        return "blank"
    construction = geometry.by_handle.get(surface.fields[3])
    if construction and construction.obj_type == "OS:Construction:AirBoundary":
        return "air"
    return "physical"


def pair_is_physical(geometry: ModelData, surface: OSMObject) -> bool:
    kinds = {source_construction_kind(geometry, surface)}
    if len(surface.fields) > 6 and surface.fields[5] == "Surface":
        mate = geometry.by_handle.get(surface.fields[6])
        if mate:
            kinds.add(source_construction_kind(geometry, mate))
    if "air" in kinds and "physical" not in kinds:
        return False
    return "physical" in kinds


def surface_zone_handle(surface: OSMObject, spaces_by_handle: Dict[str, OSMObject]) -> str:
    if len(surface.fields) <= 4:
        return ""
    space = spaces_by_handle.get(surface.fields[4])
    return space.fields[10] if space and len(space.fields) > 10 else ""


def baseline_construction_handles(template: ModelData) -> Dict[str, str]:
    result = {}
    for key, candidates in CONSTRUCTION_SEARCH.items():
        if key == "cfactor_wall":
            allowed = ["OS:Construction:CfactorUndergroundWall"]
        else:
            allowed = ["OS:Construction", "OS:Construction:CfactorUndergroundWall", "OS:Construction:FfactorGroundFloor"]
        result[key] = find_by_name(template, candidates, allowed).handle or ""
    return result


def construction_for_baseline_surface(
    surface: OSMObject,
    geometry_source: ModelData,
    spaces_by_handle: Dict[str, OSMObject],
    conditioned_handle: str,
    unconditioned_handle: str,
    constructions: Dict[str, str],
) -> str:
    f = surface.fields
    if len(f) < 7:
        return ""
    surface_type = f[2]
    boundary = f[5]
    zone = surface_zone_handle(surface, spaces_by_handle)
    space = spaces_by_handle.get(f[4])
    space_name = space.name if space else ""

    if boundary == "Outdoors":
        if surface_type == "Wall":
            return constructions["prm_wall"]
        if surface_type == "RoofCeiling":
            return constructions["proposed_roof"] if "elevator" in normalized_name(space_name) else constructions["prm_roof"]
        if surface_type == "Floor":
            return constructions["prm_floor"]
        return ""

    if boundary in {"Ground", "GroundFCfactorMethod"}:
        if surface_type == "Wall":
            return constructions["proposed_ground_wall"] if zone == unconditioned_handle else constructions["cfactor_wall"]
        if surface_type == "Floor" and zone == unconditioned_handle:
            return constructions["proposed_ground_floor"]
        return ""  # conditioned ground floors receive generated F-factor constructions later

    if boundary == "Surface":
        if not pair_is_physical(geometry_source, surface):
            return ""
        mate = geometry_source.by_handle.get(f[6])
        other_zone = surface_zone_handle(mate, spaces_by_handle) if mate else ""
        if zone == conditioned_handle and other_zone == unconditioned_handle:
            if surface_type == "Wall":
                return constructions["prm_wall"]
            if surface_type == "Floor":
                return constructions["prm_floor"]
            if surface_type == "RoofCeiling":
                return constructions["prm_roof"]
        return ""

    # For uncommon boundary types, preserve a same-name source construction if available.
    source_handle = f[3]
    source_obj = geometry_source.by_handle.get(source_handle)
    if source_obj:
        match = next((o for o in template.objects if o.name == source_obj.name and o.obj_type.startswith("OS:Construction")), None)
        if match:
            return match.handle or ""
    return ""


def baseline_subsurface_construction(
    subsurface: OSMObject,
    geometry_source: ModelData,
    spaces_by_handle: Dict[str, OSMObject],
    conditioned_handle: str,
    unconditioned_handle: str,
    constructions: Dict[str, str],
) -> str:
    if len(subsurface.fields) <= 4:
        return ""
    parent = geometry_source.by_handle.get(subsurface.fields[4])
    if not parent:
        return ""
    subtype = subsurface.fields[2].lower()
    is_window = "window" in subtype or "glassdoor" in subtype or "skylight" in subtype
    boundary = parent.fields[5] if len(parent.fields) > 5 else ""
    zone = surface_zone_handle(parent, spaces_by_handle)

    if boundary == "Outdoors":
        return constructions["prm_window"] if is_window else constructions["prm_door"]

    if boundary == "Surface" and pair_is_physical(geometry_source, parent):
        mate = geometry_source.by_handle.get(parent.fields[6]) if len(parent.fields) > 6 else None
        other_zone = surface_zone_handle(mate, spaces_by_handle) if mate else ""
        if zone == conditioned_handle and other_zone == unconditioned_handle:
            return constructions["prm_window"] if is_window else constructions["prm_door"]
    return ""


def make_ffactor_objects(
    template: ModelData,
    geometry_source: ModelData,
    imported_surfaces: List[OSMObject],
    spaces_by_handle: Dict[str, OSMObject],
    conditioned_handle: str,
    report: dict,
) -> List[OSMObject]:
    old_ffactors = template.by_type.get("OS:Construction:FfactorGroundFloor", [])
    old_ff_by_handle = {o.handle: o for o in old_ffactors if o.handle}
    old_total_exposed = sum(_float(o.fields[4]) for o in old_ffactors if len(o.fields) > 4)
    old_f_factor = _float(old_ffactors[0].fields[2], 1.26343630645112) if old_ffactors else 1.26343630645112

    target_surfaces = [
        s for s in imported_surfaces
        if len(s.fields) > 5
        and s.fields[2] == "Floor"
        and s.fields[5] in {"Ground", "GroundFCfactorMethod"}
        and surface_zone_handle(s, spaces_by_handle) == conditioned_handle
    ]
    if not target_surfaces:
        report["ffactor"] = {"count": 0, "old_total_exposed_perimeter_m": old_total_exposed}
        return []

    # Try spatial transfer from old ground floors. Shapely is optional.
    exposed: Dict[str, float] = {s.handle: 0.0 for s in target_surfaces if s.handle}
    method = "area-proportional fallback"
    try:
        from shapely.geometry import Polygon  # type: ignore

        old_floor_data = []
        for surf in template.by_type.get("OS:Surface", []):
            if len(surf.fields) <= 3 or surf.fields[3] not in old_ff_by_handle:
                continue
            pts = vertices(surf)
            poly = Polygon([(p[0], p[1]) for p in pts])
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.area <= 1e-9:
                continue
            ff = old_ff_by_handle[surf.fields[3]]
            old_floor_data.append((poly, _float(ff.fields[4]), poly.area))

        if old_floor_data:
            for target in target_surfaces:
                pts = vertices(target)
                poly = Polygon([(p[0], p[1]) for p in pts])
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.area <= 1e-9 or not target.handle:
                    continue
                value = 0.0
                for old_poly, old_ep, old_area in old_floor_data:
                    inter = poly.intersection(old_poly).area
                    if inter > 1e-9 and old_area > 0:
                        value += old_ep * inter / old_area
                exposed[target.handle] = value
            assigned = sum(exposed.values())
            if old_total_exposed > 0 and assigned > 1e-9:
                scale = old_total_exposed / assigned
                for key in exposed:
                    exposed[key] *= scale
            method = "spatial overlap transfer; total perimeter preserved"
    except Exception as exc:
        report.setdefault("warnings", []).append(f"Shapely unavailable or F-factor transfer failed ({exc}); used area-proportional fallback")

    if sum(exposed.values()) <= 1e-12 and old_total_exposed > 0:
        areas = {s.handle: polygon_area_xy(vertices(s)) for s in target_surfaces if s.handle}
        total_area = sum(areas.values())
        if total_area > 0:
            for handle, area in areas.items():
                exposed[handle] = old_total_exposed * area / total_area

    generated: List[OSMObject] = []
    for surf in target_surfaces:
        area = polygon_area_xy(vertices(surf))
        ep = exposed.get(surf.handle or "", 0.0)
        ff_handle = new_handle()
        ff_name = f"PRM Slab on Grade Floor_{surf.name}_0.73"
        ff = OSMObject(
            "OS:Construction:FfactorGroundFloor",
            [ff_handle, ff_name, f"{old_f_factor:.14g}", f"{area:.14g}", f"{ep:.14g}"],
        )
        generated.append(ff)
        surf.fields[3] = ff_handle
        surf.fields[5] = "GroundFCfactorMethod"
        if len(surf.fields) > 6:
            surf.fields[6] = ""

    report["ffactor"] = {
        "count": len(generated),
        "transfer_method": method,
        "old_total_exposed_perimeter_m": old_total_exposed,
        "new_total_exposed_perimeter_m": sum(exposed.values()),
        "f_factor_w_per_m_k": old_f_factor,
    }
    return generated


def make_baseline_infiltration(
    template: ModelData,
    imported_spaces: List[OSMObject],
    imported_surfaces: List[OSMObject],
    report: dict,
) -> List[OSMObject]:
    old = template.by_type.get("OS:SpaceInfiltration:DesignFlowRate", [])
    if not old:
        report["infiltration"] = {"count": 0, "note": "No template space-level infiltration objects"}
        return []
    ext_pattern = next((o for o in old if len(o.fields) > 4 and o.fields[4] == "Flow/ExteriorWallArea"), None)
    zero_pattern = next((o for o in old if len(o.fields) > 4 and o.fields[4] == "Flow/Space"), None)
    if not ext_pattern or not zero_pattern:
        raise CompileError("Baseline infiltration patterns are incomplete")

    has_exterior_wall = Counter()
    for surf in imported_surfaces:
        if len(surf.fields) > 5 and surf.fields[2] == "Wall" and surf.fields[5] == "Outdoors":
            has_exterior_wall[surf.fields[4]] += 1

    generated = []
    methods = Counter()
    for space in imported_spaces:
        if not space.handle:
            continue
        if "elevator" in normalized_name(space.name):
            continue  # Matches approved model behavior.
        pattern = ext_pattern if has_exterior_wall[space.handle] else zero_pattern
        obj = pattern.clone()
        obj.handle = new_handle()
        if len(obj.fields) > 1:
            obj.fields[1] = f"{space.name} Infiltration"
        if len(obj.fields) > 2:
            obj.fields[2] = space.handle
        generated.append(obj)
        methods[obj.fields[4] if len(obj.fields) > 4 else "unknown"] += 1
    report["infiltration"] = {"count": len(generated), "methods": dict(methods)}
    return generated


def update_zone_volumes(
    retained_objects: List[OSMObject],
    imported_spaces: List[OSMObject],
    bounds: Dict[str, Tuple[float, float, float, float, float, float]],
) -> Dict[str, float]:
    sums = defaultdict(float)
    for space in imported_spaces:
        if not space.handle or len(space.fields) <= 10:
            continue
        sums[space.fields[10]] += space_volume(space, bounds.get(space.handle))
    for zone in retained_objects:
        if zone.obj_type == "OS:ThermalZone" and zone.handle in sums:
            while len(zone.fields) <= 4:
                zone.fields.append("")
            zone.fields[4] = f"{sums[zone.handle]:.14g}"
    return dict(sums)


def compile_one(template_path: Path, geometry_path: Path, output_path: Path, mode: str) -> dict:
    template = parse_osm(template_path)
    geometry = parse_osm(geometry_path)
    report: dict = {
        "mode": mode,
        "template": str(template_path),
        "geometry_source": str(geometry_path),
        "output": str(output_path),
        "warnings": [],
    }

    # Version compatibility is mandatory.
    template_version = template.by_type.get("OS:Version", [None])[0]
    geometry_version = geometry.by_type.get("OS:Version", [None])[0]
    tv = template_version.fields[1] if template_version and len(template_version.fields) > 1 else "unknown"
    gv = geometry_version.fields[1] if geometry_version and len(geometry_version.fields) > 1 else "unknown"
    if tv != gv:
        raise CompileError(f"OSM version mismatch: template {tv}, geometry {gv}")
    report["openstudio_version"] = tv

    conditioned, unconditioned = discover_zones(template)
    space_type_handle, construction_set_handle, schedule_set_handle = discover_space_defaults(template)
    if not space_type_handle or not construction_set_handle:
        raise CompileError("Could not discover template SpaceType and DefaultConstructionSet assignments")

    old_geometry_handles = geometry_handle_set(template)
    old_ff_handles = {o.handle for o in template.by_type.get("OS:Construction:FfactorGroundFloor", []) if o.handle}

    remove_handles = set(old_geometry_handles) | set(old_ff_handles)
    remove_object_handles = set(remove_handles)
    for ap in dependent_additional_properties(template, remove_handles):
        if ap.handle:
            remove_object_handles.add(ap.handle)
    # Remove standards metadata tied to old F-factor constructions.
    for obj in template.by_type.get("OS:StandardsInformation:Construction", []):
        if len(obj.fields) > 1 and obj.fields[1] in old_ff_handles and obj.handle:
            remove_object_handles.add(obj.handle)

    # Remove objects that directly depend on old spaces and will be regenerated.
    if mode == "baseline":
        for obj in template.by_type.get("OS:SpaceInfiltration:DesignFlowRate", []):
            if obj.handle:
                remove_object_handles.add(obj.handle)

    retained = [obj.clone() for obj in template.objects if obj.handle not in remove_object_handles and obj.obj_type not in GEOMETRY_TYPES]

    # Verify no unexpected retained object still points to old geometry.
    unexpected = []
    allowed_old_ref_types = {"OS:AdditionalProperties", "OS:SpaceInfiltration:DesignFlowRate", "OS:Surface", "OS:SubSurface"}
    for obj in retained:
        for idx, value in enumerate(obj.fields):
            if value in old_geometry_handles:
                unexpected.append((obj.obj_type, obj.name, idx, value))
    if unexpected:
        raise CompileError(f"Unexpected template dependencies on old geometry: {unexpected[:20]}")

    # Import geometry and its AdditionalProperties only.
    new_geometry_original = [obj.clone() for obj in geometry.objects if obj.obj_type in GEOMETRY_TYPES]
    new_geometry_handles = {o.handle for o in new_geometry_original if o.handle}
    new_geometry_ap = [obj.clone() for obj in dependent_additional_properties(geometry, new_geometry_handles)]
    imported = new_geometry_original + new_geometry_ap
    occupied = {obj.handle for obj in retained if obj.handle}
    remap = remap_imported_handles(imported, occupied)
    if remap:
        report["warnings"].append(f"Remapped {len(remap)} imported handles that collided with retained template objects")

    # Rebuild a temporary geometry index after possible remapping.
    imported_geom = [o for o in imported if o.obj_type in GEOMETRY_TYPES]
    imported_ap = [o for o in imported if o.obj_type == "OS:AdditionalProperties"]
    geom_model = ModelData(geometry_path, imported_geom)
    imported_spaces = geom_model.by_type.get("OS:Space", [])
    imported_surfaces = geom_model.by_type.get("OS:Surface", [])
    imported_subsurfaces = geom_model.by_type.get("OS:SubSurface", [])
    bounds = space_bounds(geom_model)

    # Template assignments on every new space.
    zone_counts = Counter()
    zone_space_names = defaultdict(list)
    story_counts = Counter()
    for space in imported_spaces:
        while len(space.fields) <= 14:
            space.fields.append("")
        space.fields[2] = space_type_handle
        space.fields[3] = construction_set_handle
        space.fields[4] = schedule_set_handle
        zone = unconditioned if is_unconditioned_space(space.name) else conditioned
        space.fields[10] = zone.handle or ""
        zone_counts[zone.name] += 1
        zone_space_names[zone.name].append(space.name)
        if mode == "baseline":
            zmin = bounds.get(space.handle or "", (0, 0, 0, 0, 0, 0))[4]
            story_handle = discover_story_assignment(template, zmin)
            space.fields[9] = story_handle
            if story_handle:
                story_obj = template.by_handle.get(story_handle)
                story_counts[story_obj.name if story_obj else story_handle] += 1
        else:
            # Preserve proposed template behavior (no BuildingStory assignment).
            space.fields[9] = ""

    report["spaces"] = {
        "count": len(imported_spaces),
        "zone_assignments": dict(zone_counts),
        "zone_space_names": {k: sorted(v) for k, v in zone_space_names.items()},
        "story_assignments": dict(story_counts),
    }

    # Assign constructions.
    if mode == "proposed":
        for surf in imported_surfaces:
            if len(surf.fields) > 3:
                surf.fields[3] = ""
        for sub in imported_subsurfaces:
            if len(sub.fields) > 3:
                sub.fields[3] = ""
        generated_ff: List[OSMObject] = []
        generated_infiltration: List[OSMObject] = []
    elif mode == "baseline":
        c = baseline_construction_handles(template)
        spaces_by_handle = {s.handle: s for s in imported_spaces if s.handle}
        for surf in imported_surfaces:
            if len(surf.fields) > 3:
                surf.fields[3] = construction_for_baseline_surface(
                    surf, geom_model, spaces_by_handle,
                    conditioned.handle or "", unconditioned.handle or "", c,
                )
        for sub in imported_subsurfaces:
            if len(sub.fields) > 3:
                sub.fields[3] = baseline_subsurface_construction(
                    sub, geom_model, spaces_by_handle,
                    conditioned.handle or "", unconditioned.handle or "", c,
                )
        generated_ff = make_ffactor_objects(
            template, geom_model, imported_surfaces, spaces_by_handle,
            conditioned.handle or "", report,
        )
        generated_infiltration = make_baseline_infiltration(template, imported_spaces, imported_surfaces, report)
    else:
        raise CompileError(f"Unknown compile mode: {mode}")

    zone_volumes = update_zone_volumes(retained, imported_spaces, bounds)
    report["zone_volumes_m3"] = {
        (template.by_handle.get(h).name if template.by_handle.get(h) else h): v
        for h, v in zone_volumes.items()
    }

    # Stable, readable order: retained template, generated constructions, geometry, geometry metadata, generated loads.
    output_objects = retained + generated_ff + imported_geom + imported_ap + generated_infiltration
    output_model = ModelData(output_path, output_objects)

    validation = validate_compiled_model(output_model, geom_model, conditioned.handle or "", unconditioned.handle or "")
    report["validation"] = validation
    if validation["errors"]:
        raise CompileError("Compiled model failed static validation: " + "; ".join(validation["errors"][:10]))

    write_osm(output_path, output_objects)
    report["object_counts"] = dict(sorted(Counter(o.obj_type for o in output_objects).items()))
    return report


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate_compiled_model(
    model: ModelData,
    geometry_model: ModelData,
    conditioned_handle: str,
    unconditioned_handle: str,
) -> dict:
    errors: List[str] = []
    warnings: List[str] = []
    stats = Counter()
    handles = set(model.by_handle)

    # Geometry and references.
    for space in model.by_type.get("OS:Space", []):
        if len(space.fields) != 15:
            errors.append(f"Space {space.name} has {len(space.fields)} fields; expected 15 for OSM 3.10")
        if len(space.fields) <= 10 or space.fields[10] not in {conditioned_handle, unconditioned_handle}:
            errors.append(f"Space {space.name} lacks valid template thermal-zone assignment")
        if len(space.fields) <= 3 or not space.fields[2] or not space.fields[3]:
            errors.append(f"Space {space.name} lacks SpaceType or DefaultConstructionSet")

    for surf in model.by_type.get("OS:Surface", []):
        pts = vertices(surf)
        area, normal = newell_area_and_normal(pts)
        if len(pts) < 3 or area <= 1e-8:
            errors.append(f"Degenerate surface {surf.name}")
        pe = planarity_error(pts)
        if pe > 1e-4:
            errors.append(f"Nonplanar surface {surf.name}: {pe:.6g} m")
        if len(surf.fields) <= 4 or surf.fields[4] not in handles or model.by_handle[surf.fields[4]].obj_type != "OS:Space":
            errors.append(f"Surface {surf.name} has invalid Space reference")
        if len(surf.fields) > 5 and surf.fields[5] == "Surface":
            mate = model.by_handle.get(surf.fields[6]) if len(surf.fields) > 6 else None
            if not mate or mate.obj_type != "OS:Surface":
                errors.append(f"Surface {surf.name} has missing paired surface")
            elif len(mate.fields) <= 6 or mate.fields[6] != surf.handle:
                errors.append(f"Surface pair is not reciprocal: {surf.name} / {mate.name}")
        if surf.fields[2] == "Floor" and normal[2] > 1e-5:
            warnings.append(f"Floor normal points upward: {surf.name}")
        if surf.fields[2] == "RoofCeiling" and normal[2] < -1e-5:
            warnings.append(f"Roof normal points downward: {surf.name}")

    for sub in model.by_type.get("OS:SubSurface", []):
        pts = vertices(sub)
        area, _ = newell_area_and_normal(pts)
        if len(pts) < 3 or area <= 1e-8:
            errors.append(f"Degenerate subsurface {sub.name}")
        pe = planarity_error(pts)
        if pe > 1e-4:
            errors.append(f"Nonplanar subsurface {sub.name}: {pe:.6g} m")
        parent = model.by_handle.get(sub.fields[4]) if len(sub.fields) > 4 else None
        if not parent or parent.obj_type != "OS:Surface":
            errors.append(f"SubSurface {sub.name} has invalid parent Surface")

    # Only field 0 is always a handle; handle-looking Names are legal, so check refs from index 2 onward.
    dangling = []
    for obj in model.objects:
        for idx, value in enumerate(obj.fields[2:], start=2):
            if UUID_RE.fullmatch(value) and value not in handles:
                dangling.append((obj.obj_type, obj.name, idx, value))
    if dangling:
        errors.append(f"{len(dangling)} dangling handle references; first: {dangling[:5]}")

    stats["objects"] = len(model.objects)
    stats["spaces"] = len(model.by_type.get("OS:Space", []))
    stats["surfaces"] = len(model.by_type.get("OS:Surface", []))
    stats["subsurfaces"] = len(model.by_type.get("OS:SubSurface", []))
    stats["thermal_zones"] = len(model.by_type.get("OS:ThermalZone", []))
    return {"errors": errors, "warnings": warnings, "stats": dict(stats)}


def validate_input_geometry(path: Path) -> dict:
    model = parse_osm(path)
    errors = []
    warnings = []
    for surf in model.by_type.get("OS:Surface", []):
        pts = vertices(surf)
        area, normal = newell_area_and_normal(pts)
        if len(pts) < 3 or area <= 1e-8:
            errors.append(f"Degenerate surface {surf.name}")
        pe = planarity_error(pts)
        if pe > 1e-4:
            errors.append(f"Nonplanar surface {surf.name}: {pe:.6g} m")
    for sub in model.by_type.get("OS:SubSurface", []):
        pts = vertices(sub)
        area, _ = newell_area_and_normal(pts)
        if len(pts) < 3 or area <= 1e-8:
            errors.append(f"Degenerate subsurface {sub.name}")
        pe = planarity_error(pts)
        if pe > 1e-4:
            errors.append(f"Nonplanar subsurface {sub.name}: {pe:.6g} m")
    return {
        "errors": errors,
        "warnings": warnings,
        "counts": {t: len(model.by_type.get(t, [])) for t in sorted(PRIMARY_GEOMETRY_TYPES)},
    }


def compile_models(baseline: Path, proposed: Path, geometry: Path, outdir: Path) -> Tuple[Path, Path, Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    input_validation = validate_input_geometry(geometry)
    if input_validation["errors"]:
        raise CompileError("Updated geometry failed validation: " + "; ".join(input_validation["errors"][:10]))

    baseline_out = outdir / f"{baseline.stem}_UPDATED_GEOMETRY.osm"
    proposed_out = outdir / f"{proposed.stem}_UPDATED_GEOMETRY.osm"
    baseline_report = compile_one(baseline, geometry, baseline_out, "baseline")
    proposed_report = compile_one(proposed, geometry, proposed_out, "proposed")

    report = {
        "input_geometry_validation": input_validation,
        "baseline": baseline_report,
        "proposed": proposed_report,
        "critical_note": (
            "Static OSM integrity validation passed. Final compliance workflow must still run both models "
            "through the same OpenStudio 3.10.0/EnergyPlus weather-file workflow used for approval and review "
            "the EnergyPlus .err files before relying on results."
        ),
    }
    json_path = outdir / "geometry_compile_report.json"
    txt_path = outdir / "geometry_compile_report.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    txt_path.write_text(render_text_report(report), encoding="utf-8")
    return baseline_out, proposed_out, json_path, txt_path


def render_text_report(report: dict) -> str:
    lines = ["OPENSTUDIO OSM GEOMETRY COMPILER REPORT", "=" * 44, ""]
    iv = report["input_geometry_validation"]
    lines.append("UPDATED GEOMETRY INPUT")
    lines.append(f"  Counts: {iv['counts']}")
    lines.append(f"  Errors: {len(iv['errors'])}")
    lines.append("")
    for mode in ("baseline", "proposed"):
        r = report[mode]
        lines.append(mode.upper())
        lines.append(f"  Output: {r['output']}")
        lines.append(f"  OpenStudio version: {r['openstudio_version']}")
        lines.append(f"  Spaces: {r['spaces']['count']} {r['spaces']['zone_assignments']}")
        lines.append(f"  Zone volumes m3: {r['zone_volumes_m3']}")
        if "ffactor" in r:
            lines.append(f"  F-factor: {r['ffactor']}")
        if "infiltration" in r:
            lines.append(f"  Infiltration: {r['infiltration']}")
        lines.append(f"  Validation errors: {len(r['validation']['errors'])}")
        lines.append(f"  Validation warnings: {len(r['validation']['warnings'])}")
        for w in r.get("warnings", []):
            lines.append(f"  Warning: {w}")
        for w in r["validation"].get("warnings", []):
            lines.append(f"  Geometry warning: {w}")
        lines.append("")
    lines.append("IMPORTANT")
    lines.append("  " + report["critical_note"])
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# CLI / GUI
# -----------------------------------------------------------------------------

def run_cli(args: argparse.Namespace) -> int:
    baseline = Path(args.baseline).resolve()
    proposed = Path(args.proposed).resolve()
    geometry = Path(args.geometry).resolve()
    outdir = Path(args.outdir).resolve()
    for p in (baseline, proposed, geometry):
        if not p.is_file():
            raise CompileError(f"Input file not found: {p}")
    outputs = compile_models(baseline, proposed, geometry, outdir)
    print("Compilation complete:")
    for p in outputs:
        print(f"  {p}")
    return 0


def run_gui() -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception as exc:
        raise CompileError(f"Tkinter GUI unavailable: {exc}")

    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(
        "OSM Geometry Compiler",
        "Select the approved BASELINE model, approved PROPOSED model, then the NEW GEOMETRY OSM.\n\n"
        "The program writes new files and never overwrites the originals.",
    )
    baseline = filedialog.askopenfilename(title="Select approved BASELINE OSM", filetypes=[("OpenStudio Model", "*.osm")])
    if not baseline:
        return 1
    proposed = filedialog.askopenfilename(title="Select approved PROPOSED OSM", filetypes=[("OpenStudio Model", "*.osm")])
    if not proposed:
        return 1
    geometry = filedialog.askopenfilename(title="Select NEW GEOMETRY OSM", filetypes=[("OpenStudio Model", "*.osm")])
    if not geometry:
        return 1
    outdir = filedialog.askdirectory(title="Select output folder")
    if not outdir:
        return 1
    try:
        outputs = compile_models(Path(baseline), Path(proposed), Path(geometry), Path(outdir))
    except Exception as exc:
        messagebox.showerror("Compilation failed", f"{exc}\n\n{traceback.format_exc()}")
        return 2
    messagebox.showinfo("Compilation complete", "Created:\n\n" + "\n".join(str(p) for p in outputs))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safely replace OpenStudio OSM geometry while preserving approved templates")
    parser.add_argument("--baseline", help="Approved baseline OSM")
    parser.add_argument("--proposed", help="Approved proposed OSM")
    parser.add_argument("--geometry", help="Updated geometry OSM")
    parser.add_argument("--outdir", help="Output directory")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    supplied = [args.baseline, args.proposed, args.geometry, args.outdir]
    try:
        if any(supplied):
            if not all(supplied):
                parser.error("--baseline, --proposed, --geometry, and --outdir are all required together")
            return run_cli(args)
        return run_gui()
    except CompileError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
