"""Taxi routing over an airport's apt.dat taxi network (rows 1201/1202/1204).

Only airports that ship a taxi network (most busier airports from the X-Plane
Scenery Gateway, many custom sceneries) get routes; others fall back to a
generic instruction.
"""

from __future__ import annotations

import heapq
import math
import re
import logging
from dataclasses import dataclass, field

from .airports import bearing, dist_nm
from .phraseology import spell

log = logging.getLogger(__name__)

M_PER_NM = 1852.0
MIN_SEGMENT_M = 60.0


@dataclass
class Edge:
    to: int
    dist_m: float
    name: str
    runway: bool
    dep_zone: set[str] = field(default_factory=set)  # runway ends this edge is a departure hot zone for
    zone: set[str] = field(default_factory=set)  # runway ends of any hot zone (departure/arrival/ils)


@dataclass
class Ramp:
    lat: float
    lon: float
    kind: str  # gate, hangar, tie_down, misc
    name: str


@dataclass
class Network:
    nodes: dict[int, tuple[float, float]]
    adj: dict[int, list[Edge]]
    ramps: list[Ramp] = field(default_factory=list)


@dataclass
class TaxiRoute:
    runway: str
    via: list[str]  # taxiway names in order, e.g. ["B", "A"]
    crossings: list[str]  # runways crossed, e.g. ["17/35"]
    backtaxi: bool  # route uses the departure runway itself
    distance_m: float

    def describe(self) -> str:
        dest = f"runway {self.runway}" if self.runway else "parking"
        if not self.via and self.distance_m < 150:
            return f"TAXI ROUTE to {dest}: it is right there, no taxiways needed (just \"taxi to {dest}\")"
        via = ", ".join(spoken_taxiway(n) for n in self.via) or "(no named taxiways)"
        parts = [f"TAXI ROUTE to {dest}: via {via}"]
        if self.crossings:
            parts.append("crossing runway " + ", ".join(self.crossings)
                         + " (FAA: include an explicit \"cross runway ...\" for each)")
        else:
            parts.append("runway crossings: none")
        if self.backtaxi:
            parts.append(f"route uses runway {self.runway} itself (back-taxi on the runway)")
        return "; ".join(parts)


def spoken_taxiway(name: str) -> str:
    n = name.strip()
    for prefix in ("TWY ", "TAXIWAY ", "Taxiway "):
        if n.upper().startswith(prefix.upper()):
            n = n[len(prefix):]
    if len(n) <= 4 and n.replace(" ", "").isalnum():
        return spell(n)
    return n


_cache: dict[str, Network | None] = {}


def load_network(airport: dict) -> Network | None:
    if not airport.get("taxi") or "file" not in airport:
        return None
    key = f"{airport['file']}:{airport['offset']}"
    if key in _cache:
        return _cache[key]
    nodes: dict[int, tuple[float, float]] = {}
    edges: list[tuple[int, int, bool, Edge]] = []  # (from, to, twoway, edge)
    ramps: list[Ramp] = []
    try:
        with open(airport["file"], "rb") as fh:
            fh.seek(airport["offset"])
            fh.readline()  # the airport header
            for raw in fh:
                parts = raw.decode("latin-1").split()
                if not parts or not parts[0].isdigit():
                    continue
                code = int(parts[0])
                if code in (1, 16, 17, 99):
                    break
                if code == 1300 and len(parts) >= 5:
                    ramps.append(Ramp(float(parts[1]), float(parts[2]), parts[4], " ".join(parts[6:])))
                elif code == 1201 and len(parts) >= 5:
                    nodes[int(parts[4])] = (float(parts[1]), float(parts[2]))
                elif code == 1202 and len(parts) >= 3:
                    a, b = int(parts[1]), int(parts[2])
                    twoway = len(parts) < 4 or parts[3] != "oneway"
                    kind = parts[4] if len(parts) > 4 else "taxiway"
                    name = " ".join(parts[5:])
                    edges.append((a, b, twoway, Edge(b, 0.0, name, kind == "runway")))
                elif code == 1204 and len(parts) >= 3 and edges:
                    ends = {r.strip() for r in parts[2].split(",") if r.strip()}
                    e = edges[-1][3]
                    e.zone |= ends
                    if parts[1] == "departure":
                        e.dep_zone |= ends
    except OSError as ex:
        log.warning("Can't read taxi network for %s: %s", airport["id"], ex)
        _cache[key] = None
        return None
    adj: dict[int, list[Edge]] = {n: [] for n in nodes}
    for a, b, twoway, e in edges:
        if a not in nodes or b not in nodes:
            continue
        (la, lo), (lb, lob) = nodes[a], nodes[b]
        e.dist_m = dist_nm(la, lo, lb, lob) * M_PER_NM
        adj[a].append(e)
        if twoway:
            adj[b].append(Edge(a, e.dist_m, e.name, e.runway, e.dep_zone, e.zone))
    net = Network(nodes, adj, ramps) if edges else None
    _cache[key] = net
    return net


def _runway_pair(airport: dict, end: str) -> str:
    for r in airport.get("runways", []):
        if end in r["ids"]:
            return "/".join(r["ids"])
    return end


def _m(p, q) -> float:
    return dist_nm(p[0], p[1], q[0], q[1]) * M_PER_NM


def _taxi_nodes(net: Network) -> list[int]:
    return [n for n, es in net.adj.items() if any(not e.runway for e in es)]


def _dijkstra(net: Network, start: int, avoid: set[str]):
    """Shortest paths; runway edges and hot zones of runways not in `avoid`'s complement cost extra."""
    dist = {start: 0.0}
    prev: dict[int, tuple[int, Edge]] = {}
    pq = [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        for e in net.adj.get(u, []):
            w = e.dist_m * (6 if e.runway else 1) * (2 if e.zone - avoid else 1)
            nd = d + w
            if nd < dist.get(e.to, float("inf")):
                dist[e.to] = nd
                prev[e.to] = (u, e)
                heapq.heappush(pq, (nd, e.to))
    return dist, prev


def _summarize(airport: dict, path: list[Edge], same_runway: set[str]) -> tuple[list[str], list[str], bool]:
    via: list[list] = []  # [name, metres]
    crossings: list[str] = []
    backtaxi = False
    for e in path:
        if e.runway:
            pair = e.name or _runway_pair(airport, next(iter(e.zone), ""))
            if set(pair.split("/")) & same_runway:
                backtaxi = True
            elif pair not in crossings:
                crossings.append(pair)
            continue
        for end in sorted(e.zone - same_runway):
            pair = _runway_pair(airport, end)
            if pair not in crossings and not (set(pair.split("/")) & same_runway):
                crossings.append(pair)
        name = e.name.strip()
        if not name:
            continue
        if via and via[-1][0] == name:
            via[-1][1] += e.dist_m
        else:
            via.append([name, e.dist_m])
    # Controllers skip short ramp lanes and connectors; keep the last segment regardless
    names = [n for i, (n, d) in enumerate(via) if d >= MIN_SEGMENT_M or i == len(via) - 1]
    names = [n for i, n in enumerate(names) if i == 0 or names[i - 1] != n]
    return names, crossings, backtaxi


def _path(prev, start: int, goal: int) -> list[Edge]:
    path: list[Edge] = []
    n = goal
    while n != start:
        p, e = prev[n]
        path.append(e)
        n = p
    path.reverse()
    return path


def route_to_runway(airport: dict, lat: float, lon: float, runway: str) -> TaxiRoute | None:
    """Route from the aircraft to the hold-short point of `runway` (an end id like '31')."""
    net = load_network(airport)
    if not net:
        return None
    rw = next((r for r in airport.get("runways", []) if runway in r["ids"]), None)
    if not rw:
        return None
    same_runway = set(rw["ids"])
    thr = (rw["lat1"], rw["lon1"]) if rw["ids"][0] == runway else (rw["lat2"], rw["lon2"])
    taxi_nodes = _taxi_nodes(net)
    if not taxi_nodes:
        return None
    start = min(taxi_nodes, key=lambda n: _m(net.nodes[n], (lat, lon)))
    dist, prev = _dijkstra(net, start, same_runway)

    # Hold points: outer node of an edge in the departure zone of this runway end
    cands = set()
    for u, es in net.adj.items():
        if any(not e.zone and not e.runway for e in es):
            if any(runway in e.dep_zone for e in es):
                cands.add(u)
    if not cands:  # no hot-zone data: any taxiway node near the threshold
        cands = {n for n in taxi_nodes if _m(net.nodes[n], thr) < 400}
    cands = {c for c in cands if c in dist}
    if not cands:
        return None
    goal = min(cands, key=lambda c: dist[c] + 3 * _m(net.nodes[c], thr))
    path = _path(prev, start, goal)
    names, crossings, backtaxi = _summarize(airport, path, same_runway)
    return TaxiRoute(runway, names, crossings, backtaxi, sum(e.dist_m for e in path))


PARKING_WORDS = ("park", "ramp", "fbo", "transient", "tie down", "tiedown", "tie-down", "hangar", "gate")


def pick_ramp(net: Network, transcript: str, lat: float, lon: float) -> Ramp | None:
    """The parking spot the pilot asked for (by name, e.g. 'Signature'), else the nearest
    tie-down / misc spot, else the nearest of any kind."""
    if not net.ramps:
        return None
    low = transcript.lower()
    usable = [r for r in net.ramps if not re.search(r"run.?up|hold|heli|deice|de-ice|wash", r.name, re.I)]
    named = [r for r in usable if r.name and any(
        w in low for w in r.name.lower().replace("_", " ").split() if len(w) >= 5)]
    pool = (named or [r for r in usable if r.kind == "tie_down"] or [r for r in usable if r.kind == "misc"]
            or usable or net.ramps)
    return min(pool, key=lambda r: _m((r.lat, r.lon), (lat, lon)))


def route_to_parking(airport: dict, lat: float, lon: float, transcript: str = "") -> tuple[TaxiRoute, Ramp] | None:
    net = load_network(airport)
    if not net:
        return None
    ramp = pick_ramp(net, transcript, lat, lon)
    taxi_nodes = _taxi_nodes(net)
    if not ramp or not taxi_nodes:
        return None
    start = min(taxi_nodes, key=lambda n: _m(net.nodes[n], (lat, lon)))
    goal = min(taxi_nodes, key=lambda n: _m(net.nodes[n], (ramp.lat, ramp.lon)))
    # The runway we're on (just landed / still rolling out) isn't a crossing
    here = set()
    for r in airport.get("runways", []):
        if _off_centerline_m(r, lat, lon) < 60:
            here |= set(r["ids"])
    dist, prev = _dijkstra(net, start, here)
    if goal not in dist:
        return None
    path = _path(prev, start, goal)
    names, crossings, _ = _summarize(airport, path, here)
    return TaxiRoute("", names, crossings, False, sum(e.dist_m for e in path)), ramp


def _off_centerline_m(r: dict, lat: float, lon: float) -> float:
    """Distance from a point to a runway's centerline segment, metres (flat-earth, fine at airport scale)."""
    k = math.cos(math.radians(lat))
    ax, ay = r["lon1"] * k, r["lat1"]
    bx, by = r["lon2"] * k, r["lat2"]
    px, py = lon * k, lat
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / ((dx * dx + dy * dy) or 1e-12)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy)) * 60 * M_PER_NM


def on_runway(airport: dict, lat: float, lon: float) -> dict | None:
    for r in airport.get("runways", []):
        if _off_centerline_m(r, lat, lon) < 30:
            return r
    return None


def runway_exit(airport: dict, lat: float, lon: float, hdg_true: float) -> tuple[str, str] | None:
    """Next taxiway off the runway ahead of the aircraft: (taxiway name, 'left'/'right')."""
    net = load_network(airport)
    if not net:
        return None
    best = None
    for u, es in net.adj.items():
        if not any(e.runway for e in es):
            continue  # exits start at runway nodes
        p = net.nodes[u]
        d = _m(p, (lat, lon))
        if d > 1600:
            continue
        ahead = math.radians((bearing(lat, lon, p[0], p[1]) - hdg_true + 540) % 360 - 180)
        if d > 30 and (math.cos(ahead) < 0 or d * abs(math.sin(ahead)) > 45):
            continue  # behind us, or not on our runway (e.g. a parallel)
        for e in es:
            if e.runway or not e.name.strip():
                continue
            q = net.nodes[e.to]
            side = (bearing(p[0], p[1], q[0], q[1]) - hdg_true + 540) % 360 - 180
            if 20 < abs(side) < 160 and (best is None or d < best[0]):
                best = (d, e.name.strip(), "right" if side > 0 else "left")
    return (best[1], best[2]) if best else None
