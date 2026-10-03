"""Route planner: greedy nearest-neighbour TSP + 2-opt refinement,
using OpenRouteService (free, OpenStreetMap-based) for geocoding and driving costs.

Setup:
    pip install requests folium
    Get a free key at https://openrouteservice.org  (sign up -> Dashboard -> Request a token)
    export ORS_API_KEY="your-key"          (Windows: set ORS_API_KEY=your-key)
"""
from __future__ import annotations

import functools
import html
import http.server
import os
import socketserver
import sys
import threading
import webbrowser
from typing import Literal, Sequence

import requests

Matrix = list                         # list[list[float]]
Metric = Literal["distance", "duration"]
Location = str
Coord = tuple                         # (lat, lng)

INF = float("inf")
MAX_MATRIX_ELEMENTS = 3500            # ORS limit per matrix request (origins x destinations)


class ORSError(RuntimeError):
    """Raised when OpenRouteService returns an error or no result."""


class ORSClient:
    """Minimal OpenRouteService client (geocoding + driving matrix)."""

    BASE_URL = "https://api.heigit.org"   # replaces the deprecated api.openrouteservice.org

    def __init__(self, api_key: str, timeout: float = 30):
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Authorization": api_key})

    def _request(self, method: str, path: str, **kwargs) -> dict:
        resp = self.session.request(method, self.BASE_URL + path, timeout=self.timeout, **kwargs)
        if not resp.ok:
            try:
                detail = resp.json().get("error", resp.text)
                if isinstance(detail, dict):
                    detail = detail.get("message", detail)
            except ValueError:
                detail = resp.text
            raise ORSError(f"OpenRouteService error {resp.status_code}: {detail}")
        return resp.json()

    def geocode(self, text: str, focus: Coord = None) -> tuple:
        """Return (lat, lng, label) for a place name.

        `focus` (lat, lng) nudges ambiguous names (e.g. several towns called "Cogon")
        towards results near that point.
        """
        params = {"api_key": self.api_key, "text": text, "size": 1}
        if focus:
            params["focus.point.lat"], params["focus.point.lon"] = focus
        data = self._request("GET", "/pelias/v1/search", params=params)
        features = data.get("features") or []
        if not features:
            raise ORSError(f"Could not find map coordinates for {text!r}")
        lng, lat = features[0]["geometry"]["coordinates"]   # GeoJSON order is [lng, lat]
        label = features[0].get("properties", {}).get("label", text)
        return lat, lng, label

    def matrix(self, coords: Sequence[Coord], metric: Metric) -> list:
        """Return the raw n x n driving matrix (metres or seconds); None = unreachable."""
        body = {
            "locations": [[lng, lat] for lat, lng in coords],   # ORS wants [lng, lat]
            "metrics": [metric],
            "units": "m",
        }
        data = self._request("POST", "/openrouteservice/v2/matrix/driving-car", json=body)
        return data[metric + "s"]                               # "distances" / "durations"

    def route_geometry(self, coords: Sequence[Coord]) -> list:
        """Return the driving path through `coords` in order, as a list of (lat, lng)."""
        body = {"coordinates": [[lng, lat] for lat, lng in coords]}
        data = self._request("POST", "/openrouteservice/v2/directions/driving-car/geojson", json=body)
        line = data["features"][0]["geometry"]["coordinates"]
        return [(lat, lng) for lng, lat in line]


def geocode_locations(client: ORSClient, locations: Sequence[Location]) -> list:
    """Look up (lat, lng) for every location, printing what each name matched.

    The first location (the start) is looked up on its own; the others are biased
    towards it so ambiguous names resolve to the town nearest the start.
    """
    coords = []
    for k, loc in enumerate(locations):
        lat, lng, label = client.geocode(loc, focus=coords[0] if k else None)
        coords.append((lat, lng))
        print(f"  {loc}  ->  {label}")
    return coords


def get_cost_matrix(client: ORSClient, coords: Sequence[Coord], metric: Metric = "distance") -> Matrix:
    """n x n driving distance (m) or duration (s); unreachable pairs become infinity."""
    n = len(coords)
    if n * n > MAX_MATRIX_ELEMENTS:
        raise ValueError(f"Too many locations ({n}); the free API allows up to 59 per route.")

    raw = client.matrix(coords, metric)
    matrix = [[INF if v is None else v for v in row] for row in raw]
    for i in range(n):
        matrix[i][i] = 0
        for j in range(n):
            if matrix[i][j] == INF:
                print(f"Warning: no route between stop {i} and stop {j}", file=sys.stderr)
    return matrix


def route_cost(route: list, matrix: Matrix) -> float:
    """Cost of a closed tour. `route` lists each city once; the return leg is added."""
    return sum(matrix[route[k]][route[(k + 1) % len(route)]] for k in range(len(route)))


def greedy_tsp(matrix: Matrix, start: int = 0) -> list:
    """Nearest-neighbour construction. Returns the route (start city not repeated)."""
    n = len(matrix)
    if not 0 <= start < n:
        raise IndexError(f"start index {start} out of range for {n} locations")

    unvisited = set(range(n)) - {start}
    route = [start]
    current = start

    while unvisited:
        nearest = min(unvisited, key=lambda c: matrix[current][c])
        if matrix[current][nearest] == INF:
            raise ValueError(
                f"Location {current} has no drivable route to any remaining stop. "
                "Check the addresses or remove the unreachable stop."
            )
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    return route


def two_opt(route: list, matrix: Matrix) -> list:
    """Improve a tour by reversing segments while that shortens it.

    The start city (index 0 of `route`) stays fixed. The full cost is recomputed for
    each candidate, so it stays correct for asymmetric road distances (one-way streets).
    """
    best = route[:]
    best_cost = route_cost(best, matrix)
    improved = True

    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                cost = route_cost(candidate, matrix)
                if cost < best_cost:
                    best, best_cost, improved = candidate, cost, True
    return best


def format_cost(value: float, metric: Metric) -> str:
    if metric == "distance":
        return f"{value / 1000:.2f} km"
    hours, rem = divmod(int(value), 3600)
    return f"{hours}h {rem // 60}m" if hours else f"{rem // 60}m"


def build_map(
    client: ORSClient,
    route: list,
    locations: Sequence[Location],
    coords: list,
    matrix: Matrix,
    metric: Metric,
    save_path: str = "route_map.html",
) -> str:
    """Create an interactive street map of the tour; returns the saved file path."""
    import folium

    order = route + [route[0]]                                  # close the loop
    name = lambda i: html.escape(str(locations[i]).split(",")[0])

    # Prefer the real driving path; fall back to straight lines if it can't be fetched
    try:
        path = client.route_geometry([coords[i] for i in order])
    except (ORSError, requests.RequestException) as exc:
        print(f"Note: couldn't fetch road geometry ({exc}); drawing straight lines instead.")
        path = [coords[i] for i in order]

    lats = [c[0] for c in coords]
    lngs = [c[1] for c in coords]
    fmap = folium.Map(location=[sum(lats) / len(lats), sum(lngs) / len(lngs)],
                      zoom_start=11, tiles=None)
    # Selectable backgrounds (switcher at the top right of the map). None need an API key.
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri", name="Esri Streets", show=False).add_to(fmap)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="Map data &copy; OpenStreetMap contributors, SRTM | Style: OpenTopoMap (CC-BY-SA)",
        name="OpenTopoMap", show=False).add_to(fmap)
    folium.PolyLine(path, color="#2563eb", weight=5, opacity=0.85).add_to(fmap)

    def icon(label: str, color: str) -> "folium.DivIcon":
        box = (f'<div style="background:{color};color:white;border-radius:50%;width:30px;'
               f'height:30px;line-height:30px;text-align:center;font-weight:bold;'
               f'border:2px solid white;box-shadow:0 0 5px rgba(0,0,0,.6)">{label}</div>')
        return folium.DivIcon(html=box, icon_size=(30, 30), icon_anchor=(15, 15))

    for step, idx in enumerate(route):
        nxt = order[step + 1]
        label = "S" if step == 0 else str(step)
        role = "Start / end" if step == 0 else f"Stop {step}"
        popup = (f"<b>{role}: {name(idx)}</b><br>"
                 f"Next: {name(nxt)} ({format_cost(matrix[idx][nxt], metric)})")
        folium.Marker(coords[idx], icon=icon(label, "#16a34a" if step == 0 else "#2563eb"),
                      tooltip=f"{label}. {name(idx)}",
                      popup=folium.Popup(popup, max_width=250)).add_to(fmap)

    folium.LayerControl().add_to(fmap)
    fmap.fit_bounds([[min(lats), min(lngs)], [max(lats), max(lngs)]], padding=(40, 40))
    fmap.save(save_path)
    full = os.path.abspath(save_path)
    print(f"Route map saved to {full}")
    return full


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve_map(map_path: str) -> None:
    """Show the map through a tiny local web server.

    A page opened straight from disk (file://) sends no 'Referer', and OpenStreetMap's
    tile servers refuse such requests (HTTP 403). Served from http://127.0.0.1 the
    browser identifies the page, so the background tiles can load.
    """
    handler = functools.partial(_QuietHandler, directory=os.path.dirname(map_path))
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as server:   # port 0 = any free port
        url = f"http://127.0.0.1:{server.server_address[1]}/{os.path.basename(map_path)}"
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"Opening {url}")
        webbrowser.open(url)
        try:
            input("Map is open in your browser. Press Enter here when you're done to exit... ")
        except (KeyboardInterrupt, EOFError):
            pass
        server.shutdown()


def prompt_for_trip() -> tuple:
    """Ask the user for the starting point, the towns to visit and the metric.

    Returns (locations, metric) where locations[0] is the starting point.
    """
    print("=== Delivery Route Planner ===")
    region = input(
        "Region/country to append for accuracy (optional, e.g. 'Philippines'): "
    ).strip()

    def with_region(name: str) -> str:
        return f"{name}, {region}" if region else name

    start = ""
    while not start:
        start = input("\nStarting point (where the driver begins and ends): ").strip()
    locations = [with_region(start)]

    print("\nEnter each town/stop the driver must visit, one per line.")
    print("Press Enter on an empty line when you're done.")
    while True:
        stop = input(f"  Stop {len(locations)}: ").strip()
        if not stop:
            if len(locations) < 2:
                print("  Please enter at least one stop.")
                continue
            break
        full = with_region(stop)
        if full.lower() in (loc.lower() for loc in locations):
            print("  Already added, skipping.")
            continue
        locations.append(full)

    choice = input("\nOptimise for [d]istance or [t]ime? (d/t, default d): ").strip().lower()
    metric: Metric = "duration" if choice.startswith("t") else "distance"

    print(f"\n{len(locations) - 1} stop(s) plus the starting point, optimising by {metric}.")
    return locations, metric


def main() -> None:
    api_key = os.environ.get("ORS_API_KEY")
    if not api_key:
        sys.exit("Set the ORS_API_KEY environment variable first.")

    locations, metric = prompt_for_trip()
    start = 0
    client = ORSClient(api_key)

    try:
        print("Looking up town coordinates (check each one matched the right place):")
        coords = geocode_locations(client, locations)
        print("Fetching driving matrix from OpenRouteService...")
        matrix = get_cost_matrix(client, coords, metric)
        greedy = greedy_tsp(matrix, start)
    except (ORSError, requests.RequestException) as exc:
        sys.exit(f"API error: {exc}")
    except ValueError as exc:
        sys.exit(str(exc))

    greedy_cost = route_cost(greedy, matrix)
    final = two_opt(greedy, matrix)
    final_cost = route_cost(final, matrix)

    print("\n" + "=" * 50)
    print("GREEDY ROUTE (this is the route shown on the map)")
    print("=" * 50)
    for step, idx in enumerate(greedy):
        print(f"{'Start' if step == 0 else f'Stop {step}'}: {locations[idx]}")
    print(f"End:   {locations[greedy[0]]} (back to start)")
    print("-" * 50)
    print(f"Greedy tour (on map):      {format_cost(greedy_cost, metric)}")
    print(f"After 2-opt (not on map):  {format_cost(final_cost, metric)}")
    print("=" * 50)

    try:
        print("\nBuilding route map...")
        map_path = build_map(client, greedy, locations, coords, matrix, metric)
        serve_map(map_path)
    except ImportError:
        print("Map skipped: install folium with 'pip install folium'.")


if __name__ == "__main__":
    main()