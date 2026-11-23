"""
Incident maps for edacs_telegram.

geocode(location)  -> Geo or None   (LLM-extracted street address / intersection / place)
render_close(geo)  -> PNG bytes     (street-level map with the incident pinned)
render_home(geo, home) -> PNG bytes (incident + home, line between, distance/bearing)

Free services only: US Census geocoder (house numbers, TIGER ranges; US only), OSM
Nominatim (places, streets, intersections via street geometry), OSM tiles. Everything is
cached on disk; Nominatim is held to 1 request/s per its usage policy
(https://operations.osmfoundation.org/policies/nominatim/ and .../policies/tiles/).
Set `geo_region` so lookups are limited to your coverage area: LLM-extracted, misheard
street names otherwise match same-named streets in other states.
"""
import http.client
import io
import json
import logging
import math
import os
import re
import socket
import threading
import time
import urllib.parse
import urllib.request

from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("edacs_telegram.geomap")

FONT_PATHS = ["/usr/share/fonts/TTF/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
              "/usr/share/fonts/gnu-free/FreeSans.otf"]
FONT_BOLD = ["/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/usr/share/fonts/gnu-free/FreeSansBold.otf"] + FONT_PATHS

_nom_lock = threading.Lock()
_nom_next = 0.0
_cache_lock = threading.Lock()
_cache = None
CFG = {
    "cache_file": "geocache.json",
    "tile_dir": "tiles",
    "region": None,            # [south, west, north, east]; matches outside are rejected
    "state": "",               # appended to queries, e.g. "Nevada" / "NV"
    "interstates": set(),      # route numbers a bare "Highway N" means I-N (e.g. {"80"})
    "us_routes": set(),        # ... means US-N; any other number becomes "SR N"
    "force_ipv4": False,       # for hosts with AAAA records but no IPv6 route
    "user_agent": "linux-radio-stuff incident maps "
                  "(+https://github.com/dbagwell00/linux-radio-stuff)",
    "tile_url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
}


class Geo(dict):
    """{'lat', 'lon', 'approx', 'how', 'name'} — a dict so it JSON-serialises in state."""
    lat = property(lambda s: s["lat"])
    lon = property(lambda s: s["lon"])
    approx = property(lambda s: s["approx"])


def configure(**kw):
    """Override CFG keys (None values are ignored)."""
    global _cache
    for k, v in kw.items():
        if v is None:
            continue
        if k not in CFG:
            raise KeyError(f"unknown geomap setting {k!r}")
        CFG[k] = set(map(str, v)) if k in ("interstates", "us_routes") else v
        if k == "cache_file":
            _cache = None


# ---------------------------------------------------------------- http helpers
class _V4HTTPS(http.client.HTTPSConnection):
    """lin-radio has no IPv6 route but these hosts publish AAAA records: IPv4 only."""
    def connect(self):
        addr = socket.getaddrinfo(self.host, self.port, socket.AF_INET, socket.SOCK_STREAM)[0][4]
        self.sock = self._context.wrap_socket(socket.create_connection(addr, self.timeout),
                                              server_hostname=self.host)


class _V4Handler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_V4HTTPS, req)


_opener_v4 = urllib.request.build_opener(_V4Handler)
_opener = urllib.request.build_opener()


def _fetch(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": CFG["user_agent"]})
    with (_opener_v4 if CFG["force_ipv4"] else _opener).open(req, timeout=timeout) as r:
        return r.read()


def _get_json(url, timeout=20):
    return json.loads(_fetch(url, timeout))


def _nominatim(**params):
    global _nom_next
    params["format"] = "jsonv2"
    r = CFG["region"]
    if r:
        params.update(viewbox=f"{r[1]},{r[2]},{r[3]},{r[0]}", bounded=1)
    params.setdefault("limit", 5)
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(params)
    with _nom_lock:
        wait = _nom_next - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            return _get_json(url)
        except Exception as e:
            log.warning("nominatim %s failed: %s", params.get("q") or params.get("street"), e)
            return []
        finally:
            _nom_next = time.time() + 1.1


def _census(address):
    url = ("https://geocoding.geo.census.gov/geocoder/locations/onelineaddress?"
           + urllib.parse.urlencode({"address": address, "benchmark": "Public_AR_Current",
                                     "format": "json"}))
    try:
        m = _get_json(url)["result"]["addressMatches"]
    except Exception as e:
        log.warning("census geocoder failed for %r: %s", address, e)
        return None
    words = set(re.findall(r"[a-z0-9]+", address.lower()))
    given_dir = {DIRS[w] for w in words if w in DIRS}
    for a in m:
        lat, lon = a["coordinates"]["y"], a["coordinates"]["x"]
        c = a.get("addressComponents", {})
        street = set(re.findall(r"[a-z0-9]+", c.get("streetName", "").lower()))
        mdir = {DIRS.get(c.get(k, "").lower(), "") for k in ("preDirection", "suffixDirection")} - {""}
        if not _in_region(lat, lon):
            continue
        if street and not street <= words:              # fuzzy match to another street
            continue
        if given_dir and mdir and not (given_dir & mdir):  # W 5th St matched to E 5th St
            continue
        return lat, lon, a["matchedAddress"]
    return None


DIRS = {"n": "n", "north": "n", "s": "s", "south": "s", "e": "e", "east": "e",
        "w": "w", "west": "w", "ne": "ne", "nw": "nw", "se": "se", "sw": "sw"}


def _in_region(lat, lon):
    r = CFG["region"]
    return not r or (r[0] <= lat <= r[2] and r[1] <= lon <= r[3])


def _st():
    return f", {CFG['state']}" if CFG["state"] else ""


# ---------------------------------------------------------------- geocoding
def _load_cache():
    global _cache
    if _cache is None:
        try:
            _cache = json.load(open(CFG["cache_file"]))
        except Exception:
            _cache = {}
    return _cache


def _save_cache():
    tmp = CFG["cache_file"] + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_cache, f, indent=0)
    os.replace(tmp, CFG["cache_file"])


INTERSECTION = re.compile(r"\s+(?:and|&|/|at)\s+", re.I)
HOUSE = re.compile(r"^\d+[A-Za-z]?\s+\S")
HWY = re.compile(r"\b(?:(interstate|i)|(us|u\.s\.)|(sr|state route|nv|state highway)|"
                 r"(highway|hwy))[\s-]*(\d{2,3})\b(?:\s+(?:highway|hwy))?", re.I)


def _norm_hwy(text):
    """'Highway 50' -> 'US 50', 'I-80' -> 'I 80', 'Hwy 341' -> 'SR 341' (the forms OSM refs use;
    Nominatim reads a bare 'Highway 50' as house number 50)."""
    def sub(m):
        n = m.group(5)
        if m.group(1) or (m.group(4) and n in CFG["interstates"]):
            return f"I {n}"
        if m.group(2) or (m.group(4) and n in CFG["us_routes"]):
            return f"US {n}"
        return f"SR {n}"
    return HWY.sub(sub, text)


def geocode(location):
    """Best-effort lat/lon for an LLM-extracted location string; None if nothing plausible."""
    key = re.sub(r"\s+", " ", location.strip().lower())
    if not key:
        return None
    with _cache_lock:
        c = _load_cache()
        if key in c:
            return Geo(c[key]) if c[key] else None
    try:
        g = _geocode(location.strip())
    except Exception:
        log.exception("geocode %r crashed", location)
        return None            # don't cache crashes
    with _cache_lock:
        _cache[key] = dict(g) if g else None
        _save_cache()
    log.info("geocode %r -> %s", location, dict(g) if g else None)
    return g


def _geocode(loc):
    loc = _norm_hwy(re.sub(r"\([^)]*\)", "", loc)).strip(" ,")
    parts = [p.strip() for p in loc.split(",") if p.strip()]
    if not parts:
        return None
    city = parts[-1] if len(parts) > 1 else ""
    head = parts[0]
    hint = f", {city}" if city else ""

    # 1. intersection "A and B"
    if not HOUSE.match(head) and INTERSECTION.search(head):
        a, b = INTERSECTION.split(head, maxsplit=1)
        g = _intersection(a, b, city)
        if g:
            return g

    # 2. house number: Census (TIGER address ranges), then OSM address points
    if HOUSE.match(head):
        r = _census(f"{head}{hint}{_st()}")
        if r:
            return Geo(lat=r[0], lon=r[1], approx=False, how="census", name=r[2])

    # 3. Nominatim on the whole string, then on each part with the city
    tries = [loc] + [f"{p}{hint}" for p in parts[:-1] if p != head or len(parts) > 1]
    if HOUSE.match(head):                              # street without the number
        tries.append(re.sub(r"^\d+[A-Za-z]?\s+", "", head) + hint)
    seen = set()
    for q in tries:
        if q.lower() in seen:
            continue
        seen.add(q.lower())
        for r in _nominatim(q=q + _st()):
            rank = int(r.get("place_rank", 0))
            if rank < 18:                              # city / county: useless as a pin
                continue
            lat, lon = float(r["lat"]), float(r["lon"])
            exact = rank >= 26 and q == loc and not (HOUSE.match(head) and rank < 30)
            return Geo(lat=lat, lon=lon, approx=not exact, how="nominatim",
                       name=r.get("display_name", "")[:120])
    return None


def _street_lines(name, city):
    q = f"{name}, {city}{_st()}" if city else f"{name}{_st()}"
    lines = []
    for r in _nominatim(q=q, polygon_geojson=1, limit=40, layer="address", dedupe=0):
        if r.get("category") != "highway":
            continue
        g = r.get("geojson") or {}
        if g.get("type") == "LineString":
            lines.append(g["coordinates"])
        elif g.get("type") == "MultiLineString":
            lines.extend(g["coordinates"])
    return lines


def _seg_closest(p, a, b):
    """Closest point to p on segment a-b (local metres)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = dx * dx + dy * dy
    t = 0 if L == 0 else max(0, min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L))
    return a[0] + t * dx, a[1] + t * dy


def _intersection(a, b, city):
    la, lb = _street_lines(a, city), _street_lines(b, city)
    if not la or not lb:
        # one side is a place ("US 50 and Mound House"): pin the street near that place, else
        # the place itself, approximately
        place = _nominatim(q=f"{b if la else a}{', ' + city if city else ''}{_st()}")
        if place and int(place[0].get("place_rank", 0)) >= 16:
            return Geo(lat=float(place[0]["lat"]), lon=float(place[0]["lon"]), approx=True,
                       how="intersection-place", name=place[0].get("display_name", "")[:120])
        return None
    lat0 = (la[0][0][1] + lb[0][0][1]) / 2
    k = 111_320 * math.cos(math.radians(lat0))
    m = lambda c: (c[0] * k, c[1] * 111_320)           # lon/lat -> local metres
    best = None
    for A, B in ((la, lb), (lb, la)):                  # vertices of one vs segments of other
        for pa in (m(p) for line in A for p in line):
            for line in B:
                for q0, q1 in zip(line, line[1:]):
                    c = _seg_closest(pa, m(q0), m(q1))
                    d = math.hypot(pa[0] - c[0], pa[1] - c[1])
                    if best is None or d < best[0]:
                        best = (d, (pa[1] + c[1]) / 2 / 111_320, (pa[0] + c[0]) / 2 / k)
    if best and best[0] < 400:                        # OSM names often stop short of a junction
        return Geo(lat=best[1], lon=best[2], approx=best[0] > 40, how="intersection",
                   name=f"{a} & {b}")
    return None


# ---------------------------------------------------------------- geometry
def distance_mi(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(h))


def bearing(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    deg = (math.degrees(math.atan2(y, x)) + 360) % 360
    pts = "N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split()
    return pts[int((deg + 11.25) // 22.5) % 16]


def relation(geo, home):
    """'14.2 mi NNW of home' (direction from home to the incident)."""
    d = distance_mi(home[0], home[1], geo["lat"], geo["lon"])
    if d < 0.1:
        return "at home"
    return f"{d:.1f} mi {bearing(home[0], home[1], geo['lat'], geo['lon'])} of home"


# ---------------------------------------------------------------- tiles / drawing
TILE = 256


def _px(lat, lon, z):
    n = TILE * 2 ** z
    x = (lon + 180) / 360 * n
    s = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n
    return x, y


def _tile(z, x, y):
    n = 2 ** z
    x %= n
    if not 0 <= y < n:
        return Image.new("RGB", (TILE, TILE), (220, 220, 220))
    path = os.path.join(CFG["tile_dir"], str(z), str(x), f"{y}.png")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 30 * 86400:
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass
    url = CFG["tile_url"].format(z=z, x=x, y=y)
    try:
        data = _fetch(url)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "wb") as f:
            f.write(data)
        os.replace(path + ".tmp", path)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        log.warning("tile %s failed: %s", url, e)
        return Image.new("RGB", (TILE, TILE), (220, 220, 220))


def _font(size, bold=False):
    for p in (FONT_BOLD if bold else FONT_PATHS):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default(size)


class _Canvas:
    def __init__(self, clat, clon, z, w, h):
        self.z, self.w, self.h = z, w, h
        cx, cy = _px(clat, clon, z)
        self.ox, self.oy = cx - w / 2, cy - h / 2
        self.img = Image.new("RGB", (w, h))
        for tx in range(int(self.ox // TILE), int((self.ox + w) // TILE) + 1):
            for ty in range(int(self.oy // TILE), int((self.oy + h) // TILE) + 1):
                self.img.paste(_tile(z, tx, ty),
                               (int(tx * TILE - self.ox), int(ty * TILE - self.oy)))
        self.d = ImageDraw.Draw(self.img, "RGBA")

    def xy(self, lat, lon):
        x, y = _px(lat, lon, self.z)
        return x - self.ox, y - self.oy

    def pin(self, lat, lon, color, label=None, ring=0):
        x, y = self.xy(lat, lon)
        if ring:                                       # "approximate" halo
            self.d.ellipse((x - ring, y - ring, x + ring, y + ring),
                           fill=color + (40,), outline=color + (160,), width=2)
        r = 13
        self.d.polygon([(x, y), (x - r * 0.8, y - r * 1.6), (x + r * 0.8, y - r * 1.6)],
                       fill=color + (255,), outline=(255, 255, 255, 255))
        self.d.ellipse((x - r, y - 2.6 * r, x + r, y - 0.6 * r), fill=color + (255,),
                       outline=(255, 255, 255, 255), width=3)
        self.d.ellipse((x - 4, y - 1.6 * r - 4, x + 4, y - 1.6 * r + 4),
                       fill=(255, 255, 255, 255))
        if label:
            self.label(x + r + 6, y - 2.4 * r, label, _font(22, True))

    def label(self, x, y, text, font, anchor="la"):
        box = self.d.textbbox((x, y), text, font=font, anchor=anchor)
        if box[2] > self.w - 4:                        # keep on canvas
            dx = box[2] - (self.w - 4)
            x, box = x - dx, (box[0] - dx, box[1], box[2] - dx, box[3])
        if box[0] < 4:
            dx = 4 - box[0]
            x, box = x + dx, (box[0] + dx, box[1], box[2] + dx, box[3])
        self.d.rounded_rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), 6,
                                 fill=(255, 255, 255, 225), outline=(60, 60, 60, 200))
        self.d.text((x, y), text, font=font, fill=(20, 20, 20, 255), anchor=anchor)

    def banner(self, text):
        f = _font(24, True)
        self.d.rectangle((0, 0, self.w, 40), fill=(25, 25, 30, 210))
        self.d.text((12, 20), text, font=f, fill=(255, 255, 255, 255), anchor="lm")

    def png(self):
        f = _font(13)
        t = "© OpenStreetMap contributors"
        box = self.d.textbbox((0, 0), t, font=f)
        tw, th = box[2] - box[0], box[3] - box[1]
        self.d.rectangle((self.w - tw - 10, self.h - th - 8, self.w, self.h),
                         fill=(255, 255, 255, 200))
        self.d.text((self.w - tw - 5, self.h - th - 6), t, font=f, fill=(40, 40, 40, 255))
        out = io.BytesIO()
        self.img.save(out, "PNG", optimize=True)
        return out.getvalue()


RED, BLUE = (214, 39, 40), (31, 100, 200)
W, H = 960, 720


def render_close(geo, title=""):
    z = 15 if geo["approx"] else 16
    c = _Canvas(geo["lat"], geo["lon"], z, W, H)
    c.pin(geo["lat"], geo["lon"], RED, ring=70 if geo["approx"] else 0)
    if title:
        c.banner(("≈ " if geo["approx"] else "") + title)
    return c.png()


def render_home(geo, home, title=""):
    lat1, lon1, lat2, lon2 = geo["lat"], geo["lon"], home[0], home[1]
    margin = 110
    z = 16
    while z > 7:
        x1, y1 = _px(lat1, lon1, z)
        x2, y2 = _px(lat2, lon2, z)
        if abs(x1 - x2) <= W - 2 * margin and abs(y1 - y2) <= H - 2 * margin - 40:
            break
        z -= 1
    # centre on the midpoint in pixel space
    x1, y1 = _px(lat1, lon1, z)
    x2, y2 = _px(lat2, lon2, z)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2 - 20       # shift 20 px down, clear of the banner
    n = TILE * 2 ** z
    clon = mx / n * 360 - 180
    clat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * my / n))))
    c = _Canvas(clat, clon, z, W, H)
    a, b = c.xy(lat2, lon2), c.xy(lat1, lon1)
    seg = math.hypot(b[0] - a[0], b[1] - a[1])
    if seg > 1:                                        # dashed line home -> incident
        step = 14
        for i in range(0, int(seg), step * 2):
            t0, t1 = i / seg, min(1, (i + step) / seg)
            c.d.line([(a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0),
                      (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)],
                     fill=(30, 30, 30, 220), width=5)
    c.pin(lat2, lon2, BLUE, label="Home")
    c.pin(lat1, lon1, RED, label=None)
    rel = relation(geo, home)
    c.label((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 28, ("≈ " if geo["approx"] else "") + rel,
            _font(24, True), anchor="mt")
    if title:
        c.banner(title)
    return c.png()
