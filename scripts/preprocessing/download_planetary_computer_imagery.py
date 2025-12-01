"""
Download imagery for:
1. Retail activity (malls)
2. Port congestion
3. Industrial zones
4. City-level macro activity

Datasets:
- US retail  -> NAIP  (1m, asset key: "image")
- Everything else -> Sentinel-2 L2A (10m, asset key: "visual")
"""

from pathlib import Path
from typing import Dict, List

import requests
from pystac_client import Client
import planetary_computer
import time

# ------------------------
#  CONFIG
# ------------------------

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Years you care about: pre-COVID, pre-COVID2, COVID, post-COVID
YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

# Rough buffer size around each point (in degrees ~ km/111)
BUFFER_DEG = 0.03  # ~3–4 km around the point

# Output root
OUT_ROOT = Path("data/raw/satellite").resolve()

OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ------------------------
#  LOCATION DEFINITIONS
# ------------------------
# You can add more or tweak names if geocoding fails.

LOCATIONS: Dict[str, Dict] = {
    # 1. RETAIL ACTIVITY
    "Mall_of_America": {
        "name": "Mall of America, Bloomington, Minnesota, USA",
        "country": "USA",
        "category": "retail",
        "lat": 44.8548,
        "lon": -93.2422,
    },
    "Westfield_Century_City": {
        "name": "Westfield Century City, Los Angeles, California, USA",
        "country": "USA",
        "category": "retail",
        "lat": 34.0575,
        "lon": -118.4166,
    },
    "The_Grove_LA": {
        "name": "The Grove, Los Angeles, California, USA",
        "country": "USA",
        "category": "retail",
        "lat": 34.0719,
        "lon": -118.3567,
    },
    "South_Coast_Plaza": {
        "name": "South Coast Plaza, Costa Mesa, California, USA",
        "country": "USA",
        "category": "retail",
        "lat": 33.6905,
        "lon": -117.8867,
    },
    "Tysons_Corner_Center": {
        "name": "Tysons Corner Center, Tysons, Virginia, USA",
        "country": "USA",
        "category": "retail",
        "lat": 38.9188,
        "lon": -77.2217,
    },
    "King_of_Prussia": {
        "name": "King of Prussia Mall, King of Prussia, Pennsylvania, USA",
        "country": "USA",
        "category": "retail",
        "lat": 40.0892,
        "lon": -75.3958,
    },
    "Roosevelt_Field": {
        "name": "Roosevelt Field Mall, Garden City, New York, USA",
        "country": "USA",
        "category": "retail",
        "lat": 40.7375,
        "lon": -73.6111,
    },

    "Westfield_London": {
        "name": "Westfield London, London, UK",
        "country": "UK",
        "category": "retail",
        "lat": 51.5074,
        "lon": -0.2208,
    },
    "Bluewater": {
        "name": "Bluewater Shopping Centre, Dartford, UK",
        "country": "UK",
        "category": "retail",
        "lat": 51.4389,
        "lon": 0.2667,
    },
    "Trafford_Centre": {
        "name": "Trafford Centre, Manchester, UK",
        "country": "UK",
        "category": "retail",
        "lat": 53.4667,
        "lon": -2.3458,
    },

    "Galeries_Lafayette": {
        "name": "Galeries Lafayette, Paris, France",
        "country": "France",
        "category": "retail",
        "lat": 48.8738,
        "lon": 2.3320,
    },
    "La_Maquinista": {
        "name": "La Maquinista, Barcelona, Spain",
        "country": "Spain",
        "category": "retail",
        "lat": 41.4358,
        "lon": 2.1989,
    },
    "Centro_Oberhausen": {
        "name": "Centro, Oberhausen, Germany",
        "country": "Germany",
        "category": "retail",
        "lat": 51.4906,
        "lon": 6.8783,
    },

    "Pacific_Mall_Delhi": {
        "name": "Pacific Mall, Tagore Garden, New Delhi, India",
        "country": "India",
        "category": "retail",
        "lat": 28.6417,
        "lon": 77.1167,
    },
    "Select_Citywalk": {
        "name": "Select CITYWALK, Saket, New Delhi, India",
        "country": "India",
        "category": "retail",
        "lat": 28.5244,
        "lon": 77.2167,
    },
    "Mall_of_Asia_Manila": {
        "name": "SM Mall of Asia, Manila, Philippines",
        "country": "Philippines",
        "category": "retail",
        "lat": 14.5364,
        "lon": 120.9822,
    },

    # 2. PORT CONGESTION
    "Port_of_Los_Angeles": {
        "name": "Port of Los Angeles, California, USA",
        "country": "USA",
        "category": "port",
        "lat": 33.7406,
        "lon": -118.2728,
    },
    "Port_of_Long_Beach": {
        "name": "Port of Long Beach, California, USA",
        "country": "USA",
        "category": "port",
        "lat": 33.7544,
        "lon": -118.1933,
    },
    "Port_of_New_York_New_Jersey": {
        "name": "Port of New York and New Jersey, USA",
        "country": "USA",
        "category": "port",
        "lat": 40.6694,
        "lon": -74.0447,
    },
    "Port_of_Savannah": {
        "name": "Port of Savannah, Georgia, USA",
        "country": "USA",
        "category": "port",
        "lat": 32.1361,
        "lon": -81.1428,
    },
    "Port_of_Houston": {
        "name": "Port of Houston, Texas, USA",
        "country": "USA",
        "category": "port",
        "lat": 29.7342,
        "lon": -95.2656,
    },

    "Port_of_Rotterdam": {
        "name": "Port of Rotterdam, Netherlands",
        "country": "Netherlands",
        "category": "port",
        "lat": 51.9500,
        "lon": 4.1400,
    },
    "Port_of_Antwerp": {
        "name": "Port of Antwerp-Bruges, Belgium",
        "country": "Belgium",
        "category": "port",
        "lat": 51.2833,
        "lon": 4.3167,
    },
    "Port_of_Hamburg": {
        "name": "Port of Hamburg, Germany",
        "country": "Germany",
        "category": "port",
        "lat": 53.5394,
        "lon": 9.9733,
    },
    "Port_of_Valencia": {
        "name": "Port of Valencia, Spain",
        "country": "Spain",
        "category": "port",
        "lat": 39.4500,
        "lon": -0.3217,
    },

    "Port_of_Shanghai": {
        "name": "Port of Shanghai, China",
        "country": "China",
        "category": "port",
        "lat": 31.2304,
        "lon": 121.4737,
    },
    "Port_of_Ningbo": {
        "name": "Port of Ningbo-Zhoushan, China",
        "country": "China",
        "category": "port",
        "lat": 29.8683,
        "lon": 121.5440,
    },
    "Port_of_Singapore": {
        "name": "Port of Singapore, Singapore",
        "country": "Singapore",
        "category": "port",
        "lat": 1.2644,
        "lon": 103.8220,
    },
    "Port_of_Busan": {
        "name": "Port of Busan, South Korea",
        "country": "South Korea",
        "category": "port",
        "lat": 35.1028,
        "lon": 129.0403,
    },
    "Port_of_Hong_Kong": {
        "name": "Port of Hong Kong, Hong Kong",
        "country": "Hong Kong",
        "category": "port",
        "lat": 22.2908,
        "lon": 114.1501,
    },

    "Port_of_Durban": {
        "name": "Port of Durban, South Africa",
        "country": "South Africa",
        "category": "port",
        "lat": -29.8587,
        "lon": 31.0218,
    },
    "Port_of_Mombasa": {
        "name": "Port of Mombasa, Kenya",
        "country": "Kenya",
        "category": "port",
        "lat": -4.0435,
        "lon": 39.6682,
    },
    "Port_of_Lagos": {
        "name": "Port of Apapa, Lagos, Nigeria",
        "country": "Nigeria",
        "category": "port",
        "lat": 6.4474,
        "lon": 3.3903,
    },

    "Port_of_Jebel_Ali": {
        "name": "Port of Jebel Ali, Dubai, UAE",
        "country": "UAE",
        "category": "port",
        "lat": 25.0133,
        "lon": 55.0333,
    },
    "Port_of_Salalah": {
        "name": "Port of Salalah, Oman",
        "country": "Oman",
        "category": "port",
        "lat": 16.9392,
        "lon": 54.0058,
    },

    # 3. INDUSTRIAL ACTIVITY
    "Shenzhen_Electronics": {
        "name": "Shenzhen electronics manufacturing zone, Shenzhen, China",
        "country": "China",
        "category": "industrial",
        "lat": 22.5431,
        "lon": 114.0579,
    },
    "Suzhou_Industrial_Park": {
        "name": "Suzhou Industrial Park, Suzhou, China",
        "country": "China",
        "category": "industrial",
        "lat": 31.3017,
        "lon": 120.7378,
    },
    "Pune_Hinjawadi": {
        "name": "Hinjawadi IT Park, Pune, India",
        "country": "India",
        "category": "industrial",
        "lat": 18.5912,
        "lon": 73.7389,
    },
    "Detroit_Auto": {
        "name": "Detroit auto manufacturing plants, Detroit, Michigan, USA",
        "country": "USA",
        "category": "industrial",
        "lat": 42.3314,
        "lon": -83.0458,
    },
    "Tijuana_Manufacturing": {
        "name": "Industrial zone, Tijuana, Mexico",
        "country": "Mexico",
        "category": "industrial",
        "lat": 32.5149,
        "lon": -117.0382,
    },

    # 4. CITY-LEVEL MACRO FOOTPRINT
    "Los_Angeles": {
        "name": "Los Angeles, California, USA",
        "country": "USA",
        "category": "city",
        "lat": 34.0522,
        "lon": -118.2437,
    },
    "New_York_City": {
        "name": "New York City, USA",
        "country": "USA",
        "category": "city",
        "lat": 40.7128,
        "lon": -74.0060,
    },
    "Chicago": {
        "name": "Chicago, Illinois, USA",
        "country": "USA",
        "category": "city",
        "lat": 41.8781,
        "lon": -87.6298,
    },
    "London": {
        "name": "London, UK",
        "country": "UK",
        "category": "city",
        "lat": 51.5074,
        "lon": -0.1278,
    },
    "Paris": {
        "name": "Paris, France",
        "country": "France",
        "category": "city",
        "lat": 48.8566,
        "lon": 2.3522,
    },
    "Tokyo": {
        "name": "Tokyo, Japan",
        "country": "Japan",
        "category": "city",
        "lat": 35.6762,
        "lon": 139.6503,
    },
    "Beijing": {
        "name": "Beijing, China",
        "country": "China",
        "category": "city",
        "lat": 39.9042,
        "lon": 116.4074,
    },
    "Mumbai": {
        "name": "Mumbai, India",
        "country": "India",
        "category": "city",
        "lat": 19.0760,
        "lon": 72.8777,
    },
    "Johannesburg": {
        "name": "Johannesburg, South Africa",
        "country": "South Africa",
        "category": "city",
        "lat": -26.2041,
        "lon": 28.0473,
    },
    "Sao_Paulo": {
        "name": "São Paulo, Brazil",
        "country": "Brazil",
        "category": "city",
        "lat": -23.5505,
        "lon": -46.6333,
    },
}


# ------------------------
#  HELPERS
# ------------------------

def choose_collection(country: str, category: str) -> str:
    """
    Decide which Planetary Computer collection to use.
    - US retail -> NAIP (1 m aerial imagery)
    - otherwise -> Sentinel-2 L2A (global, 10 m)
    """
    if country.upper() == "USA" and category == "retail":
        return "naip"
    return "sentinel-2-l2a"


def compute_bbox(lat: float, lon: float, buffer_deg: float) -> List[float]:
    """
    Compute a bounding box [min_lon, min_lat, max_lon, max_lat] from lat/lon.
    """
    min_lon = lon - buffer_deg
    max_lon = lon + buffer_deg
    min_lat = lat - buffer_deg
    max_lat = lat + buffer_deg

    return [min_lon, min_lat, max_lon, max_lat]


def download_asset(asset_href: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↳ Downloading to {out_path}")
    with requests.get(asset_href, stream=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def download_for_location(
    catalog: Client,
    loc_key: str,
    loc_cfg: Dict,
    years: List[int],
):
    place_name = loc_cfg["name"]
    country = loc_cfg["country"]
    category = loc_cfg["category"]
    lat = loc_cfg["lat"]
    lon = loc_cfg["lon"]

    collection_id = choose_collection(country, category)
    print(f"\n=== {loc_key} | {place_name} | {collection_id} ===")

    # Compute bbox from hardcoded lat/lon
    bbox = compute_bbox(lat, lon, BUFFER_DEG)
    print(f"  BBOX: {bbox}")

    for year in years:
        # Compute output path first so we can resume by skipping existing files
        out_path = OUT_ROOT / collection_id / str(year) / f"{loc_key}_{collection_id}_{year}.tif"
        if out_path.exists():
            print(f"  Year {year} | already downloaded at {out_path}, skipping.")
            continue

        time_range = f"{year}-01-01/{year}-12-31"
        print(f"  Year {year} | datetime={time_range}")

        search_kwargs = {
            "collections": [collection_id],
            "bbox": bbox,
            "datetime": time_range,
            "max_items": 1,
        }

        # Only Sentinel-2 has eo:cloud_cover; NAIP does not.
        if collection_id == "sentinel-2-l2a":
            search_kwargs["query"] = {"eo:cloud_cover": {"lt": 20}}
            search_kwargs["sortby"] = ["eo:cloud_cover"]

        search = catalog.search(**search_kwargs)
        items = list(search.get_items())

        if not items:
            print("    ⚠ No items found for this year.")
            continue

        item = items[0]
        print(f"    Using item: {item.id}")

        # Sign item URLs so we can actually read assets :contentReference[oaicite:4]{index=4}
        planetary_computer.sign_inplace(item)

        if collection_id == "naip":
            asset_key = "image"  # NAIP COG with RGBIR bands :contentReference[oaicite:5]{index=5}
        else:
            asset_key = "visual"  # Sentinel-2 3-band RGB TCI :contentReference[oaicite:6]{index=6}

        asset = item.assets.get(asset_key)
        if asset is None:
            print(f"    ⚠ Asset '{asset_key}' not found, available: {list(item.assets.keys())}")
            continue

        href = asset.href
        download_asset(href, out_path)

        # Be kind to APIs
        time.sleep(1)


# ------------------------
#  MAIN
# ------------------------

if __name__ == "__main__":
    catalog = Client.open(
        STAC_URL,
        modifier=planetary_computer.sign_inplace,  # sign STAC assets in-place
    )

    print("Connected to:", catalog.title)

    for loc_key, loc_cfg in LOCATIONS.items():
        try:
            download_for_location(catalog, loc_key, loc_cfg, YEARS)
        except Exception as e:
            print(f"❌ Error for {loc_key}: {e}")
            time.sleep(1)
