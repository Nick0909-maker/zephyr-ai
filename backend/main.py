import asyncio
import math
import time
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# CONFIGURATION
# ============================================================

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
ECMWF_URL = "https://api.open-meteo.com/v1/ecmwf"
ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

CACHE_SECONDS = 60
INTELLIGENCE_CACHE_SECONDS = 300

# ============================================================
# LOCATIONS
# ============================================================

LOCATIONS = {
    "chamoli": {
        "name": "Chamoli / Joshimath",
        "city": "Joshimath",
        "state": "Uttarakhand",
        "latitude": 30.5669,
        "longitude": 79.5640,
    },
    "shimla": {
        "name": "Shimla",
        "city": "Shimla",
        "state": "Himachal Pradesh",
        "latitude": 31.1048,
        "longitude": 77.1734,
    },
    "guwahati": {
        "name": "Guwahati",
        "city": "Guwahati",
        "state": "Assam",
        "latitude": 26.1445,
        "longitude": 91.7362,
    },
    "mumbai": {
        "name": "Mumbai",
        "city": "Mumbai",
        "state": "Maharashtra",
        "latitude": 19.0760,
        "longitude": 72.8777,
    },
    "puri": {
        "name": "Puri",
        "city": "Puri",
        "state": "Odisha",
        "latitude": 19.8135,
        "longitude": 85.8312,
    },
}

# ============================================================
# WMO WEATHER CODES
# ============================================================

WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall",
    77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
    82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}

# ============================================================
# GLOBAL STATE
# ============================================================

http_client: httpx.AsyncClient | None = None

cache = {"timestamp": 0.0, "data": None}
intelligence_cache: dict[str, dict[str, Any]] = {}
cache_lock = asyncio.Lock()

# ============================================================
# FASTAPI LIFESPAN / APP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(20.0, connect=10.0),
        headers={"User-Agent": "ZephyrAI/2.0 weather-dashboard"},
        follow_redirects=True,
    )

    print("==========================================")
    print("        ZEPHYR AI BACKEND STARTED")
    print("==========================================")
    print("Weather source : Open-Meteo")
    print("Intelligence   : ECMWF + Ensemble")
    print("API server     : http://127.0.0.1:8000")
    print("==========================================")

    yield

    await http_client.aclose()
    http_client = None


app = FastAPI(
    title="Zephyr AI Weather Intelligence API",
    description="Coordinate-based weather intelligence using Open-Meteo, ECMWF and ensemble forecasts.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:5500",
                "https://zephyr-ai-1-qdh7.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# BASIC HELPERS
# ============================================================

def weather_description(code: int | None) -> str:
    if code is None:
        return "Unknown"
    return WEATHER_CODES.get(code, f"Weather condition code {code}")


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
        if not math.isfinite(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def validate_coordinates(latitude: float, longitude: float) -> None:
    if not -90 <= latitude <= 90:
        raise HTTPException(status_code=400, detail="Latitude must be between -90 and 90.")
    if not -180 <= longitude <= 180:
        raise HTTPException(status_code=400, detail="Longitude must be between -180 and 180.")


def get_risk_level(
    weather_code: int | None,
    precipitation_probability: float | None,
    precipitation: float | None,
    wind_gust: float | None = None,
) -> str:
    if weather_code in [95, 96, 99]:
        return "HIGH"
    if weather_code in [65, 67, 82, 86]:
        return "HIGH"
    if wind_gust is not None and wind_gust >= 60:
        return "HIGH"
    if precipitation_probability is not None and precipitation_probability >= 70:
        return "MODERATE"
    if precipitation is not None and precipitation >= 10:
        return "MODERATE"
    if weather_code in [63, 81, 80]:
        return "MODERATE"
    return "LOW"


def extract_hourly_forecast(hourly: dict[str, Any], limit: int = 24) -> list[dict[str, Any]]:
    times = hourly.get("time", [])
    temperatures = hourly.get("temperature_2m", [])
    rain_probability = hourly.get("precipitation_probability", [])
    precipitation = hourly.get("precipitation", [])
    weather_codes = hourly.get("weather_code", [])
    wind_gusts = hourly.get("wind_gusts_10m", [])

    result = []
    limit = min(limit, len(times))

    for i in range(limit):
        code = weather_codes[i] if i < len(weather_codes) else None
        result.append({
            "time": times[i],
            "temperature_c": temperatures[i] if i < len(temperatures) else None,
            "precipitation_probability": rain_probability[i] if i < len(rain_probability) else None,
            "precipitation_mm": precipitation[i] if i < len(precipitation) else None,
            "wind_gust_kmh": wind_gusts[i] if i < len(wind_gusts) else None,
            "weather_code": code,
            "condition": weather_description(code),
        })

    return result

# ============================================================
# FORECAST FETCH
# ============================================================

async def fetch_weather_coordinates(
    latitude: float,
    longitude: float,
    name: str | None = None,
    city: str | None = None,
    state: str | None = None,
) -> dict[str, Any]:
    if http_client is None:
        raise RuntimeError("HTTP client is not initialized.")

    validate_coordinates(latitude, longitude)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join([
            "temperature_2m", "relative_humidity_2m", "apparent_temperature",
            "precipitation", "rain", "cloud_cover", "pressure_msl",
            "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "weather_code",
        ]),
        "hourly": ",".join([
            "temperature_2m", "precipitation_probability", "precipitation",
            "rain", "wind_gusts_10m", "weather_code",
        ]),
        "daily": ",".join([
            "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
            "precipitation_probability_max", "weather_code", "wind_gusts_10m_max",
        ]),
        "forecast_days": 3,
        "timezone": "auto",
    }

    response = await http_client.get(OPEN_METEO_URL, params=params)
    response.raise_for_status()
    data = response.json()

    current = data.get("current", {})
    current_units = data.get("current_units", {})
    hourly = data.get("hourly", {})
    daily = data.get("daily", {})

    weather_code = current.get("weather_code")
    precipitation_probability = (
        hourly.get("precipitation_probability", [None])[0]
        if hourly.get("precipitation_probability") else None
    )

    risk = get_risk_level(
        weather_code,
        precipitation_probability,
        current.get("precipitation"),
        current.get("wind_gusts_10m"),
    )

    location_name = name or city or "Custom location"

    return {
        "id": "custom",
        "location": {
            "name": location_name,
            "city": city,
            "state": state,
            "latitude": data.get("latitude", latitude),
            "longitude": data.get("longitude", longitude),
            "elevation_m": data.get("elevation"),
        },
        "source": {
            "provider": "Open-Meteo",
            "type": "Forecast API",
            "synthetic_data": False,
        },
        "observation": {
            "time": current.get("time"),
            "timezone": data.get("timezone"),
        },
        "current": {
            "temperature_c": current.get("temperature_2m"),
            "temperature_unit": current_units.get("temperature_2m", "°C"),
            "feels_like_c": current.get("apparent_temperature"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "wind_direction_deg": current.get("wind_direction_10m"),
            "wind_gust_kmh": current.get("wind_gusts_10m"),
            "precipitation_mm": current.get("precipitation"),
            "rain_mm": current.get("rain"),
            "cloud_cover_percent": current.get("cloud_cover"),
            "pressure_hpa": current.get("pressure_msl"),
            "weather_code": weather_code,
            "condition": weather_description(weather_code),
            "risk_level": risk,
        },
        "hourly": extract_hourly_forecast(hourly, 24),
        "daily": {
            "time": daily.get("time", []),
            "temperature_max_c": daily.get("temperature_2m_max", []),
            "temperature_min_c": daily.get("temperature_2m_min", []),
            "precipitation_sum_mm": daily.get("precipitation_sum", []),
            "precipitation_probability_max": daily.get("precipitation_probability_max", []),
            "wind_gust_max_kmh": daily.get("wind_gusts_10m_max", []),
            "weather_code": daily.get("weather_code", []),
        },
    }


async def fetch_location(location_id: str, location: dict[str, Any]) -> dict[str, Any]:
    result = await fetch_weather_coordinates(
        location["latitude"], location["longitude"],
        name=location["name"], city=location["city"], state=location["state"],
    )
    result["id"] = location_id
    return result

# ============================================================
# GEOCODING
# ============================================================

async def search_locations(query: str) -> list[dict[str, Any]]:
    if http_client is None:
        raise RuntimeError("HTTP client is not initialized.")

    query = query.strip()
    if len(query) < 2:
        return []

    response = await http_client.get(
        GEOCODING_URL,
        params={
            "name": query,
            "count": 8,
            "language": "en",
            "format": "json",
        },
    )
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("results", []):
        results.append({
            "name": item.get("name"),
            "city": item.get("name"),
            "state": item.get("admin1"),
            "country": item.get("country"),
            "country_code": item.get("country_code"),
            "latitude": item.get("latitude"),
            "longitude": item.get("longitude"),
            "timezone": item.get("timezone"),
            "elevation_m": item.get("elevation"),
        })

    return results

# ============================================================
# ECMWF + ENSEMBLE INTELLIGENCE
# ============================================================

async def fetch_ecmwf(latitude: float, longitude: float) -> dict[str, Any]:
    if http_client is None:
        raise RuntimeError("HTTP client is not initialized.")

    response = await http_client.get(
        ECMWF_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join([
                "temperature_2m", "precipitation", "rain", "weather_code",
                "wind_speed_10m", "wind_gusts_10m", "cloud_cover", "cape",
            ]),
            "forecast_hours": 48,
            "timezone": "auto",
        },
    )
    response.raise_for_status()
    return response.json()


async def fetch_ensemble(latitude: float, longitude: float) -> dict[str, Any]:
    if http_client is None:
        raise RuntimeError("HTTP client is not initialized.")

    response = await http_client.get(
        ENSEMBLE_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "models": "ecmwf_ifs025_ensemble",
            "hourly": ",".join([
                "temperature_2m", "precipitation", "wind_gusts_10m",
            ]),
            "forecast_hours": 24,
            "timezone": "auto",
        },
    )
    response.raise_for_status()
    return response.json()


def ensemble_member_values(hourly: dict[str, Any], variable: str, index: int) -> list[float]:
    values = []
    prefix = f"{variable}_member"
    for key, series in hourly.items():
        if key.startswith(prefix) and isinstance(series, list) and index < len(series):
            value = safe_float(series[index])
            if value is not None:
                values.append(value)
    return values


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def standard_deviation(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    avg = sum(values) / len(values)
    return math.sqrt(sum((x - avg) ** 2 for x in values) / len(values))


def build_ensemble_summary(data: dict[str, Any]) -> dict[str, Any]:
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    result = []

    for i, timestamp in enumerate(times):
        temps = ensemble_member_values(hourly, "temperature_2m", i)
        rain = ensemble_member_values(hourly, "precipitation", i)
        gusts = ensemble_member_values(hourly, "wind_gusts_10m", i)

        rain_wet_members = sum(1 for value in rain if value >= 0.1)
        rain_probability = (rain_wet_members / len(rain) * 100) if rain else None

        result.append({
            "time": timestamp,
            "members": len(temps),
            "temperature_mean_c": mean(temps),
            "temperature_spread_c": standard_deviation(temps),
            "temperature_p10_c": percentile(temps, 0.10),
            "temperature_p90_c": percentile(temps, 0.90),
            "rain_probability_percent": rain_probability,
            "precipitation_mean_mm": mean(rain),
            "precipitation_p90_mm": percentile(rain, 0.90),
            "wind_gust_mean_kmh": mean(gusts),
            "wind_gust_p90_kmh": percentile(gusts, 0.90),
        })

    return result


def build_model_comparison(base, ecmwf):

    def normalize_hourly(source, temperature_keys):
        """
        Convert different hourly response formats into:

        {
            "timestamp": temperature
        }
        """

        if not isinstance(source, dict):
            return {}

        hourly = source.get("hourly", {})

        result = {}

        # -----------------------------
        # CASE 1: hourly is a list
        # -----------------------------
        if isinstance(hourly, list):

            for item in hourly:

                if not isinstance(item, dict):
                    continue

                timestamp = item.get("time")

                if timestamp is None:
                    continue

                temperature = None

                for key in temperature_keys:

                    if key in item:
                        temperature = safe_float(
                            item.get(key)
                        )

                        if temperature is not None:
                            break

                if temperature is not None:
                    result[str(timestamp)] = temperature

            return result

        # -----------------------------
        # CASE 2: hourly is a dictionary
        # -----------------------------
        if isinstance(hourly, dict):

            times = hourly.get("time", [])

            temperatures = None

            for key in temperature_keys:

                if key in hourly:
                    temperatures = hourly.get(key)
                    break

            if not isinstance(times, list):
                return {}

            if not isinstance(temperatures, list):
                return {}

            for i, timestamp in enumerate(times):

                if i >= len(temperatures):
                    continue

                temperature = safe_float(
                    temperatures[i]
                )

                if temperature is not None:
                    result[str(timestamp)] = temperature

            return result

        return {}

    # ----------------------------------------
    # Normalize Best Match
    # ----------------------------------------

    base_data = normalize_hourly(
        base,
        [
            "temperature_c",
            "temperature_2m"
        ]
    )

    # ----------------------------------------
    # Normalize ECMWF
    # ----------------------------------------

    ecmwf_data = normalize_hourly(
        ecmwf,
        [
            "temperature_2m",
            "temperature_c"
        ]
    )

    # ----------------------------------------
    # Compare matching timestamps
    # ----------------------------------------

    comparisons = []

    for timestamp, base_temperature in base_data.items():

        if timestamp not in ecmwf_data:
            continue

        ecmwf_temperature = ecmwf_data[timestamp]

        difference = (
            base_temperature
            - ecmwf_temperature
        )

        comparisons.append({
            "time": timestamp,
            "base_temperature_c": round(
                base_temperature,
                2
            ),
            "ecmwf_temperature_c": round(
                ecmwf_temperature,
                2
            ),
            "temperature_difference_c": round(
                difference,
                2
            )
        })

    return comparisons[:24]

    base_temp = base_hourly.get("temperature_2m", [])
    e_temp = e_hourly.get("temperature_2m", [])

    comparisons = []
    e_lookup = {t: i for i, t in enumerate(e_times)}

    for i, timestamp in enumerate(base_times[:24]):
        j = e_lookup.get(timestamp)
        if j is None:
            continue
        a = safe_float(base_temp[i]) if i < len(base_temp) else None
        b = safe_float(e_temp[j]) if j < len(e_temp) else None
        if a is None or b is None:
            continue
        comparisons.append({
            "time": timestamp,
            "generic_temperature_c": a,
            "ecmwf_temperature_c": b,
            "temperature_difference_c": round(b - a, 2),
        })

    return comparisons


def calculate_confidence(
    model_comparison: list[dict[str, Any]],
    ensemble: list[dict[str, Any]],
) -> dict[str, Any]:
    differences = [abs(x["temperature_difference_c"]) for x in model_comparison]
    spreads = [x["temperature_spread_c"] for x in ensemble if x.get("temperature_spread_c") is not None]

    avg_model_difference = mean(differences)
    avg_spread = mean(spreads)

    score = 90.0
    reasons = []

    if avg_model_difference is not None:
        if avg_model_difference > 3:
            score -= 30
            reasons.append("Forecast models differ noticeably")
        elif avg_model_difference > 1.5:
            score -= 15
            reasons.append("Some model disagreement")
        else:
            reasons.append("Models are broadly aligned")

    if avg_spread is not None:
        if avg_spread > 3:
            score -= 25
            reasons.append("Ensemble spread is high")
        elif avg_spread > 1.5:
            score -= 10
            reasons.append("Moderate ensemble uncertainty")
        else:
            reasons.append("Ensemble spread is relatively low")

    score = max(0, min(100, round(score)))

    if score >= 75:
        label = "HIGH"
    elif score >= 50:
        label = "MEDIUM"
    else:
        label = "LOW"

    return {
        "score": score,
        "level": label,
        "average_model_temperature_difference_c": round(avg_model_difference, 2) if avg_model_difference is not None else None,
        "average_ensemble_temperature_spread_c": round(avg_spread, 2) if avg_spread is not None else None,
        "reasons": reasons,
    }


async def build_weather_intelligence(
    latitude: float,
    longitude: float,
    name: str | None = None,
) -> dict[str, Any]:
    validate_coordinates(latitude, longitude)

    cache_key = f"{round(latitude, 4)}:{round(longitude, 4)}"
    cached = intelligence_cache.get(cache_key)
    if cached and time.time() - cached["timestamp"] < INTELLIGENCE_CACHE_SECONDS:
        return cached["data"]

    base_task = fetch_weather_coordinates(latitude, longitude, name=name)
    ecmwf_task = fetch_ecmwf(latitude, longitude)
    ensemble_task = fetch_ensemble(latitude, longitude)

    base, ecmwf_result, ensemble_raw = await asyncio.gather(
        base_task,
        ecmwf_task,
        ensemble_task,
    )

    ensemble_summary = build_ensemble_summary(ensemble_raw)
    model_comparison = build_model_comparison(base, ecmwf_result)
    confidence = calculate_confidence(model_comparison, ensemble_summary)

    next_hours = ensemble_summary[:24]
    rain_peak = max(
        [x["rain_probability_percent"] for x in next_hours if x.get("rain_probability_percent") is not None],
        default=None,
    )
    gust_peak = max(
        [x["wind_gust_p90_kmh"] for x in next_hours if x.get("wind_gust_p90_kmh") is not None],
        default=None,
    )

    intelligence = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "location": base["location"],
        "summary": {
            "confidence": confidence,
            "next_24h_peak_ensemble_rain_probability_percent": round(rain_peak, 1) if rain_peak is not None else None,
            "next_24h_peak_p90_wind_gust_kmh": round(gust_peak, 1) if gust_peak is not None else None,
        },
        "models": {
            "forecast": "Open-Meteo Best Match",
            "deterministic": "ECMWF IFS HRES",
            "ensemble": "ECMWF IFS 0.25° Ensemble",
        },
        "model_comparison": model_comparison,
        "ensemble": ensemble_summary,
        "method": {
            "type": "model-agreement-and-ensemble-uncertainty",
            "synthetic_data": False,
            "note": "Confidence is an analytical indicator based on model disagreement and ensemble spread; it is not a guarantee of forecast accuracy.",
        },
    }

    intelligence_cache[cache_key] = {"timestamp": time.time(), "data": intelligence}
    return intelligence

# ============================================================
# DEFAULT LOCATIONS SNAPSHOT
# ============================================================

async def build_weather_snapshot() -> dict[str, Any]:
    tasks = [fetch_location(location_id, location) for location_id, location in LOCATIONS.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    locations = []
    for (location_id, location), result in zip(LOCATIONS.items(), results):
        if isinstance(result, Exception):
            locations.append({
                "id": location_id,
                "location": location,
                "error": True,
                "error_message": str(result),
                "source": {"provider": "Open-Meteo", "synthetic_data": False},
            })
        else:
            locations.append(result)

    successful = sum(1 for item in locations if not item.get("error"))
    return {
        "success": True,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {"provider": "Open-Meteo", "synthetic_data": False},
        "locations": locations,
        "summary": {
            "total_locations": len(locations),
            "successful_locations": successful,
            "failed_locations": len(locations) - successful,
        },
    }


async def get_cached_weather(force_refresh: bool = False) -> dict[str, Any]:
    current_time = time.time()
    if not force_refresh and cache["data"] is not None and current_time - cache["timestamp"] < CACHE_SECONDS:
        return cache["data"]

    async with cache_lock:
        current_time = time.time()
        if not force_refresh and cache["data"] is not None and current_time - cache["timestamp"] < CACHE_SECONDS:
            return cache["data"]

        data = await build_weather_snapshot()
        cache["data"] = data
        cache["timestamp"] = time.time()
        return data

# ============================================================
# API ROUTES
# ============================================================

@app.get("/")
async def root():
    return {
        "name": "Zephyr AI Weather Intelligence API",
        "status": "online",
        "version": "2.0.0",
        "source": "Open-Meteo",
        "synthetic_data": False,
        "intelligence": ["ECMWF IFS HRES", "ECMWF IFS Ensemble"],
        "endpoints": {
            "health": "/health",
            "all_weather": "/api/weather",
            "single_location": "/api/weather/{location_id}",
            "coordinates": "/api/weather/coordinates?lat=28.6139&lon=77.2090",
            "search": "/api/search?q=Delhi",
            "intelligence": "/api/intelligence?lat=28.6139&lon=77.2090",
            "locations": "/api/locations",
        },
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "backend": "online",
        "weather_api": "Open-Meteo",
        "ecmwf": "enabled",
        "ensemble": "enabled",
        "synthetic_data": False,
    }


@app.get("/api/locations")
async def get_locations():
    return {"success": True, "locations": LOCATIONS}


@app.get("/api/search")
async def search(q: str = Query(..., min_length=2, max_length=80)):
    try:
        return {"success": True, "query": q, "results": await search_locations(q)}
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail=f"Geocoding API request failed: {error}")


@app.get("/api/weather")
async def get_weather(refresh: bool = False):
    try:
        return await get_cached_weather(force_refresh=refresh)
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail=f"Weather API request failed: {error}")
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/api/weather/coordinates")
async def get_weather_by_coordinates(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    name: str | None = Query(default=None, max_length=120),
    city: str | None = Query(default=None, max_length=120),
    state: str | None = Query(default=None, max_length=120),
):
    try:
        return await fetch_weather_coordinates(lat, lon, name=name, city=city, state=state)
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail=f"Weather API request failed: {error}")
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/api/intelligence")
async def get_intelligence(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    name: str | None = Query(default=None, max_length=120),
):
    try:
        return await build_weather_intelligence(lat, lon, name=name)
    except httpx.HTTPStatusError as error:
        detail = error.response.text[:500] if error.response is not None else str(error)
        raise HTTPException(status_code=502, detail=f"Weather intelligence provider failed: {detail}")
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail=f"Weather intelligence request failed: {error}")
    except Exception as error:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/api/weather/{location_id}")
async def get_single_weather(location_id: str):
    location_id = location_id.lower()
    if location_id not in LOCATIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown location '{location_id}'. Available locations: {', '.join(LOCATIONS.keys())}",
        )

    try:
        return await fetch_location(location_id, LOCATIONS[location_id])
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail=f"Weather API request failed: {error}")
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

    # ============================================================
# ZEPHYR AI — PHASE 3: RISK & ALERT ENGINE
# ============================================================

def calculate_risk_score(
    rain_probability=None,
    precipitation_mm=None,
    wind_gust_kmh=None,
    temperature_c=None,
    ensemble_spread_c=None,
):
    score = 0
    reasons = []

    # -------------------------
    # RAIN
    # -------------------------
    rain_probability = safe_float(rain_probability)
    precipitation_mm = safe_float(precipitation_mm)

    if rain_probability is not None:
        if rain_probability >= 80:
            score += 35
            reasons.append("Very high probability of precipitation")
        elif rain_probability >= 60:
            score += 25
            reasons.append("High probability of precipitation")
        elif rain_probability >= 40:
            score += 15
            reasons.append("Moderate probability of precipitation")

    if precipitation_mm is not None:
        if precipitation_mm >= 10:
            score += 25
            reasons.append("Potentially heavy precipitation")
        elif precipitation_mm >= 5:
            score += 15
            reasons.append("Moderate precipitation expected")
        elif precipitation_mm >= 2:
            score += 8
            reasons.append("Measurable precipitation expected")

    # -------------------------
    # WIND
    # -------------------------
    wind_gust_kmh = safe_float(wind_gust_kmh)

    if wind_gust_kmh is not None:
        if wind_gust_kmh >= 70:
            score += 35
            reasons.append("Very strong wind gusts possible")
        elif wind_gust_kmh >= 50:
            score += 25
            reasons.append("Strong wind gusts possible")
        elif wind_gust_kmh >= 35:
            score += 15
            reasons.append("Elevated wind gusts possible")

    # -------------------------
    # HEAT
    # -------------------------
    temperature_c = safe_float(temperature_c)

    if temperature_c is not None:
        if temperature_c >= 45:
            score += 35
            reasons.append("Extreme heat conditions")
        elif temperature_c >= 40:
            score += 25
            reasons.append("Very high temperature")
        elif temperature_c >= 37:
            score += 15
            reasons.append("High temperature")

    # -------------------------
    # FORECAST UNCERTAINTY
    # -------------------------
    ensemble_spread_c = safe_float(ensemble_spread_c)

    if ensemble_spread_c is not None:
        if ensemble_spread_c >= 2.5:
            score += 15
            reasons.append("High ensemble uncertainty")
        elif ensemble_spread_c >= 1.5:
            score += 8
            reasons.append("Moderate ensemble uncertainty")

    score = min(score, 100)

    if score >= 70:
        level = "HIGH"
    elif score >= 40:
        level = "MODERATE"
    else:
        level = "LOW"

    return {
        "score": score,
        "level": level,
        "reasons": reasons,
    }


def build_risk_timeline(intelligence):
    """
    Creates an hourly risk timeline from ensemble forecast data.
    """

    ensemble = intelligence.get("ensemble", [])

    timeline = []

    for item in ensemble:
        risk = calculate_risk_score(
            rain_probability=item.get(
                "rain_probability_percent"
            ),
            precipitation_mm=item.get(
                "precipitation_mean_mm"
            ),
            wind_gust_kmh=item.get(
                "wind_gust_p90_kmh"
            ),
            temperature_c=item.get(
                "temperature_mean_c"
            ),
            ensemble_spread_c=item.get(
                "temperature_spread_c"
            ),
        )

        timeline.append({
            "time": item.get("time"),
            **risk,
        })

    return timeline


def find_critical_window(timeline):
    """
    Finds the highest-risk forecast hour.
    """

    if not timeline:
        return None

    valid = [
        item
        for item in timeline
        if item.get("score") is not None
    ]

    if not valid:
        return None

    highest = max(
        valid,
        key=lambda item: item.get("score", 0)
    )

    return {
        "time": highest.get("time"),
        "score": highest.get("score"),
        "level": highest.get("level"),
        "reasons": highest.get("reasons", []),
    }


async def build_risk_intelligence(
    latitude,
    longitude,
    name=None,
):
    """
    Builds Phase-3 weather risk intelligence.

    Uses the existing Zephyr intelligence endpoint internally.
    """

    intelligence = await build_weather_intelligence(
        latitude,
        longitude,
        name=name,
    )

    timeline = build_risk_timeline(
        intelligence
    )

    critical_window = find_critical_window(
        timeline
    )

    scores = [
        item["score"]
        for item in timeline
        if item.get("score") is not None
    ]

    overall_score = (
        max(scores)
        if scores
        else 0
    )

    if overall_score >= 70:
        overall_level = "HIGH"
    elif overall_score >= 40:
        overall_level = "MODERATE"
    else:
        overall_level = "LOW"

    return {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "location": intelligence.get(
            "location",
            {}
        ),

        "overall": {
            "score": overall_score,
            "level": overall_level,
        },

        "critical_window": critical_window,

        "timeline": timeline,

        "method": {
            "synthetic_data": False,
            "type": "ensemble-based weather risk analysis",
            "note": (
                "Risk levels are analytical indicators "
                "derived from forecast variables and "
                "ensemble uncertainty. They are not official "
                "weather warnings."
            ),
        },
    }


# ============================================================
# PHASE 3 API
# ============================================================

@app.get("/api/risk")
async def api_risk(
    lat: float,
    lon: float,
    name: Optional[str] = None,
):
    try:
        return await build_risk_intelligence(
            lat,
            lon,
            name=name,
        )

    except Exception as exc:
        print("\n===== RISK ENGINE ERROR =====")
        traceback.print_exc()
        print("=============================\n")

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )