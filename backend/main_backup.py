import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# CONFIGURATION
# ============================================================

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

CACHE_SECONDS = 60


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
# WEATHER CODE TRANSLATION
# Open-Meteo WMO weather codes
# ============================================================

WEATHER_CODES = {
    0: "Clear sky",

    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",

    45: "Fog",
    48: "Depositing rime fog",

    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",

    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",

    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",

    66: "Light freezing rain",
    67: "Heavy freezing rain",

    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",

    77: "Snow grains",

    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",

    85: "Slight snow showers",
    86: "Heavy snow showers",

    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


# ============================================================
# GLOBAL STATE
# ============================================================

http_client: httpx.AsyncClient | None = None

cache = {
    "timestamp": 0.0,
    "data": None,
}

cache_lock = asyncio.Lock()


# ============================================================
# FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(15.0)
    )

    print("==========================================")
    print("        ZEPHYR AI BACKEND STARTED")
    print("==========================================")
    print("Weather source : Open-Meteo")
    print("API server     : http://127.0.0.1:8000")
    print("==========================================")

    yield

    await http_client.aclose()
    http_client = None


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Zephyr AI Weather Intelligence API",
    description="Weather intelligence backend powered by Open-Meteo.",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HELPERS
# ============================================================

def weather_description(code: int | None) -> str:
    if code is None:
        return "Unknown"

    return WEATHER_CODES.get(
        code,
        f"Weather condition code {code}"
    )


def get_risk_level(
    weather_code: int | None,
    precipitation_probability: float | None,
    precipitation: float | None,
) -> str:

    # Thunderstorm
    if weather_code in [95, 96, 99]:
        return "HIGH"

    # Heavy rain
    if weather_code in [65, 67, 82]:
        return "HIGH"

    # Moderate rain / significant precipitation probability
    if weather_code in [63, 81]:
        return "MODERATE"

    if precipitation_probability is not None:
        if precipitation_probability >= 70:
            return "MODERATE"

    if precipitation is not None:
        if precipitation >= 10:
            return "MODERATE"

    return "LOW"


def extract_hourly_forecast(hourly: dict[str, Any]) -> list[dict[str, Any]]:
    times = hourly.get("time", [])
    temperatures = hourly.get("temperature_2m", [])
    rain_probability = hourly.get(
        "precipitation_probability",
        []
    )
    precipitation = hourly.get(
        "precipitation",
        []
    )
    weather_codes = hourly.get(
        "weather_code",
        []
    )

    result = []

    # Only expose the next 24 hours
    limit = min(24, len(times))

    for i in range(limit):
        result.append({
            "time": times[i] if i < len(times) else None,

            "temperature_c": (
                temperatures[i]
                if i < len(temperatures)
                else None
            ),

            "precipitation_probability": (
                rain_probability[i]
                if i < len(rain_probability)
                else None
            ),

            "precipitation_mm": (
                precipitation[i]
                if i < len(precipitation)
                else None
            ),

            "weather_code": (
                weather_codes[i]
                if i < len(weather_codes)
                else None
            ),

            "condition": weather_description(
                weather_codes[i]
                if i < len(weather_codes)
                else None
            ),
        })

    return result


async def fetch_location(
    location_id: str,
    location: dict[str, Any],
) -> dict[str, Any]:

    if http_client is None:
        raise RuntimeError("HTTP client is not initialized.")

    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],

        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation",
            "rain",
            "cloud_cover",
            "pressure_msl",
            "wind_speed_10m",
            "wind_direction_10m",
            "weather_code",
        ]),

        "hourly": ",".join([
            "temperature_2m",
            "precipitation_probability",
            "precipitation",
            "rain",
            "weather_code",
        ]),

        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "precipitation_probability_max",
            "weather_code",
        ]),

        "forecast_days": 3,
        "timezone": "auto",
    }

    response = await http_client.get(
        OPEN_METEO_URL,
        params=params,
    )

    response.raise_for_status()

    data = response.json()

    current = data.get("current", {})
    current_units = data.get("current_units", {})

    hourly = data.get("hourly", {})

    daily = data.get("daily", {})

    weather_code = current.get("weather_code")

    precipitation_probability = None

    if hourly.get("precipitation_probability"):
        precipitation_probability = (
            hourly["precipitation_probability"][0]
        )

    risk = get_risk_level(
        weather_code,
        precipitation_probability,
        current.get("precipitation"),
    )

    return {
        "id": location_id,

        "location": {
            "name": location["name"],
            "city": location["city"],
            "state": location["state"],
            "latitude": location["latitude"],
            "longitude": location["longitude"],
        },

        "source": {
            "provider": "Open-Meteo",
            "type": "Weather API",
            "synthetic_data": False,
        },

        "observation": {
            "time": current.get("time"),
            "timezone": data.get("timezone"),
        },

        "current": {
            "temperature_c": current.get("temperature_2m"),
            "temperature_unit": current_units.get(
                "temperature_2m",
                "°C"
            ),

            "feels_like_c": current.get(
                "apparent_temperature"
            ),

            "humidity_percent": current.get(
                "relative_humidity_2m"
            ),

            "wind_speed_kmh": current.get(
                "wind_speed_10m"
            ),

            "wind_direction_deg": current.get(
                "wind_direction_10m"
            ),

            "precipitation_mm": current.get(
                "precipitation"
            ),

            "rain_mm": current.get(
                "rain"
            ),

            "cloud_cover_percent": current.get(
                "cloud_cover"
            ),

            "pressure_hpa": current.get(
                "pressure_msl"
            ),

            "weather_code": weather_code,

            "condition": weather_description(
                weather_code
            ),

            "risk_level": risk,
        },

        "hourly": extract_hourly_forecast(hourly),

        "daily": {
            "time": daily.get("time", []),

            "temperature_max_c": daily.get(
                "temperature_2m_max",
                []
            ),

            "temperature_min_c": daily.get(
                "temperature_2m_min",
                []
            ),

            "precipitation_sum_mm": daily.get(
                "precipitation_sum",
                []
            ),

            "precipitation_probability_max": daily.get(
                "precipitation_probability_max",
                []
            ),

            "weather_code": daily.get(
                "weather_code",
                []
            ),
        },
    }


# ============================================================
# FETCH ALL LOCATIONS
# ============================================================

async def build_weather_snapshot() -> dict[str, Any]:

    tasks = [
        fetch_location(
            location_id,
            location,
        )
        for location_id, location in LOCATIONS.items()
    ]

    results = await asyncio.gather(
        *tasks,
        return_exceptions=True,
    )

    locations = []

    for (location_id, _), result in zip(
        LOCATIONS.items(),
        results,
    ):

        if isinstance(result, Exception):

            locations.append({
                "id": location_id,

                "location": LOCATIONS[location_id],

                "error": True,

                "error_message": str(result),

                "source": {
                    "provider": "Open-Meteo",
                    "synthetic_data": False,
                },
            })

        else:
            locations.append(result)

    successful = sum(
        1
        for item in locations
        if not item.get("error")
    )

    return {
        "success": True,

        "generated_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),

        "source": {
            "provider": "Open-Meteo",
            "synthetic_data": False,
        },

        "locations": locations,

        "summary": {
            "total_locations": len(locations),
            "successful_locations": successful,
            "failed_locations": (
                len(locations) - successful
            ),
        },
    }


# ============================================================
# CACHED SNAPSHOT
# ============================================================

async def get_cached_weather(
    force_refresh: bool = False,
) -> dict[str, Any]:

    current_time = time.time()

    # Return cached data
    if (
        not force_refresh
        and cache["data"] is not None
        and current_time - cache["timestamp"]
        < CACHE_SECONDS
    ):
        return cache["data"]

    async with cache_lock:

        # Check again after acquiring lock
        current_time = time.time()

        if (
            not force_refresh
            and cache["data"] is not None
            and current_time - cache["timestamp"]
            < CACHE_SECONDS
        ):
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
        "version": "1.0.0",
        "source": "Open-Meteo",
        "synthetic_data": False,
        "endpoints": {
            "health": "/health",
            "all_weather": "/api/weather",
            "single_location": "/api/weather/{location_id}",
            "locations": "/api/locations",
        },
    }


@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "backend": "online",
        "weather_api": "Open-Meteo",
        "synthetic_data": False,
    }


@app.get("/api/locations")
async def get_locations():

    return {
        "success": True,
        "locations": LOCATIONS,
    }


@app.get("/api/weather")
async def get_weather(
    refresh: bool = False,
):

    try:

        return await get_cached_weather(
            force_refresh=refresh
        )

    except httpx.HTTPError as error:

        raise HTTPException(
            status_code=502,
            detail=f"Weather API request failed: {error}",
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@app.get("/api/weather/{location_id}")
async def get_single_weather(
    location_id: str,
):

    location_id = location_id.lower()

    if location_id not in LOCATIONS:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown location '{location_id}'. "
                f"Available locations: "
                f"{', '.join(LOCATIONS.keys())}"
            ),
        )

    try:

        return await fetch_location(
            location_id,
            LOCATIONS[location_id],
        )

    except httpx.HTTPError as error:

        raise HTTPException(
            status_code=502,
            detail=f"Weather API request failed: {error}",
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ============================================================
# RUN MESSAGE
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )