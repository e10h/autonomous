import math

from . import config


def rotate_xy(x: float, y: float, degrees: float) -> tuple[float, float]:
    rad = math.radians(degrees)
    return (
        x * math.cos(rad) - y * math.sin(rad),
        x * math.sin(rad) + y * math.cos(rad),
    )


def xy_to_latlon(x: float, y: float) -> tuple[float, float]:
    if config.GPS_ROTATION_OFFSET != 0:
        x, y = rotate_xy(x, y, config.GPS_ROTATION_OFFSET)

    d_lat = y / config.EARTH_RADIUS
    d_lon = x / (config.EARTH_RADIUS * math.cos(math.radians(config.LAT0)))
    return config.LAT0 - math.degrees(d_lat), config.LON0 - math.degrees(d_lon)


def latlon_to_xy(
    lat: float,
    lon: float,
    origin_lat: float = config.LAT0,
    origin_lon: float = config.LON0,
) -> tuple[float, float]:
    d_lat = math.radians(lat - origin_lat)
    d_lon = math.radians(lon - origin_lon)
    x = -d_lon * config.EARTH_RADIUS * math.cos(math.radians(origin_lat))
    y = -d_lat * config.EARTH_RADIUS

    if config.GPS_ROTATION_OFFSET != 0:
        x, y = rotate_xy(x, y, -config.GPS_ROTATION_OFFSET)
    return x, y
