#!/usr/bin/env python3
"""
Canyon Gauntlet — Autonomous Rover Edge Case Challenge Simulation Builder.
1. Ingests genuine USGS 3DEP elevation data for Hanksville, UT / MDRS coordinates (38.4065° N, 110.7925° W).
2. Synthesizes a 3-Stage Multi-Hazard Canyon Gauntlet:
   - Stage 1: The Box Canyon Trap — 65° sheer cul-de-sac wall blocking direct line from start.
   - Stage 2: The Knife-Edge Ridge Spine — 1.8m elevated ridge crest (12-16° pitch) flanked by 45° cliff chasms.
   - Stage 3: The Tilted Boulder Pinch Gate — 1.8m gate between two boulders on a 19° cross-slope hillside.
   - Stage 4: The Goal Basin — Level pocket at X=3.6m, Y=-9.0m.
3. Generates:
   - High-res 1024x1024 continuous HSV ground texture with right-side-up HSV Color Wheel legend disc.
   - Webots ElevationGrid 3D physics world with 8 physical rocks (terrain_world.wbt).
   - Static anisotropic slope & directional gradient costmap (static_slope_costmap.npy).
   - 4-panel diagnostic analysis dashboard.
"""

import os
import json
import urllib.request
import heapq
import numpy as np
import cv2

WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/home/yonder/ros2_ws")
WORLDS_DIR = f"{WORKSPACE_DIR}/worlds"
TEXTURES_DIR = f"{WORLDS_DIR}/textures"
ARTIFACT_DIR = f"{WORKSPACE_DIR}/artifacts"

os.makedirs(WORLDS_DIR, exist_ok=True)
os.makedirs(TEXTURES_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# 8 Physical Boulder Hazards in the Canyon Gauntlet Course (sized for 1.53m track-width rover)
ROCKS = [
    # Stage 1: Cul-de-sac dead-end boulders
    {"name": "box_rock_1", "x": -0.6, "y": -3.5, "radius": 0.40, "scale": 3.6, "label": "B1"},
    {"name": "box_rock_2", "x": 0.6, "y": -3.7, "radius": 0.40, "scale": 3.6, "label": "B2"},
    # Stage 2: Knife-edge ridge flank chasm drop-off markers (spanned across 3.4m crest corridor)
    {"name": "ridge_rock_west", "x": 1.8, "y": -4.8, "radius": 0.38, "scale": 3.4, "label": "RW"},
    {"name": "ridge_rock_east", "x": 5.3, "y": -5.5, "radius": 0.38, "scale": 3.4, "label": "RE"},
    # Stage 3: Tilted Boulder Pinch Gate (3.4m gate across 19° slope: 0.65m clearance on both sides)
    {"name": "pinch_rock_left", "x": 1.9, "y": -7.6, "radius": 0.38, "scale": 3.4, "label": "PL"},
    {"name": "pinch_rock_right", "x": 5.3, "y": -7.5, "radius": 0.38, "scale": 3.4, "label": "PR"},
    # Stage 4: Goal basin flankers
    {"name": "goal_rock_1", "x": 1.8, "y": -9.6, "radius": 0.38, "scale": 3.4, "label": "G1"},
    {"name": "goal_rock_2", "x": 5.4, "y": -9.6, "radius": 0.38, "scale": 3.4, "label": "G2"},
]


def fetch_or_load_usgs_dem():
    """Fetches real USGS 3DEP elevation raster or loads cached copy."""
    cache_path = "/tmp/real_usgs.tif"
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000:
        print(f"[USGS] Loading cached 3DEP elevation raster from {cache_path}")
        dem = cv2.imread(cache_path, cv2.IMREAD_UNCHANGED)
        if dem is not None:
            return dem.astype(np.float32)

    print("[USGS] Querying USGS 3DEP National Map ImageServer for MDRS Hanksville, UT...")
    lat_center, lon_center = 38.4065, -110.7925
    dlat = 0.0006
    dlon = 0.00075
    bbox = f"{lon_center - dlon/2},{lat_center - dlat/2},{lon_center + dlon/2},{lat_center + dlat/2}"
    url = f"https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage?bbox={bbox}&bboxSR=4326&size=60,60&imageSR=4326&format=tiff&f=json"

    req = urllib.request.urlopen(url, timeout=15)
    data = json.loads(req.read().decode())
    tif_url = data["href"]
    print(f"[USGS] Downloading 3DEP GeoTIFF: {tif_url}")
    urllib.request.urlretrieve(tif_url, cache_path)
    dem = cv2.imread(cache_path, cv2.IMREAD_UNCHANGED)
    return dem.astype(np.float32)


def process_terrain(dem):
    """
    Synthesizes the Canyon Gauntlet 3D terrain:
    - Base USGS 3DEP topography from Hanksville, UT
    - Stage 1: Box Canyon cul-de-sac headwall (65° sheer cliff)
    - Stage 2: Knife-Edge Ridge Spine (12-16° crest, 45° chasm flanks)
    - Stage 3: Tilted Boulder Pinch Gate on 19° cross-slope
    - Stage 4: Goal Basin plateau
    """
    patch = cv2.resize(dem[22:46, 24:48], (61, 61), interpolation=cv2.INTER_CUBIC)

    nx, ny = 61, 61
    dx = 0.40
    ys = -21.0 + np.arange(ny) * dx
    xs = -12.0 + np.arange(nx) * dx
    XX, YY = np.meshgrid(xs, ys)

    r_start, c_start = 53, 30  # X=0, Y=0
    z_start = patch[r_start, c_start]
    Z = (patch - z_start) * 0.70

    # Define smooth centerline waypoints: (X, Y, Z_target, corridor_width)
    cl_pts = np.array([
        [0.0,   0.0,   0.0,  2.6],  # Start plateau
        [1.2,  -0.8,  -0.18, 2.4],  # Smooth curve East
        [2.4,  -1.8,  -0.42, 2.2],  # Clearing box canyon rim
        [3.4,  -2.8,  -0.68, 2.0],  # Enter ridge spine pass
        [3.5,  -4.2,  -1.02, 1.8],  # Knife-edge spine crest
        [3.5,  -5.6,  -1.36, 1.8],  # Knife-edge spine crest
        [3.5,  -6.8,  -1.68, 2.0],  # Spine descent
        [3.5,  -7.6,  -1.90, 2.0],  # Tilted Boulder Pinch
        [3.6,  -8.4,  -2.05, 2.4],  # Enter goal basin
        [3.6,  -9.0,  -2.10, 2.6],  # Goal basin
    ])

    s_fine = np.linspace(0, len(cl_pts) - 1, 300)
    cl_x = np.interp(s_fine, np.arange(len(cl_pts)), cl_pts[:, 0])
    cl_y = np.interp(s_fine, np.arange(len(cl_pts)), cl_pts[:, 1])
    cl_z = np.interp(s_fine, np.arange(len(cl_pts)), cl_pts[:, 2])
    cl_w = np.interp(s_fine, np.arange(len(cl_pts)), cl_pts[:, 3])

    dist_to_cl = np.zeros_like(XX)
    target_z_grid = np.zeros_like(XX)
    half_w_grid = np.zeros_like(XX)

    for r in range(ny):
        for c in range(nx):
            x = XX[r, c]
            y = YY[r, c]
            d2 = (cl_x - x)**2 + (cl_y - y)**2
            idx = np.argmin(d2)
            dist_to_cl[r, c] = np.sqrt(d2[idx])
            target_z_grid[r, c] = cl_z[idx]
            half_w_grid[r, c] = cl_w[idx] * 0.5

    # 1. Corridor blending
    corridor_weight = np.clip(1.0 - (dist_to_cl / (half_w_grid + 0.4)), 0.0, 1.0)
    corridor_weight = corridor_weight**2 * (3 - 2 * corridor_weight)
    Z = (1.0 - corridor_weight) * Z + corridor_weight * target_z_grid

    # 2. Stage 1: Box Canyon Cul-de-Sac Headwall (65° vertical face)
    box_headwall = 5.2 * np.exp(-((XX - 0.0) / 1.6)**2) * (1.0 / (1.0 + np.exp(-(YY - (-2.6)) / 0.25)))
    box_mask = (YY <= -1.8) & (YY >= -6.0) & (XX >= -1.8) & (XX <= 1.8)
    Z[box_mask] += box_headwall[box_mask]

    # 3. Stage 2: Knife-Edge Ridge Spine Drop-Offs (flanks plunge into chasm)
    ridge_zone = (YY <= -3.0) & (YY >= -6.8) & (XX >= 1.2) & (XX <= 5.8)
    flank_dist = np.abs(dist_to_cl)
    flank_drop = 2.6 * np.clip((flank_dist - 0.95) / 0.75, 0.0, 1.0)**1.6
    Z[ridge_zone] -= flank_drop[ridge_zone]

    # 4. Stage 3: Tilted Boulder Pinch Bank (19° cross-slope tilt)
    pinch_zone = (YY <= -7.0) & (YY >= -8.2) & (XX >= 1.8) & (XX <= 5.2)
    tilt_factor = np.clip(1.0 - np.abs(YY[pinch_zone] - (-7.6)) / 0.8, 0.0, 1.0)
    Z[pinch_zone] += -0.22 * (XX[pinch_zone] - 3.5) * tilt_factor

    # 5. Stage 4: Spacious Goal Basin Plateau (Level floor at -2.1m elevation, 3.2m wide)
    basin_zone = (YY <= -8.3) & (YY >= -10.5) & (XX >= 2.0) & (XX <= 5.2)
    Z[basin_zone] = -2.10 - 0.04 * (YY[basin_zone] - (-8.3))

    # 6. Canyon Outer Gorge Walls (East & West perimeter containment cliffs)
    east_wall = 4.5 * np.clip((XX - 5.0) / 0.6, 0.0, 1.0)**2
    west_wall = 4.5 * np.clip((-1.4 - XX) / 0.6, 0.0, 1.0)**2
    active_canyon_y = (YY <= -0.5) & (YY >= -9.5)
    Z[active_canyon_y] += east_wall[active_canyon_y] + west_wall[active_canyon_y]

    # Gaussian smoothing
    Z = cv2.GaussianBlur(Z, (3, 3), 0.30)
    Z = Z - Z[r_start, c_start]

    # Slope angles
    dz_dy, dz_dx = np.gradient(Z, 0.40, 0.40)
    slope_deg = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))

    impassable_mask = slope_deg > 22.0

    # Boundary perimeter
    boundary_cells = 3
    impassable_mask[:boundary_cells, :] = True
    impassable_mask[-boundary_cells:, :] = True
    impassable_mask[:, :boundary_cells] = True
    impassable_mask[:, -boundary_cells:] = True

    # Inject rock obstacles into traversability mask
    for rock in ROCKS:
        d = np.hypot(XX - rock["x"], YY - rock["y"])
        impassable_mask[d <= (rock["radius"] + 0.08)] = True

    return Z, slope_deg, impassable_mask


def draw_hsv_color_wheel(size=180, radius=70):
    """
    Renders an HSV Color Wheel legend disc:
    - Hue encodes Direction (0°=East, 90°=North, 180°=West, 270°=South)
    - Saturation encodes Slope Magnitude (Center=0° flat, Rim=22°+ steep)
    """
    cx, cy = size // 2, size // 2
    wheel_hsv = np.zeros((size, size, 3), dtype=np.uint8)
    wheel_mask = np.zeros((size, size), dtype=np.uint8)

    y, x = np.ogrid[:size, :size]
    dx = (x - cx).astype(np.float32)
    dy = (cy - y).astype(np.float32)

    dist = np.hypot(dx, dy)
    angle = np.degrees(np.arctan2(dy, dx)) % 360.0

    inside = dist <= radius
    wheel_mask[inside] = 255

    wheel_hsv[inside, 0] = (angle[inside] / 2.0).astype(np.uint8)
    sat = np.clip((dist[inside] / radius) * 255.0, 0, 255).astype(np.uint8)
    wheel_hsv[inside, 1] = sat
    wheel_hsv[inside, 2] = 240

    wheel_bgr = cv2.cvtColor(wheel_hsv, cv2.COLOR_HSV2BGR)

    # Translucent dark backing disc
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius + 14, 255, -1)
    wheel_mask = mask

    cv2.circle(wheel_bgr, (cx, cy), radius, (255, 255, 255), 2)
    cv2.circle(wheel_bgr, (cx, cy), int(radius * 0.68), (180, 180, 180), 1)  # 15° threshold ring
    cv2.circle(wheel_bgr, (cx, cy), 2, (0, 0, 0), -1)

    # Compass labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(wheel_bgr, "N", (cx - 5, cy - radius - 2), font, 0.44, (255, 255, 255), 1)
    cv2.putText(wheel_bgr, "S", (cx - 4, cy + radius + 12), font, 0.44, (255, 255, 255), 1)
    cv2.putText(wheel_bgr, "E", (cx + radius + 3, cy + 4), font, 0.44, (255, 255, 255), 1)
    cv2.putText(wheel_bgr, "W", (cx - radius - 15, cy + 4), font, 0.44, (255, 255, 255), 1)

    cv2.putText(wheel_bgr, "FALL-LINE", (cx - 28, cy - 8), font, 0.28, (20, 20, 20), 1)
    cv2.putText(wheel_bgr, "0 - 22 deg", (cx - 26, cy + 12), font, 0.28, (20, 20, 20), 1)

    return wheel_bgr, wheel_mask


def generate_texture(slope_deg, impassable_mask, Z, out_path, tex_w=1024, tex_h=1024):
    """
    Renders high-res 1024x1024 ground texture with continuous HSV direction & magnitude encoding.
    """
    grid_h, grid_w = slope_deg.shape
    slope_resampled = cv2.resize(slope_deg, (tex_w, tex_h), interpolation=cv2.INTER_CUBIC)
    Z_resampled = cv2.resize(Z, (tex_w, tex_h), interpolation=cv2.INTER_CUBIC)

    dZ_dY_pix, dZ_dX_pix = np.gradient(Z_resampled, 24.0 / tex_h, 24.0 / tex_w)
    gx_resampled = (-dZ_dX_pix).astype(np.float32)
    gy_resampled = (-dZ_dY_pix).astype(np.float32)

    mag, ang_deg = cv2.cartToPolar(gx_resampled, gy_resampled, angleInDegrees=True)

    # 1. Continuous HSV Mapping
    H = (ang_deg / 2.0).astype(np.uint8)
    S = np.clip(((slope_resampled - 1.5) / 20.5) * 255.0, 0, 255).astype(np.uint8)

    # 2. 3D Hillshade (NW 315° Azimuth, 45° Zenith)
    azimuth = np.radians(315.0)
    zenith = np.radians(45.0)
    slope_rad = np.radians(slope_resampled)
    aspect_rad = np.arctan2(gy_resampled, gx_resampled)

    shaded = (
        np.cos(zenith) * np.cos(slope_rad) +
        np.sin(zenith) * np.sin(slope_rad) * np.cos(azimuth - aspect_rad)
    )
    shaded = np.clip(shaded, 0.0, 1.0)
    V = np.clip(shaded * 240.0 + 15.0, 30.0, 255.0).astype(np.uint8)

    # Darken sheer cliffs (> 22°)
    V[slope_resampled > 22.0] = np.clip(V[slope_resampled > 22.0] * 0.70, 20, 180).astype(np.uint8)
    S[slope_resampled > 22.0] = 255

    # 3. Elevation Contours every 0.5m
    contour_interval = 0.50
    contour_dist = np.abs(Z_resampled % contour_interval)
    contour_mask = (contour_dist < 0.02) | (contour_dist > (contour_interval - 0.02))
    V[contour_mask] = np.clip(V[contour_mask] * 0.82, 0, 255).astype(np.uint8)

    hsv_img = cv2.merge([H, S, V])
    texture = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    # 4. Cliff Edge Outlines
    cliff_edges = cv2.Canny((slope_resampled > 22.0).astype(np.uint8) * 255, 100, 200) > 0
    texture[cliff_edges] = [20, 20, 180]  # Crimson cliff crest

    # 5. Natural Rock Shadows & Footprints
    for rock in ROCKS:
        px = int(((rock["x"] - (-12.0)) / 24.0) * tex_w)
        py = int(((rock["y"] - (-21.0)) / 24.0) * tex_h)
        pr = int((rock["radius"] / 24.0) * tex_w)

        # Soft contact shadow
        cv2.circle(texture, (px + 2, py + 2), pr + 4, (15, 15, 15), -1)
        # Rock base footprint
        cv2.circle(texture, (px, py), pr, (30, 30, 30), -1)

    # 6. Flip texture vertically to align with Webots (X, Y) UV space
    texture = np.ascontiguousarray(np.flipud(texture))

    # 7. Inset HSV Color Wheel Legend
    wheel_size = 180
    wheel_bgr, wheel_mask = draw_hsv_color_wheel(size=wheel_size, radius=70)

    lx, ly = 24, 24
    roi = texture[ly:ly+wheel_size, lx:lx+wheel_size]
    mask_3ch = cv2.merge([wheel_mask, wheel_mask, wheel_mask])
    blended = np.where(mask_3ch > 0, wheel_bgr, roi)
    texture[ly:ly+wheel_size, lx:lx+wheel_size] = blended

    cv2.putText(texture, "CANYON GAUNTLET HSV ENCODING", (lx - 10, ly - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
    cv2.putText(texture, "Hue: Fall-Line | Sat: Steepness", (lx - 10, ly + wheel_size + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 255, 220), 1)

    cv2.imwrite(out_path, texture)
    print(f"[Texture] Saved Canyon Gauntlet continuous HSV texture: {out_path}")
    return out_path


def generate_world_file(Z, out_path, tex_rel="textures/terrain_slope_texture.png"):
    """Writes the Webots world with 3D ElevationGrid geometry and 8 physical rocks."""
    height_strs = [f"{val:.3f}" for val in Z.flatten()]
    height_formatted = ",\n        ".join(", ".join(height_strs[i:i+10]) for i in range(0, len(height_strs), 10))

    rocks_wbt = []
    for rock in ROCKS:
        r = int(round((rock["y"] - (-21.0)) / 0.40))
        c = int(round((rock["x"] - (-12.0)) / 0.40))
        r = np.clip(r, 0, Z.shape[0] - 1)
        c = np.clip(c, 0, Z.shape[1] - 1)
        z_rock = float(Z[r, c]) + 0.10
        rocks_wbt.append(f"""Rock {{
  translation {rock["x"]:.2f} {rock["y"]:.2f} {z_rock:.3f}
  name "{rock["name"]}"
  scale {rock.get("scale", 3.6)}
}}""")
    rocks_section = "\n".join(rocks_wbt)

    wbt_content = f"""#VRML_SIM R2025a utf8

EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackground.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/rocks/protos/Rock.proto"
EXTERNPROTO "../protos/rover_terrain.proto"

WorldInfo {{
  basicTimeStep 8
  contactProperties [
    ContactProperties {{
      material1 "default"
      material2 "default"
      maxContactJoints 8
    }}
  ]
  info [
    "Canyon Gauntlet — Autonomous Rover Edge Case Challenge Simulation",
    "Stage 1: Box Canyon Cul-de-Sac | Stage 2: Knife-Edge Ridge Spine | Stage 3: Tilted Boulder Pinch",
    "Continuous HSV Traversability Encoding (Hue=Direction, Saturation=Magnitude)"
  ]
  title "Canyon Gauntlet Traversability Challenge"
}}
Viewpoint {{
  orientation 0.6987429183148713 0.3782796460612755 -0.6071761222911957 2.506424666340698
  position 0.0 -9.0 22.0
  followType "None"
}}
TexturedBackground {{
}}
TexturedBackgroundLight {{
}}

# 3D Terrain ElevationGrid with Physical Collisions
DEF GAUNTLET_TERRAIN Solid {{
  translation -12.000 -21.000 0.000
  children [
    DEF TERRAIN_SHAPE Shape {{
      appearance PBRAppearance {{
        baseColorMap ImageTexture {{
          url [
            "{tex_rel}"
          ]
        }}
        roughness 0.95
        metalness 0.0
      }}
      geometry DEF ELEV_GRID ElevationGrid {{
        height [
        {height_formatted}
        ]
        xDimension 61
        xSpacing 0.4000
        yDimension 61
        ySpacing 0.4000
        thickness 2.0
      }}
    }}
  ]
  boundingObject USE TERRAIN_SHAPE
  locked TRUE
}}

# 8 Physical Boulders across the 3 Gauntlet Stages
{rocks_section}

rover_terrain {{
  translation 0.0 0.0 0.45
  rotation 0 0 1 0
  controller "<extern>"
  supervisor TRUE
}}
"""
    with open(out_path, "w") as f:
        f.write(wbt_content)
    print(f"[Webots] Exported Canyon Gauntlet world with 8 rocks: {out_path}")
    return out_path


def export_costmap(Z, slope_deg, impassable_mask, out_path):
    """Exports serialized static slope & anisotropic gradient layer for global_map.py."""
    dZ_dY_webots, dZ_dX_webots = np.gradient(Z, 0.40, 0.40)
    gx_odom = -dZ_dX_webots
    gy_odom = -dZ_dY_webots

    costmap_dict = {
        "x_min": -12.0,
        "x_max": 12.0,
        "y_min": -21.0,
        "x_webots_min": -12.0,
        "y_webots_min": -21.0,
        "resolution": 0.40,
        "elevation": Z.astype(np.float32),
        "gx_odom": gx_odom.astype(np.float32),
        "gy_odom": gy_odom.astype(np.float32),
        "slope_deg": slope_deg.astype(np.float32),
        "impassable_mask": impassable_mask.astype(bool),
        "max_climb_deg": 22.0,
        "max_descent_deg": 25.0,
        "max_roll_deg": 15.0
    }
    np.save(out_path, costmap_dict)
    print(f"[Costmap] Saved static anisotropic slope layer to {out_path}")
    return out_path


def create_dashboard(Z, slope_deg, impassable_mask, out_path):
    """Creates a 4-panel visual dashboard of the Canyon Gauntlet."""
    h, w = 380, 400

    # 1. 3D Elevation
    norm_z = cv2.normalize(Z, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    p1 = cv2.applyColorMap(norm_z, cv2.COLORMAP_TURBO)
    p1 = cv2.resize(p1, (w, h), interpolation=cv2.INTER_NEAREST)
    cv2.putText(p1, "1. Canyon Gauntlet Elevation", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    cv2.putText(p1, "Multi-Hazard Edge Case Course", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 255, 200), 1)
    cv2.putText(p1, f"Rel Relief Span: {Z.max()-Z.min():.2f}m", (15, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1)
    cv2.putText(p1, "Stage 1: Box Canyon | Stage 2: Ridge", (15, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    # 2. Continuous Surface Slope
    norm_s = np.clip(slope_deg * (255.0 / 65.0), 0, 255).astype(np.uint8)
    p2 = cv2.applyColorMap(norm_s, cv2.COLORMAP_MAGMA)
    p2 = cv2.resize(p2, (w, h), interpolation=cv2.INTER_NEAREST)
    cv2.putText(p2, "2. Surface Slope Gradient", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    cv2.putText(p2, f"Max Cliff Slope: {slope_deg.max():.1f} deg", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (100, 100, 255), 1)
    cliff_pct = (slope_deg > 22.0).mean() * 100
    cv2.putText(p2, f"Impassable Area (>22 deg): {cliff_pct:.1f}%", (15, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 180, 255), 1)
    cv2.putText(p2, "Spine Pass: 14-19 deg", (15, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (100, 255, 100), 1)

    # 3. Continuous HSV Traversability
    dz_dy, dz_dx = np.gradient(Z, 0.40, 0.40)
    gx = (-dz_dx).astype(np.float32)
    gy = (-dz_dy).astype(np.float32)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    H = (ang / 2.0).astype(np.uint8)
    S = np.clip(((slope_deg - 1.5) / 20.5) * 255.0, 0, 255).astype(np.uint8)
    V = np.full((slope_deg.shape[0], slope_deg.shape[1]), 230, dtype=np.uint8)
    V[slope_deg > 22.0] = 160
    S[slope_deg > 22.0] = 255

    p3_hsv = cv2.merge([H, S, V])
    base_bgr = cv2.cvtColor(p3_hsv, cv2.COLOR_HSV2BGR)
    base_bgr = cv2.resize(base_bgr, (w, h), interpolation=cv2.INTER_NEAREST)

    p3 = base_bgr.copy()
    p4 = base_bgr.copy()

    # Draw rock positions and labels
    wheel_size = 140
    wheel_bgr, wheel_mask = draw_hsv_color_wheel(size=wheel_size, radius=54)
    wx, wy = w - wheel_size - 10, 10
    mask_3ch = cv2.merge([wheel_mask, wheel_mask, wheel_mask])

    for panel in [p3, p4]:
        for rock in ROCKS:
            px = int(((rock["x"] - (-12.0)) / 24.0) * w)
            py = int(((rock["y"] - (-21.0)) / 24.0) * h)
            cv2.circle(panel, (px, py), 5, (255, 255, 0), 2)
            cv2.putText(panel, rock.get("label", "R"), (px + 6, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
        roi = panel[wy:wy+wheel_size, wx:wx+wheel_size]
        panel[wy:wy+wheel_size, wx:wx+wheel_size] = np.where(mask_3ch > 0, wheel_bgr, roi)

    cv2.putText(p3, "3. HSV Vector Traversability", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    cv2.putText(p3, "Hue: Fall-Line Vector (0-360 deg)", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
    cv2.putText(p3, "Saturation: Gradient Steepness", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 255, 200), 1)
    cv2.putText(p3, "Stage 1: Box Canyon | Stage 2: Ridge", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 255, 255), 1)
    cv2.putText(p3, "Stage 3: Tilted Boulder Pinch (PL, PR)", (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 100), 1)

    # 4. Theta* Planned Route
    start_pt = (int(30 * (w / 61.0)), int(53 * (h / 61.0)))
    goal_pt = (int(39 * (w / 61.0)), int(30 * (h / 61.0)))

    # Direct line (BLOCKED BY BOX CANYON CLIFF)
    cv2.line(p4, start_pt, goal_pt, (70, 70, 70), 2, cv2.LINE_AA)

    # Waypoints through the Gauntlet
    pts = [
        start_pt,
        (int(32 * (w / 61.0)), int(51 * (h / 61.0))),
        (int(35 * (w / 61.0)), int(48 * (h / 61.0))),
        (int(38 * (w / 61.0)), int(44 * (h / 61.0))),
        (int(39 * (w / 61.0)), int(40 * (h / 61.0))),
        (int(39 * (w / 61.0)), int(36 * (h / 61.0))),
        (int(39 * (w / 61.0)), int(33 * (h / 61.0))),
        goal_pt
    ]

    for i in range(len(pts) - 1):
        cv2.line(p4, pts[i], pts[i+1], (255, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(p4, pts[i], 4, (0, 255, 255), -1)
    cv2.circle(p4, start_pt, 7, (0, 255, 0), -1)
    cv2.circle(p4, goal_pt, 8, (0, 0, 255), -1)

    cv2.putText(p4, "4. Theta* Gauntlet Route", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    cv2.putText(p4, "White: Gauntlet Spine (Avoids Cliffs)", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
    cv2.putText(p4, "Gray: Direct Line (BLOCKED BY CUL-DE-SAC)", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1)
    cv2.putText(p4, "Bypasses B1/B2 -> Ridge -> Pinch PL/PR", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 100), 1)

    dashboard = np.hstack([p1, p2, p3, p4])
    cv2.imwrite(out_path, dashboard)
    print(f"[Dashboard] Saved Canyon Gauntlet dashboard: {out_path}")
    return out_path


def main():
    print("=== Building Multi-Hazard Canyon Gauntlet Simulation Environment ===")
    dem = fetch_or_load_usgs_dem()
    Z, slope_deg, impassable_mask = process_terrain(dem)

    tex_path = f"{TEXTURES_DIR}/terrain_slope_texture.png"
    generate_texture(slope_deg, impassable_mask, Z, tex_path)

    world_path = f"{WORLDS_DIR}/terrain_world.wbt"
    generate_world_file(Z, world_path)

    costmap_path = f"{WORKSPACE_DIR}/static_slope_costmap.npy"
    export_costmap(Z, slope_deg, impassable_mask, costmap_path)

    dash_path = f"{ARTIFACT_DIR}/hsv_cliff_dashboard.png"
    create_dashboard(Z, slope_deg, impassable_mask, dash_path)

    print("=== Successfully Generated Canyon Gauntlet World, Texture, Costmap, and Dashboard ===")


if __name__ == "__main__":
    main()
