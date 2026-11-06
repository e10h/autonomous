# Rotates world XY before lat/lon conversion; adjust if driving along +X gives a diagonal GPS track.
GPS_ROTATION_OFFSET = 0.0

LAT0 = 32.8822983464544
LON0 = -117.23971214571563
EARTH_RADIUS = 6378137.0

# Must match the drivetrain node: angular.z = (right - left) / TRACK_WIDTH_M.
TRACK_WIDTH_M = 1.53

BASE_FRAME_ID = "base_link"
CAMERA_FRAME_ID = "oakd_depth_camera"

OAKD_DEPTH_IMAGE_TOPIC = "/autonomous/obstacle_avoidance/oakd/depth/image_rect"
OAKD_DEPTH_CAMERA_INFO_TOPIC = "/autonomous/obstacle_avoidance/oakd/depth/camera_info"
OAKD_POINTCLOUD_TOPIC = "/autonomous/obstacle_avoidance/oakd/pointcloud"

ENABLE_DEPTH_IMAGE_PUBLISH = True
ENABLE_POINTCLOUD_PUBLISH = True
POINTCLOUD_STRIDE = 4

ENABLE_OCCUPANCY_GRID_PUBLISH = True
OCCUPANCY_CHUNK_SIZE_M = 0.10
OCCUPANCY_MAP_WIDTH_M = 2.0
OCCUPANCY_MAP_DEPTH_M = 2.0
OCCUPANCY_MIN_POINTS_PER_CELL = 8
# Depth sensor height above ground, used to drop floor returns. Tune from the startup calibration log.
CAMERA_HEIGHT_M = 0.9144
OCCUPANCY_FLOOR_THRESHOLD_M = 0.05
OCCUPANCY_CEILING_THRESHOLD_M = 3.0

MARKER_HEIGHT = 0.05
Z_BASE = 0.05

# Extra rotation for the blue (transformed) waypoints, e.g. +/-90 if the state machine rotates them.
TRANSFORMED_WAYPOINT_ROTATION_DEG = 0.0
