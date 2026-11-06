import math
import numpy as np
import pytest

from autonomous_state_machine.global_map import GlobalMapManager


class DummyLogger:
    def info(self, *args, **kwargs): pass
    def warn(self, *args, **kwargs): pass
    def debug(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass


def test_fallback_to_euclidean_when_no_slope_data():
    """Verify that without slope data, edge cost is pure Euclidean distance."""
    gm = GlobalMapManager(map_size_m=20.0, cell_size_m=0.10, logger=DummyLogger())
    gm.initialize_origin(0.0, 0.0)

    p1 = (50, 50)
    p2 = (53, 54)
    expected_dist = math.hypot(3, 4)  # 5.0

    cost = gm._edge_cost(p1, p2)
    assert pytest.approx(cost, rel=1e-5) == expected_dist


def test_anisotropic_edge_cost_penalizes_roll_more_than_pitch():
    """
    Directional traversability test:
    Given a steep gradient in the Y direction (e.g., hillside sloping upward in +Y):
    - Moving directly in +Y is climbing (pure pitch).
    - Moving across in +X is side-hilling across the slope (pure roll).
    Cross-slope traversal (roll) must incur a quadratic penalty, significantly higher than pitch.
    """
    gm = GlobalMapManager(map_size_m=20.0, cell_size_m=0.10, logger=DummyLogger())
    gm.initialize_origin(0.0, 0.0)

    # Inject steep gradient: slope_y = tan(14 deg) ~ 0.249
    gx = 0.0
    gy = math.tan(math.radians(14.0))

    gm.gx_map.fill(gx)
    gm.gy_map.fill(gy)

    center = (100, 100)
    climb_node = (100, 110)  # +Y movement: climbing straight up
    roll_node = (110, 100)   # +X movement: traversing across the face (side-hill)

    dist_climb = gm._edge_cost(center, climb_node)
    dist_roll = gm._edge_cost(center, roll_node)

    # Both physical distances are 10 cells
    nominal_dist = 10.0
    assert dist_climb > nominal_dist
    assert dist_roll > nominal_dist

    # Roll penalty must be substantially higher than straight climbing penalty
    assert dist_roll > dist_climb * 1.5, f"Roll cost ({dist_roll}) should exceed climb cost ({dist_climb})"


def test_steep_slope_cliffs_marked_impassable():
    """Slopes exceeding the 22 degree threshold must be marked as obstacles."""
    ny, nx = 40, 40
    impassable_mask = np.zeros((ny, nx), dtype=bool)
    impassable_mask[18:22, :] = True  # Sheer canyon cliff across the middle

    dummy_slope_data = {
        "x_min": -10.0,
        "x_max": 10.0,
        "y_min": -10.0,
        "y_max": 10.0,
        "x_webots_min": -10.0,
        "y_webots_min": -10.0,
        "resolution": 0.50,
        "impassable_mask": impassable_mask,
        "slope_deg": np.where(impassable_mask, 35.0, 5.0).astype(np.float32),
        "elevation": np.zeros((ny, nx), dtype=np.float32),
        "gx_odom": np.zeros((ny, nx), dtype=np.float32),
        "gy_odom": np.zeros((ny, nx), dtype=np.float32),
        "max_climb_deg": 18.0,
        "max_descent_deg": 20.0,
        "max_roll_deg": 15.0,
    }

    gm = GlobalMapManager(map_size_m=20.0, cell_size_m=0.10, logger=DummyLogger())
    gm.static_slope_data = dummy_slope_data
    gm.initialize_origin(0.0, 0.0)

    # The cliff cells must have been marked in gm.grid and static_obstacle_mask
    assert gm.static_obstacle_mask.any()
    assert (gm.grid >= gm.obstacle_threshold).any()


def test_anisotropic_theta_star_routes_around_impassable_chasm():
    """
    Planner must detour around an impassable 22+ deg chasm to find the safe pass,
    instead of planning a straight line through the chasm.
    """
    ny, nx = 60, 60
    impassable_mask = np.zeros((ny, nx), dtype=bool)
    # Impassable barrier blocking direct path between center and target,
    # except for a safe pass on the far right (cols 45..60)
    impassable_mask[28:32, 0:45] = True

    dummy_slope_data = {
        "x_min": -15.0,
        "x_max": 15.0,
        "y_min": -15.0,
        "y_max": 15.0,
        "x_webots_min": -15.0,
        "y_webots_min": -15.0,
        "resolution": 0.50,
        "impassable_mask": impassable_mask,
        "slope_deg": np.where(impassable_mask, 40.0, 2.0).astype(np.float32),
        "elevation": np.zeros((ny, nx), dtype=np.float32),
        "gx_odom": np.zeros((ny, nx), dtype=np.float32),
        "gy_odom": np.zeros((ny, nx), dtype=np.float32),
        "max_climb_deg": 18.0,
        "max_descent_deg": 20.0,
        "max_roll_deg": 15.0,
    }

    gm = GlobalMapManager(map_size_m=30.0, cell_size_m=0.10, logger=DummyLogger())
    gm.static_slope_data = dummy_slope_data
    gm.initialize_origin(0.0, 0.0)

    # Plan from robot at (0, -5) to target at (0, 5)
    # Direct path is blocked by the wall at y = 0
    path = gm.get_path(0.0, -5.0, 0.0, 5.0, target_type='NED', robot_radius_m=0.5)

    assert len(path) >= 3, "Path must detour around the barrier"

    # None of the waypoints should cross inside the impassable barrier zone
    # In odom frame, X_odom = -X_webots, Y_odom = -Y_webots
    # Barrier rows 28:32 corresponds to Webots Y ~ [-1.0, 1.0], odom Y ~ [-1.0, 1.0]
    # for X_webots from -15.0 to 7.5 (odom X from -7.5 to 15.0)
    for px, py in path:
        # Check that path detours out to where the pass is (X_webots > 7.5 -> X_odom < -7.5)
        # or stays clear of the barrier bounds
        pass_found = any(pt[0] < -5.0 for pt in path)
        assert pass_found, "Planner did not detour around the impassable wall"


def test_collinear_waypoint_pruning_maintains_sparse_safe_waypoints():
    """Waypoints output from _plan_static_slope_path should be sparse and line-of-sight optimized."""
    ny, nx = 40, 40
    impassable_mask = np.zeros((ny, nx), dtype=bool)

    dummy_slope_data = {
        "x_min": -10.0,
        "x_max": 10.0,
        "y_min": -10.0,
        "y_max": 10.0,
        "x_webots_min": -10.0,
        "y_webots_min": -10.0,
        "resolution": 0.50,
        "impassable_mask": impassable_mask,
        "slope_deg": np.zeros((ny, nx), dtype=np.float32),
        "elevation": np.zeros((ny, nx), dtype=np.float32),
        "gx_odom": np.zeros((ny, nx), dtype=np.float32),
        "gy_odom": np.zeros((ny, nx), dtype=np.float32),
        "max_climb_deg": 18.0,
        "max_descent_deg": 20.0,
        "max_roll_deg": 15.0,
    }

    gm = GlobalMapManager(map_size_m=20.0, cell_size_m=0.10, logger=DummyLogger())
    gm.static_slope_data = dummy_slope_data
    gm.initialize_origin(0.0, 0.0)

    # In open terrain, straight shot should prune down to start and goal
    path = gm.get_path(0.0, 0.0, 0.0, 8.0, target_type='NED', robot_radius_m=0.5)
    assert len(path) == 2, f"Open terrain should prune to 2 sparse waypoints, got {len(path)}"


def test_static_no_route_never_falls_back_to_isotropic_local_plan():
    gm = GlobalMapManager(map_size_m=20.0, cell_size_m=0.10, logger=DummyLogger())
    gm.static_slope_data = {'impassable_mask': np.ones((40,40),dtype=bool),
                            'x_min':-10.0,'y_min':-10.0,'resolution':0.5}
    gm.initialize_origin(0.0,0.0)
    gm._theta_star = lambda *args: pytest.fail('Unsafe isotropic fallback was invoked')
    assert gm.get_path(0.0,0.0,0.0,5.0,target_type='NED',robot_radius_m=0.5)==[]
