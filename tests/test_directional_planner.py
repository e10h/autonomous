import math
import numpy as np
import pytest
from autonomous_state_machine.terrain import TerrainGrid, TraversabilityLimits
from autonomous_state_machine.directional_planner import (
    DirectionalThetaStar,
    sparse_waypoints,
    PlanningBudgetExceeded,
)


def grid(n=35, gx=0.0, gy=0.0, dy=1.0):
    x = np.full((n, n), gx, dtype="float32")
    y = np.full((n, n), gy, dtype="float32")
    return TerrainGrid(
        np.zeros((n, n), dtype=bool), x, x.copy(), y, y.copy(), 0.0, 0.0, 1.0, dy
    )


def test_same_hillside_climb_safe_cross_slope_forbidden():
    terrain = grid(gy=math.tan(math.radians(18)))
    planner = DirectionalThetaStar(terrain, robot_radius_m=0)
    assert math.isfinite(planner.segment_cost((10, 10), (10, 20)))
    assert math.isinf(planner.segment_cost((10, 10), (20, 10)))


def test_uphill_downhill_limits_and_cost_are_asymmetric():
    terrain = grid(gx=math.tan(math.radians(18)))
    planner = DirectionalThetaStar(
        terrain, TraversabilityLimits(climb_deg=12, descent_deg=20), 0
    )
    assert math.isinf(planner.segment_cost((10, 10), (20, 10)))
    assert math.isfinite(planner.segment_cost((20, 10), (10, 10)))
    unrestricted = DirectionalThetaStar(terrain, robot_radius_m=0)
    assert unrestricted.segment_cost((20, 10), (10, 10)) > unrestricted.segment_cost(
        (10, 10), (20, 10)
    )


def test_segment_checks_thin_hazard_away_from_midpoint():
    terrain = grid()
    terrain.gx_min[:, 6] = terrain.gx_max[:, 6] = math.tan(math.radians(19))
    planner = DirectionalThetaStar(terrain, TraversabilityLimits(climb_deg=15), 0)
    assert math.isinf(planner.segment_cost((2, 15), (30, 15)))


def test_supercover_forbids_corner_cutting():
    terrain = grid()
    terrain.blocked[10, 11] = True
    planner = DirectionalThetaStar(terrain, robot_radius_m=0)
    assert math.isinf(planner.segment_cost((10, 10), (11, 11)))


def test_footprint_checks_terrain_beside_centerline():
    terrain = grid()
    terrain.gy_min[16, :] = terrain.gy_max[16, :] = math.tan(math.radians(20))
    assert math.isfinite(
        DirectionalThetaStar(terrain, robot_radius_m=0).segment_cost((5, 15), (25, 15))
    )
    assert math.isinf(
        DirectionalThetaStar(terrain, robot_radius_m=1).segment_cost((5, 15), (25, 15))
    )


def test_barrier_detour_and_every_returned_segment_is_safe():
    terrain = grid()
    terrain.blocked[16:19, :25] = True
    planner = DirectionalThetaStar(terrain, robot_radius_m=0.6)
    path = planner.plan((10, 8), (10, 27), timeout_s=10)
    assert len(path) >= 3
    assert any(x > 25 for x, y in path)
    assert all(
        math.isfinite(planner.segment_cost(a, b)) for a, b in zip(path, path[1:])
    )
    sparse = sparse_waypoints(path, 5)
    assert all(math.dist(a, b) <= 5 + 1e-9 for a, b in zip(sparse, sparse[1:]))
    assert all(
        math.isfinite(planner.segment_cost(a, b)) for a, b in zip(sparse, sparse[1:])
    )


def test_no_route_does_not_erase_endpoints_or_cliffs():
    terrain = grid()
    terrain.blocked[10, 10] = True
    terrain.blocked[16:19, :] = True
    planner = DirectionalThetaStar(terrain, robot_radius_m=0)
    assert planner.plan((10, 10), (20, 20)) == []
    assert planner.plan((10, 8), (10, 27)) == []
    assert terrain.blocked[10, 10]


def test_unknown_and_out_of_map_fail_closed():
    terrain = grid()
    terrain.gx_min[10, 10] = np.nan
    planner = DirectionalThetaStar(terrain, robot_radius_m=0)
    assert planner.plan((10, 10), (20, 20)) == []
    with pytest.raises(ValueError, match="outside"):
        planner.plan((-10, 0), (20, 20))
    assert math.isinf(planner.segment_cost((-10, 0), (20, 20)))


def test_north_up_raster_uses_signed_y_resolution():
    terrain = grid(gy=math.tan(math.radians(18)), dy=-1)
    planner = DirectionalThetaStar(
        terrain, TraversabilityLimits(climb_deg=12, descent_deg=20), 0
    )
    assert math.isfinite(planner.segment_cost((10, -10), (10, -20)))
    assert math.isinf(planner.segment_cost((10, -20), (10, -10)))


def test_heterogeneous_cost_integrates_entire_leg():
    terrain = grid()
    terrain.gy_min[:, 5:12] = terrain.gy_max[:, 5:12] = math.tan(math.radians(12))
    planner = DirectionalThetaStar(terrain, robot_radius_m=0)
    assert (
        planner.segment_cost((2, 15), (30, 15))
        > planner.segment_cost((2, 25), (4, 25)) * 14
    )


def test_hard_cap_applies_even_with_permissive_directional_limits():
    terrain = grid(gx=math.tan(math.radians(24)))
    planner = DirectionalThetaStar(
        terrain, TraversabilityLimits(climb_deg=30, descent_deg=30), 0
    )
    assert planner.plan((10, 10), (20, 10)) == []


def test_budget_exhaustion_is_not_reported_as_no_route():
    terrain = grid()
    terrain.blocked[16:19, :25] = True
    with pytest.raises(PlanningBudgetExceeded):
        DirectionalThetaStar(terrain, robot_radius_m=0).plan(
            (10, 8), (10, 27), max_expansions=1
        )


def test_nonfinite_time_budget_is_rejected():
    with pytest.raises(ValueError, match='budgets'):
        DirectionalThetaStar(grid(), robot_radius_m=0).plan((10,10),(20,20),timeout_s=float('nan'))
