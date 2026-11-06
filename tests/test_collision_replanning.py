from threading import Lock
from types import SimpleNamespace

import numpy as np

from autonomous_state_machine.align import AlignAndMove
from autonomous_state_machine.global_map import GlobalMapManager
from autonomous_state_machine.autonomous_state_machine import AutonomousStateMachine
from autonomous_state_machine.states.new_obstacle_avoidance_controller_state import NewObstacleAvoidanceControllerState


class Logger:
    def info(self, *args, **kwargs): pass
    def warn(self, *args, **kwargs): pass
    def debug(self, *args, **kwargs): pass


def test_new_obstacle_invalidates_existing_route_and_detour_clears_it():
    gm = GlobalMapManager(map_size_m=12, cell_size_m=0.1)
    gm.initialize_origin(0, 0)
    assert gm.path_is_clear([(0, 0), (4, 0)], 0.5)
    gm.grid[80, 60] = 100  # two meters forward, observed after initial planning
    assert not gm.path_is_clear([(0, 0), (4, 0)], 0.5)
    path = gm.get_path(0, 0, 4, 0, robot_radius_m=0.5)
    assert len(path) > 2
    assert gm.path_is_clear(path, 0.5)


def test_planner_does_not_clear_obstacle_at_start_or_goal():
    gm = GlobalMapManager(map_size_m=12, cell_size_m=0.1)
    gm.initialize_origin(0, 0)
    gm.grid[100, 60] = 100
    assert gm.get_path(0, 0, 4, 0, robot_radius_m=0.5) == []
    gm.grid[60, 60] = 100
    assert gm.get_path(0, 0, 3, 0, robot_radius_m=0.5) == []


def test_diagonal_cannot_cut_obstacle_corner():
    gm = GlobalMapManager()
    grid = np.zeros((3, 3), dtype=np.uint8)
    grid[1, 0] = grid[0, 1] = 1
    assert not gm._line_of_sight(grid, (0, 0), (1, 1))
    assert (1, 1) not in gm._get_neighbors(grid, (0, 0))


def test_running_avoidance_stops_when_new_obstacle_has_no_safe_detour():
    gm = SimpleNamespace(initialize_origin=lambda *a: None, update_map=lambda *a: None,
                         path_is_clear=lambda *a: False,
                         nearest_free_point=lambda x, y, r: (x, y),
                         get_path=lambda *a, **kw: [])
    sd = SimpleNamespace(local_odom_position=(0, 0), cur_heading=0, global_map=gm,
                         mutex=Lock(), obstacle_avoidance_running=True,
                         obstacle_avoidance_controller=SimpleNamespace(user_unreached_pts=[(4, 0)]),
                         last_path_check_time=0, odom_ready=True, new_waypoints=[(4, 0)],
                         goal_odom=(4, 0), post_oa_state=None, navigation_blocked=False)
    stops = []
    node = SimpleNamespace(shared_data=sd, cur_state=SimpleNamespace(),
                           state_instance=SimpleNamespace(forward_spiral_state=1, backward_spiral_state=2),
                           _parse_pointcloud2_xyz=lambda m: np.array([[0, 0, 1]]),
                           publish_stop=lambda: stops.append(True), get_logger=lambda: Logger())
    AutonomousStateMachine._on_pointcloud(node, None)
    assert sd.navigation_blocked
    assert stops == [True]


def test_path_refresh_does_not_restart_reverse_maneuver():
    sd = SimpleNamespace(new_obstacle_avoidance_waypoints_flag=True,
                         obstacle_avoidance_new_waypoints=[(0, 0), (2, 1)], logger=Logger())
    state = NewObstacleAvoidanceControllerState(sd, SimpleNamespace())
    state.phase = "follow"
    assert state._load_shared_waypoints()
    assert state.phase == "follow"


def test_segment_drift_stops_for_realign_instead_of_driving_blind():
    ctrl = AlignAndMove([(4, 0)], logger=Logger())
    ctrl.start_next_target()
    ctrl.phase = "move"
    sd = SimpleNamespace(local_odom_position=(0, 0), cur_heading=0.4)
    linear, angular, done = ctrl.calc_velocity(sd)
    assert (linear, angular, done) == (0.0, 0.0, False)
    assert ctrl.phase == "align"


def test_rejected_static_route_stops_and_never_shrinks_rover_clearance():
    from autonomous_state_machine.states.new_controller_state import NewControllerState
    from autonomous_state_machine import config
    requests=[];commands=[]
    def no_route(*args,**kwargs):
        requests.append(kwargs['robot_radius_m'])
        return []
    sd=SimpleNamespace(mutex=Lock(),controller=object(),navigation_blocked=False,
        auton_publisher=SimpleNamespace(publish=commands.append),
        new_waypoints_odom=[(0,0),(4,0)],new_waypoints_flag=True,logger=Logger(),
        global_map=SimpleNamespace(static_slope_data={},get_path=no_route))
    state=NewControllerState(sd,SimpleNamespace(new_controller_state='hold',drive_state='drive'))
    state.on_enter()
    assert requests==[config.OA_ROBOT_RADIUS_M]
    assert sd.controller is None and sd.navigation_blocked
    assert not sd.new_waypoints_flag
    assert commands[-1].linear.x==0 and commands[-1].angular.z==0
    assert state.step()=='hold'
