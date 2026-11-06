"""Pace Webots fast mode toward a requested simulation/wall-time ratio."""
import time


class SimulationSpeed:
    def __init__(self, robot):
        self.robot = robot
        self.requested = 1.0
        self.actual = None
        self._reset()

    def _reset(self):
        self.wall_anchor = self.sample_wall = time.monotonic()
        self.sim_anchor = self.sample_sim = self.robot.getTime()
        self.actual = None

    def set_speed(self, value):
        if value not in (1.0, 5.0, 20.0):
            return False
        if value != self.requested:
            self.requested = float(value)
            self._reset()
            self.robot.simulationSetMode(
                self.robot.SIMULATION_MODE_FAST if value > 1
                else self.robot.SIMULATION_MODE_REAL_TIME)
        return True

    def pace(self):
        sim_now = self.robot.getTime()
        if self.requested > 1:
            deadline = self.wall_anchor + (sim_now - self.sim_anchor) / self.requested
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        wall_now = time.monotonic()
        if wall_now - self.sample_wall >= 1.0:
            self.actual = (sim_now - self.sample_sim) / (wall_now - self.sample_wall)
            self.sample_wall, self.sample_sim = wall_now, sim_now
