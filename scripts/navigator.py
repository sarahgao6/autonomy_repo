#!/usr/bin/env python3

import numpy as np
import rclpy
import scipy.interpolate
import typing as T

from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.grids import StochOccupancyGrid2D
from asl_tb3_msgs.msg import TurtleBotState, TurtleBotControl


class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init                     
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        ########## Code starts here ##########
        for i in range(len(x)):
            if x[i] < self.statespace_lo[i] or x[i] > self.statespace_hi[i]:
                return False
        
        return self.occupancy.is_free(np.asarray(x))

        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        ########## Code starts here ##########
        return np.linalg.norm(np.array(x1) - np.array(x2))
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        neighbors = []
        ########## Code starts here ##########
        for dx in [-self.resolution, 0, self.resolution]:
            for dy in [-self.resolution, 0, self.resolution]:
                if dx == 0 and dy == 0:
                    continue 
                neighbor = self.snap_to_grid((x[0] + dx, x[1] + dy))

                if self.is_free(neighbor):
                    neighbors.append(neighbor)
        ########## Code ends here ##########
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

 

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        ########## Code starts here ##########
        while len(self.open_set) > 0:
            x_current = self.find_best_est_cost_through()
            if x_current == self.x_goal:
                self.path = self.reconstruct_path()
                return True
            self.open_set.remove(x_current)
            self.closed_set.add(x_current)
            for x_neigh in self.get_neighbors(x_current):
                if x_neigh in self.closed_set:
                    continue
                tentative_cost_to_arrive = self.cost_to_arrive[x_current] + self.distance(x_current, x_neigh)
                if x_neigh not in self.open_set:
                    self.open_set.add(x_neigh)
                elif tentative_cost_to_arrive > self.cost_to_arrive[x_neigh]:
                    continue
                self.came_from[x_neigh] = x_current
                self.cost_to_arrive[x_neigh] = tentative_cost_to_arrive
                self.est_cost_through[x_neigh] = tentative_cost_to_arrive + self.distance(x_neigh, self.x_goal)
        return False

        ########## Code ends here ##########

class Navigator(BaseNavigator):
    def __init__(self) -> None:
        super().__init__("navigator")

        # heading controller gain
        self.kp = 2.0

        # trajectory tracking gains
        self.kpx = 2.0
        self.kpy = 2.0
        self.kdx = 2.0
        self.kdy = 2.0

        # tracking controller history
        self.V_PREV_THRESH = 0.0001
        self.V_prev = 0.
        self.t_prev = 0.

    def compute_heading_control(self, state: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:
        heading_error = wrap_angle(goal.theta - state.theta)
        control = TurtleBotControl()
        control.omega = self.kp * heading_error
        return control

    def compute_trajectory_tracking_control(self, state: TurtleBotState, plan: TrajectoryPlan, t: float) -> TurtleBotControl:
        dt = t - self.t_prev

        x_d = float(scipy.interpolate.splev(t, plan.path_x_spline, der=0))
        xd_d = float(scipy.interpolate.splev(t, plan.path_x_spline, der=1))
        xdd_d = float(scipy.interpolate.splev(t, plan.path_x_spline, der=2))
        y_d = float(scipy.interpolate.splev(t, plan.path_y_spline, der=0))
        yd_d = float(scipy.interpolate.splev(t, plan.path_y_spline, der=1))
        ydd_d = float(scipy.interpolate.splev(t, plan.path_y_spline, der=2))

        # avoid the V = 0 singularity
        if self.V_prev < self.V_PREV_THRESH:
            self.V_prev = self.V_PREV_THRESH

        xd = self.V_prev * np.cos(state.theta)
        yd = self.V_prev * np.sin(state.theta)

        u1 = xdd_d + self.kpx * (x_d - state.x) + self.kdx * (xd_d - xd)
        u2 = ydd_d + self.kpy * (y_d - state.y) + self.kdy * (yd_d - yd)

        J = np.array([[np.cos(state.theta), -self.V_prev * np.sin(state.theta)],
                      [np.sin(state.theta),  self.V_prev * np.cos(state.theta)]])
        a, om = np.linalg.solve(J, np.array([u1, u2]))

        V = self.V_prev + a * dt

        # save history for the next step
        self.t_prev = t
        self.V_prev = V

        control = TurtleBotControl()
        control.v = V
        control.omega = om
        return control

    def compute_trajectory_plan(self, state: TurtleBotState, goal: TurtleBotState, occupancy: StochOccupancyGrid2D,
                                resolution: float, horizon: float) -> T.Optional[TrajectoryPlan]:
        # 1. A* path planning
        astar = AStar(
            statespace_lo=(state.x - horizon, state.y - horizon),
            statespace_hi=(state.x + horizon, state.y + horizon),
            x_init=(state.x, state.y),
            x_goal=(goal.x, goal.y),
            occupancy=occupancy,
            resolution=resolution,
        )

        # 2. give up if there is no path, or it is too short to fit a cubic spline
        if not astar.solve() or len(astar.path) < 4:
            return None

        # 5. reset tracking controller history
        self.V_prev = 0.
        self.t_prev = 0.

        # 6. time stamps assuming constant velocity
        v_desired = 0.15
        spline_alpha = 0.05
        path = np.asarray(astar.path)
        segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        ts = np.concatenate([[0], np.cumsum(segment_lengths / v_desired)])

        # 7. cubic spline fit
        path_x_spline = scipy.interpolate.splrep(ts, path[:, 0], k=3, s=spline_alpha)
        path_y_spline = scipy.interpolate.splrep(ts, path[:, 1], k=3, s=spline_alpha)

        # 8. trajectory plan
        return TrajectoryPlan(
            path=path,
            path_x_spline=path_x_spline,
            path_y_spline=path_y_spline,
            duration=ts[-1],
        )


if __name__ == "__main__":
    rclpy.init()
    node = Navigator()
    rclpy.spin(node)
    rclpy.shutdown()




