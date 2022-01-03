import math
from typing import Callable
import gym
from gym import spaces
import numpy as np
from persist.policies import UniPol
import persist.vessels as v
import persist.river as r
from persist.parse_settings import load_config
from persist.visualization import Plotter
import configparser as c
import random
import warnings
import sys

# AGENT ID === 0

class Persist(gym.Env):
    def __init__(self, vessels=10) -> None:

        # Path to initial config file. This is still needed to initialize the
        # river correctly, although all values get overwritten anyways
        self.c_path = "/home/niklaspaulig/Dropbox/TU Dresden/persist_python/train.ini"

        # Overwrite the number of ships to initialize.
        _overwrite_config(self.c_path, vessels)

        # Basepoint distance for discretization of river
        self.bpd = 20.

        # Init Gym Env
        super().__init__()

        # Number of vessels to be present at any time
        self.n_vessel = vessels

        # Set the ID of the agent
        self.AGENT_ID = 0

        # Set the inital properties of the agent
        self.agent_y_start = 190.
        self.agent_des_power = 1E6

        # Agent initial lookahead and behind distance in meters
        self.agent_lookahead = 2000.
        self.agent_lookbehind = 300.

        # Load config for dT init
        sim_conf = load_config(self.c_path, "Simulation")
        self.dT = sim_conf["speed-up"]

        # Timestep counter
        self.tc = counter()
        self.current_timestep = self.tc()

        self.max_episode_steps = 2000

        # Make the rhine data globally available to the methods of the class
        self.r = r.River(self.c_path)
        self.v = v.Ships(self.r, self.c_path)

        # Initializer renderer
        self.plotter = Plotter(self.r,self.v,self.c_path)

        self.max_power = 1E6
        self.max_x = self.r.water_depth.shape[0] * \
            self.bpd - (1.5*self.agent_lookahead)
        self.max_y = 500.

        # List of vessels that manually need respawning
        # TODO This is crap. Integrate this into the respawn function
        self.resp = []
        # ------------------------------------------------------------

        # Set the x Goal the agent has to arrive at.
        self.xg = self.max_x
        self.opt_dist = 60.  # Optimal distance to the inferred fairway border
        self.crash_dist = 5  # Minimum distance from border causing a crash

        # Anything reward
        self.r_const = 1.05

        # Gym inherits
        self.observation_space = spaces.Box(
                high=np.inf, low=-np.inf, shape=((7*self.n_vessel+11),))

        self.action_space = spaces.Box(low=-1., high=1., shape=(2, 1))

    def reset(self):

        # Randomize the initial starting position of the agent
        self.agent_x_start = np.float(random.randrange(20_000, 30_000, 100))

        # Generate random directions for non-agent-vessels
        dirs = _rand_directions(self.n_vessel - 1)

        # Generate y-coordinates for vessel spawn based on the directions of the vessels
        y_locs = _generate_ylocs(dirs)
        y_locs = np.hstack([self.agent_y_start, y_locs])

        # Load an empty ship instance in order to manually override its values
        self.v = v.Ships(self.r, self.c_path)

        # Override vessel properties according to train env
        self.v.num_ships = self.n_vessel
        self.v.direction = np.append(1, dirs)
        self.v.y_location = y_locs
        self.v.length = np.array([100.] * self.n_vessel)
        self.v.width = np.array([10.] * self.n_vessel)
        self.v.eff_width = self.v.width
        self.v.mass = np.array([100E3] * self.n_vessel)
        self.v.desired_power = np.append(self.agent_des_power, _rand_powers(self.n_vessel - 1))
        self.v.vx = 1. * self.v.direction

        # Manually override x-locations for every vessel
        # Currently the agent has the lowest x-position. All other vessels are
        # spawned in equal distances from each other
        # ------------------------------------------------------------------------
        #
        #                                     <-[VESSEL]
        #         |-equal dist-|                         |-equal dist-|
        # [VESSEL]->            [AGENT]->                             [VESSEL]->
        #
        # ------------------------------------------------------------------------
        # |equal dist| is currently set to 300, however this is arbitrary
        equaldist = 500
        self.v.x_location = np.array([
            np.array([self.agent_x_start]),
            self.agent_x_start +
            np.linspace(equaldist, (self.n_vessel-1)
                        * equaldist, self.n_vessel-1)
        ], dtype=object)
        self.v.x_location = np.hstack(self.v.x_location)

        # Calculate the current distance from the agent to its goal
        self.dist_to_goal = self.xg - self.v.x_location[self.AGENT_ID]

        wd = self.r.get_water_depth(self.v)
        r_prof = self.r.get_river_profile(self.v)
        str_vel = self.r.mean_stream_vel(self.v)

        # Get the current state of the agent (non-normalized)
        self.state = self.v.pol[self.AGENT_ID].observe(wd, str_vel, r_prof)

        return self.state

    def step(self, action):

        # Agent Timestep ------------------------------------------
        wd = self.r.get_water_depth(self.v)
        r_prof = self.r.get_river_profile(self.v)
        str_vel = self.r.mean_stream_vel(self.v)

        lat_action, long_action = action

        self.v.power[self.AGENT_ID] = np.maximum(0, self.v.desired_power[self.AGENT_ID] * long_action)

        # Calculate lateral accerlation, squat and cf value
        # The only error that can appear here, is calculating nonsensical acceleations
        # due to too low water depth.
        try:
            acc, squat, cf = self.v.pol[self.AGENT_ID].compute_acc(wd, str_vel, r_prof, self.dT)
        except RuntimeError as e:
            reward = self._calc_reward(self.r_const, crash=True)
            done = True
            self.state = self.v.pol[self.AGENT_ID].observe(wd, str_vel, r_prof)
            return self.state, reward, done, {e}

        self.v.ax[self.AGENT_ID] = self.v.direction[self.AGENT_ID] * acc
        self.v.squat[self.AGENT_ID] = squat
        self.v.heading_cf[self.AGENT_ID] = cf

        new_vx = self.v.vx[self.AGENT_ID] + self.v.ax[self.AGENT_ID] * self.dT
        new_x_location = self.v.x_location[self.AGENT_ID] + \
            0.5 * (self.v.vx[self.AGENT_ID] + new_vx) * self.dT
        self.v.x_location[self.AGENT_ID] = new_x_location
        self.v.vx[self.AGENT_ID] = new_vx

        # Lateral simulation
        self.v.ay[self.AGENT_ID] = lat_action * \
            self.v.pol[self.AGENT_ID].upper_acc_bound

        new_vy = self.v.vy[self.AGENT_ID] + self.v.ay[self.AGENT_ID] * self.dT

        new_y_location = self.v.y_location[self.AGENT_ID] + \
            0.5 * (self.v.vy[self.AGENT_ID] + new_vy) * self.dT
        self.v.y_location[self.AGENT_ID] = new_y_location
        self.v.vy[self.AGENT_ID] = new_vy

        self.v.compute_heading_from_cf(self.AGENT_ID)

        # No, this is not how any of this works...fix it!
        try:
            self.dist_to_border, self.dist_to_up, self.dist_to_lo = self.v.pol[self.AGENT_ID]._dist_to_aground()
        except RuntimeError as e:
            reward = self._calc_reward(self.r_const, crash=True)
            done = True
            self.state = self.v.pol[self.AGENT_ID].observe(wd, str_vel, r_prof)
            return self.state, reward, done, {e}

        self.dist_to_any_border = np.minimum(self.dist_to_up, self.dist_to_lo)

        # Crash handling. Check if any vessel polygon intersects with the agent
        # polygon.
        """
        for id in range(1, self.v.num_ships):
            if self.v.crash_heading_box[self.AGENT_ID].intersects(self.v.crash_heading_box[id]):
                reward = self._calc_reward(self.r_const, crash=True)
                done = True
                self.state = self.v.pol[self.AGENT_ID].observe(
                    wd, str_vel, r_prof)
                return self.state, reward, done, {}
        """
        
        # Crash handling. Check if the agent vessel intersects with any other vessel.
        for id in range(1, self.v.num_ships):
            if self.v.heading_box[self.AGENT_ID].intersects(self.v.heading_box[id]):
                reward = self._calc_reward(self.r_const, crash=True)
                done = True
                self.state = self.v.pol[self.AGENT_ID].observe(
                    wd, str_vel, r_prof)
                return self.state, reward, done, {}
            
        # Absolute distance to any border
        if self.dist_to_any_border < self.crash_dist:
            reward = self._calc_reward(self.r_const, crash=True)
            done = True
            self.state = self.v.pol[self.AGENT_ID].observe(wd, str_vel, r_prof)
            return self.state, reward, done, {}

        reward = self._calc_reward(self.r_const)
        done = self._done()
        self.current_timestep = self.tc()

        # End of Agent----------------------------------------------------------

        # Simulate dummy timestep for all non-agent vessels
        # If a vessel is driven into a non-drivable area, its
        # movement is skipped and it is added to the respawn list.
        for id in range(1, self.v.num_ships):
            if wd[id] < self.r.min_water_under_keel:
                warnings.warn("Vessel Placement failed. Water too shallow. Skipping...")
                self.resp.append(id)
                continue
            else:
                self.v.dummy_timestep(id, self.r, self.dT, wd, r_prof, str_vel)

        # Respawn vessels that are invisible to the agent
        self._respawn(self._invisible)

        # Observe the next state
        self.state = self.v.pol[self.AGENT_ID].observe(wd, str_vel, r_prof)

        return self.state, reward, done, {}

    # Check if any vessel is outside the visible range of the agent
    # and return its index.
    def _invisible(self) -> list:
        l, u = self.agent_lookbehind, self.agent_lookahead
        agx = self.v.x_location[self.AGENT_ID]
        rm = self.resp
        for v in range(1, self.n_vessel):  # No need to check the agent
            xpos = self.v.x_location[v]
            if xpos < agx - l or xpos > agx + u:
                rm.append(v)
            else:
                continue
        return rm

    # Take all vessels outside the visible range and let them reappear randomly
    # at the beginning or end of the river. TODO See notes to discuss with Fabian
    def _respawn(self, _c_i: Callable) -> None:
        to_set = ["desired_power", "y_location", "x_location"]  # Attrs to be set
        _, u = self.agent_lookbehind, self.agent_lookahead
        agx = self.v.x_location[self.AGENT_ID]
        IDs = _c_i()  # Fetch vessel IDs that need respawning
        if not IDs:
            pass
        for id in IDs:
            pd, = _rand_powers(1)
            x = agx + u - 10  # Spawn new vessel 10 meter inside the visible range
            y = self.r.midpoint_index[int(x//self.bpd)] * self.bpd
            # Check if placement would cause collision
            if self._spawn_collision(x):
                continue
            specs = [pd, y, x]
            for i, val in enumerate(to_set):
                a = getattr(self.v, val)
                a[id] = specs[i]
                setattr(self.v, val, a)
            self._reset_dynamics(id)

        # Reset the manual respawn list
        self.resp = []

    # Check whether the spawning of a new vessel would cause a collision
    # i.e See if a vessel is present 2 time its length around the xloc of the vessel to be spawned
    def _spawn_collision(self, spawn_loc: float) -> bool:
        for id in self.v.ship_id:
            xpos = self.v.x_location[id]
            l = self.v.length[id]
            if abs(spawn_loc - xpos) < 2*l:
                return True
            else:
                continue
        return False

    # Calculate reward (EXPERIMENTAL)
    # Use reward from Guo et. al (2021)
    # TODO: Impl social forces for other vessels
    def _calc_reward(self, c: float, crash=False) -> float:

        n_dist_to_goal = self.xg - self.v.x_location[self.AGENT_ID]
        dist_diff = self.dist_to_goal - n_dist_to_goal

        # Distance reward
        dr = np.sign(dist_diff) * pow(c, dist_diff)

        # Get the distance to the fairway border and
        # the reward based on it
        lane_reward = self._lane_reward(self.dist_to_any_border)

        ttc = self._ttc_reward()

        # Overwrite the current distance to the goal
        # with the new distance
        self.dist_to_goal = n_dist_to_goal

        if crash:
            # TODO Relate to lane and distance reward
            return -0.1 * (dr + lane_reward + ttc)
        return dr + lane_reward + ttc

    # Reset x and y dynamics for a given vessel ID
    def _reset_dynamics(self, ID: int) -> None:
        self.v.vx[ID] = np.array([1. if self.v.direction[ID] == 1 else -1.])
        self.v.vy[ID] = 0.
        self.v.ax[ID] = 0.
        self.v.ay[ID] = 0.

    # Set the done flag and reset the timestep counter
    # if max-episode-length is reached
    def _done(self) -> bool:
        return False

    def _lane_reward(self, dist: float) -> float:

        def base(x):
            return -1/(0.1*x)

        if dist > self.crash_dist and dist < self.opt_dist:
            return base(dist) - base(self.opt_dist)
        else:
            return -pow(dist - self.opt_dist, 2)/pow(self.opt_dist, 2)
    
    # Calculate reward for time to collision. If vessel is more than 
    def _ttc_reward(self):
        
        # Get the lowest ttc
        min_ttc = np.amin(self.v.pol[self.AGENT_ID].ttc)
        
        if min_ttc > 60.:
            return 0.
        
        def int_sqrt(x, n = 5):
            return math.sqrt(n*x)

        return int_sqrt(min_ttc) - int_sqrt(self.opt_dist)

    def render(self, mode="human"):

        if mode == "human":
            self.plotter.update()
        else:
            raise NotImplementedError(
                "Only human render mode available for now")


# Helper functions
def _rand_directions(n_vessels: int) -> np.array:
    return np.array([np.random.choice([1, -1]) for _ in range(n_vessels)])

# Ships are being placed on fixed y-coords during spawn
def _generate_ylocs(directions: np.array) -> np.array:
    return np.array([190. if n == 1 else 310. for n in directions])

def _rand_powers(n_vessels: int) -> np.array:
    return np.array([random.randrange(3E5, 6E5, 5E4) for _ in range(n_vessels)])

# Timestep counter
def counter() -> Callable:
    i = 0

    def incr() -> int:
        nonlocal i
        i += 1
        return i
    return incr

# Easy to read power function
def pow(base: float, exponent: float) -> float:
    return base**exponent

# This is awful. Consider generating a completely new
# config file upon env init.
def _overwrite_config(path: str, n_vessels: int) -> None:
    dummy = ' '.join(map(str, np.array([1.]*n_vessels)))
    conf = c.ConfigParser()
    conf.read(path)
    conf.set("Ships", "vessel-count", str(n_vessels))
    conf.set("Ships", "lengths", dummy)
    conf.set("Ships", "widths", dummy)
    conf.set("Ships", "masses", dummy)
    conf.set("Ships", "y-locations", dummy)
    conf.set("Ships", "directions", dummy)
    with open(path, "w") as f:
        conf.write(f)

# ---------------------------------------------
