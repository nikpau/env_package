import math
from typing import Callable
import gym
from gym import spaces
import numpy as np
import persistpy.vessels as v
import persistpy.river as r
from persistpy.parse_settings import load_config
import configparser as c
import random
import sys

np.set_printoptions(suppress=True,linewidth=sys.maxsize,threshold=sys.maxsize)

sys.path.append("../persist_python")

"""
Plan for Environment:
    - Master Policy Output: 
        - [+1: increase level, 0: nop, -1: decrease level]
        - [0,1,2,3] -> one action per ovetaking level
        
        
    - Agent Observation space:
        - Relative positions and velocities of all vessels
        - Overtaking level of all vessels (incl. agent)
        - Desired engine power
        - River characteristics (mean stream vel, river profile)
            - Desired speed / leader speed >= 1.2 [experimental]
            - Calculate lookahead distance required for an entire overtaking maneuver
            - Look up river characteristics in interval [curr. pos, curr. pos + lookahead_dist]
        - Make decision every n timesteps [n = 5; try different values]
    - Reward:
        - 0: no violation of safety distance; -1: violation of former
        - Safety distance modeled as additional vessel width and length 
                (every vessel gains invisible width and length to implement safety distance)
        - If collision with invisible border -> -5 reward, else 0
        - [Reward for staying in high overtaking level = 1e-3*overtaking_level]
        - [reward: average speed if overtaking successful | high negative if unsuccessful]
    - Max Episode steps: THINK!!
    - Vessels dissapear when outside lookahead distance and respawn inside [both direction]
    - Non-Agent vessels move according to the lateral and longitudinal policy
    - There are n vessels placed on arbitary positions around the agent ship
    - The ships are spawned in the logical river cooridnate system 

"""
# AGENT ID === 0
class DecisionPolicy(gym.Env):
    def __init__(self, vessels = 10) -> None:

        # Path to initial config file. This is still needed to initialize the
        # river correctly, although all values get overwritten anyways
        #self.c_path = "/home/neural/Dropbox/TU Dresden/persist_python/train.ini"
        self.c_path = "/home/s2075466/persist_data/train.ini"
        
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
        self.agent_x_start = 10000.
        self.agent_y_start = 190.
        self.agent_des_power = 1E6

        # Agent initial lookahead and behind distance in meters
        self.agent_lookahead = 2000.
        self.agent_lookbehind = 300.
        
        
        # Load config for dT init
        sim_conf = load_config(self.c_path,"Simulation")
        self.dT = sim_conf["speed-up"]
        
        # Timestep counter
        self.tc = counter()
        self.current_timestep = self.tc()
        
        self.max_episode_lenth = 2000

        # Make the rhine data globally available to the methods of the class
        self.r = r.River(self.c_path)
        self.r.water_depth = (self.r.water_depth - 1).clip(min = 0) # Reduce river width by altering water depth
        
        # Constants for normalization ---------------------------------
        
        # Mean and sd of stream velocity for normalizing
        self.mean_str_vel = np.mean(self.r.stream_vel.flatten())
        self.sd_str_vel = np.std(self.r.stream_vel.flatten())

        # Same as above for water depth
        self.mean_water_depth = np.mean(self.r.water_depth.flatten())
        self.sd_water_depth = np.std(self.r.water_depth.flatten())
        
        self.max_power = 1E6
        self.max_x = self.r.water_depth.shape[0] * self.bpd - (1.5*self.agent_lookahead)
        self.max_y = 500.
        # ------------------------------------------------------------

        # Set the x Goal the agent has to arrive at.
        self.xg = self.max_x
        
        # Anything reward
        self.r_const = 1.05
        
        # Gym inherits
        self.observation_space = spaces.Box(high=np.inf, low=-np.inf, shape=(11*self.n_vessel,))
        self.action_space = spaces.Discrete(4) # {0,1,2,3}


    def reset(self):
        
        # Randomize the initial starting position of the agent
        self.agent_x_start = np.float(random.randrange(20_000, 90_000, 500))

        # Generate random directions for non-agent-vessels
        dirs = _rand_directions(self.n_vessel - 1)

        # Generate y-coordinates for vessel spawn based on the directions of the vessels
        y_locs = _generate_ylocs(dirs)
        y_locs = np.hstack([self.agent_y_start,y_locs])
        
        # Load an empty ship instance in order to manually override its values
        self.v = v.Ships(self.r, self.c_path)

        # Override vessel properties according to train env
        self.v.num_ships = self.n_vessel
        self.v.direction = np.append(1,dirs)
        self.v.y_location = y_locs
        self.v.length = np.array([100.] * self.n_vessel)
        self.v.width = np.array([10.] * self.n_vessel)
        self.v.eff_width = self.v.width
        self.v.mass = np.array([100E3] * self.n_vessel)
        self.v.overtaking_level = [0] * self.n_vessel
        self.v.desired_power = np.append(self.agent_des_power,_rand_powers(self.n_vessel - 1))
        self.v.vx = 1. * self.v.direction

        # Manually override x-locations for every vessel
        # Currently the agent has the lowest x-position. All other vessels are
        # spawned in equal distances from each other
        #------------------------------------------------------------------------
        #
        #                                     <-[VESSEL]
        #         |-equal dist-|                         |-equal dist-| 
        # [VESSEL]->            [AGENT]->                             [VESSEL]->
        #
        #------------------------------------------------------------------------
        # |equal dist| is currently set to 300, however this is arbitrary
        equaldist = 500
        self.v.x_location = np.array([
            np.array([self.agent_x_start]), 
            self.agent_x_start +  np.linspace(equaldist, (self.n_vessel-1) * equaldist, self.n_vessel-1)
        ])
        self.v.x_location = np.hstack(self.v.x_location)
        
        # Calculate the current distance from the agent to its goal
        self.dist_to_goal = self.xg - self.v.x_location[self.AGENT_ID]

        # Get the normalized river properties for the lookahead and lookbehind dist
        self.river_state = self._get_river_properties()
        self.vessel_state = self._get_vessel_propterties()

        self.state = np.append(self.vessel_state, np.hstack(self.river_state))

        return self.state
    
    def step(self, action: int):
        action = int(action)
        self.v.overtaking_level[self.AGENT_ID] = action
        
        wd = self.r.get_water_depth(self.v)
        r_prof = self.r.get_river_profile(self.v)
        str_vel = self.r.mean_stream_vel(self.v)

        try:
            for id in self.v.ship_id:
                self.v.simulate_timestep(id, self.r, self.dT,wd,r_prof,str_vel)
        except RuntimeError as e:
            if e[1] == self.AGENT_ID:
                reward = self._calc_reward(self.r_const,crash=True)
                state = np.array([self._get_vessel_propterties(),
                                  self._get_river_properties()])
                done = True
                return state, reward, done, {e}
            else:
                raise RuntimeError(f"Vessel {e[1]} crashed. Pls investigate!")
        
        reward = self._calc_reward(self.r_const)
        done = self._done()
        self.current_timestep = self.tc()
        
        # Respawn vessels that are invisible to the agent
        self._respawn(self._invisible)
        
        self.river_state = self._get_river_properties()
        self.vessel_state = self._get_vessel_propterties()

        self.state = np.append(self.vessel_state, self.river_state)
        return self.state, reward, done , {}
        

    # Return the stream velocity and the water depth for the entire river
    # from the lookbehind to lookahead distance.
    # Also the entire x range from lookbehind to lookahead is saved internally as
    # self.visible_range = [min(lookbehind), max(lookahead)]
    def _get_river_properties(self) -> np.array:
        
        wd = self.r.get_water_depth(self.v)
        str_vel = self.r.mean_stream_vel(self.v)
        river_prof = self.r.get_river_profile(self.v)


        # TODO Order of observations??
        obs = np.hstack([wd,str_vel,river_prof])
        return obs
    
    # Receive vessel properties and normalize them before return
    def _get_vessel_propterties(self) -> np.array:
        obs = np.concatenate([
            self.v.overtaking_level,
            np.tanh(self.v.ax),
            np.tanh(self.v.ay),
            np.tanh(self.v.vx),
            np.tanh(self.v.vy),
            self.v.x_location / self.max_x,
            self.v.y_location / self.max_y,
            self.v.desired_power / self.max_power])

        return obs
    
    # Check if any vessel is outside the visible range of the agent
    # and return its index.
    def _invisible(self) -> list:
        l, u = self.agent_lookbehind, self.agent_lookahead
        agx = self.v.x_location[self.AGENT_ID]
        rm = []
        for v in range(1,self.n_vessel): # No need to check the agent
            xpos = self.v.x_location[v]
            if xpos < agx - l or xpos > agx + u:
                rm.append(v)
            else:
                continue
        return rm

    # Take all vessels outside the visible range and let them reappear randomly 
    # at the beginning or end of the river. TODO See notes to discuss with Fabian
    def _respawn(self, _c_i: Callable) -> None:
        to_set = ["desired_power","y_location","x_location"] # Attrs to be set 
        _, u = self.agent_lookbehind, self.agent_lookahead
        agx = self.v.x_location[self.AGENT_ID]
        IDs = _c_i() # Fetch vessel IDs that need respawning
        if not IDs:
            pass
        for id in IDs:
            pd, = _rand_powers(1)
            x = agx + u - 10  # Spawn new vessel 10 meter inside the visible range
            y = self.r.midpoint_index[int(x//self.bpd)] * self.bpd
            if self._spawn_collision(x): # Check if placement would cause collision
                continue
            specs = [pd,y,x]
            for i,val in enumerate(to_set):
                a = getattr(self.v, val)
                a[id] = specs[i]
                setattr(self.v, val, a)
            self._reset_dynamics(id)

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
    def _calc_reward(self, c: float, crash = False) -> float:
        if crash:
            return -20.
        n_dist_to_goal = self.xg - self.v.x_location[self.AGENT_ID]
        dist_diff = self.dist_to_goal - n_dist_to_goal
        
        # Distance reward
        dr = np.sign(dist_diff) * pow(c, dist_diff)
        
        # Angle between agent and its leader
        # cos_theta = self._angle_to_closest_vessel(self._closest_vessel)
        # theta = math.acos(cos_theta) * 180/math.pi
        # print(f"Theta =  {theta}")

        # if theta > 45:
        #     angle_reward = 1.
        # else:
        #     angle_reward = 0.
        
        # if self.v.overtaking_level[self.AGENT_ID] != 0:
        #     otl_reward = -0.01
        # else:
        #     otl_reward = 0.
        
        # self.dist_to_goal = n_dist_to_goal
        
        return dr

    # Find the closest vessel to the agent in the same dir as the agent
    def _closest_vessel(self, dir = 1) -> int:
        
        # Get agent x
        agx = self.v.x_location[self.AGENT_ID]
        
        # Get all vessels with same dir and larger x position as agent
        if dir == 1:
            ids = self.v.ship_id[np.where((self.v.direction == 1) & (self.v.x_location > agx))]
        elif dir == -1:
            ids = self.v.ship_id[np.where((self.v.direction == -1) & (self.v.x_location < agx))]
        
        # Get all differences in distances
        diff = self.v.x_location[ids] - agx
        
        # Find lowest index
        if ids.size == 0:
            return []
        
        idx = np.argmin(diff)
        
        return ids[idx]
    
    def _angle_to_closest_vessel(self, find_vessel: Callable) -> float:
        
        # Get the index of the closest vessel
        closest = find_vessel()
        
        if not closest:
            return 0
        
        # Get x and y of closest vessel
        closest_x = self.v.x_location[closest]
        closest_y = self.v.y_location[closest]

        # Get x and y for agent
        agx = self.v.x_location[self.AGENT_ID]
        agy = self.v.y_location[self.AGENT_ID]
        
        xdiff = abs(closest_x - agx)
        ydiff = abs(closest_y - agy) # For both directions

        # Distance between vessels
        dbv = math.sqrt(xdiff**2 + ydiff**2)
        
        # Cosine of angle in x direction between agent and closest vessel
        cos_theta = xdiff / dbv
        
        return cos_theta
        
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


# Helper functions
def _rand_directions(n_vessels: int) -> np.array:
    return np.array([np.random.choice([1,-1]) for _ in range(n_vessels)])

# Ships are being placed on fixed y-coords during spawn
def _generate_ylocs(directions: np.array) -> np.array:
    return np.array([190. if n==1 else 310. for n in directions])

def _rand_powers(n_vessels: int) -> np.array:
    return np.array([random.randrange(3E5, 4E5, 5E4) for _ in range(n_vessels)])

# Timestep counter
def counter() -> Callable:
    i = 0
    def incr() -> int:
        nonlocal i
        i += 1
        return i
    return incr

# Easy to read power function
def pow(base: float,exponent: float) -> float:
    return base**exponent

# Oh man. This is awful. Consider generating a completely new
# config file upon env init.
def _overwrite_config(path: str, n_vessels: int) -> None:
    dummy = ' '.join(map(str,np.array([1.]*n_vessels)))
    conf = c.ConfigParser()
    conf.read(path)
    conf.set("Ships","vessel-count",str(n_vessels))
    conf.set("Ships","lengths",dummy)
    conf.set("Ships","widths",dummy)
    conf.set("Ships","masses",dummy)
    conf.set("Ships","y-locations",dummy)
    conf.set("Ships","directions",dummy)
    with open(path, "w") as f:
        conf.write(f)

# ---------------------------------------------

# env = DecisionPolicy(10)
# f = env.reset()
# print(f.shape)
# rs = []
# ax = []
# for _ in range(1000):
#     s,r,d,_ = env.step(1)
#     rs.append(r)
#     # ax.append(env.v.ax[0])
#     print(f"Step: {env.current_timestep} Cum Reward: {np.sum(rs)}")
#     print(s.shape)
