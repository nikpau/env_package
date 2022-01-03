from gym.envs.registration import register 

register(
    id="MyMountainCar-v0", 
    entry_point="current_envs.envs:MountainCar",
)

register(
    id="Ski-v0", 
    entry_point="current_envs.envs:Ski",
)

register(
    id="ObstacleAvoidance-v0", 
    entry_point="current_envs.envs:ObstacleAvoidance",
)

register(
    id="DecisionPolicy-v0", 
    entry_point="current_envs.envs:DecisionPolicy",
)

register(
    id="Persist-v0", 
    entry_point="current_envs.envs:Persist",
)
