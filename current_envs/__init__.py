from gym.envs.registration import register 

register(
    id="DecisionPolicy-v0", 
    entry_point="current_envs.envs:DecisionPolicy",
)

register(
    id="Persist-v0", 
    entry_point="current_envs.envs:Persist",
)
