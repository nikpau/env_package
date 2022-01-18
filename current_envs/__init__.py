from gym.envs.registration import register 

register(
    id="DecisionPolicy-v0", 
    entry_point="current_envs.envs:DecisionPolicy",
)
