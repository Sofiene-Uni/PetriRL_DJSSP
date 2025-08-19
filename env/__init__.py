from gymnasium.envs.registration import register


register(
    id="petrirl-fms-v0",
    entry_point="petrirl.envs.fms.gym_env:FmsEnv",
    )

register(
    id="petrirl-moo-v0",
    entry_point="petrirl.envs.moo.gym_env:MooEnv",
    )

register(
    id="petrirl-dft-v0",
    entry_point="petrirl.envs.dft.gym_env:DftEnv",
    )

register(
    id="petrirl-shu-v0",
    entry_point="petrirl.envs.shu.gym_env:ShuEnv",
    )



