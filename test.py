import gym
import magical

# magical.register_envs() must be called before making any Gym envs
magical.register_envs()

# creating a demo variant for one task
env = gym.make('FindDupe-Demo-v0')
env.reset()
env.render(mode='human')
env.close()

# We can also make the test variant of the same environment, or add a
# preprocessor to the environment. In this case, we are creating a
# TestShape variant of the original environment, and applying the
# LoRes4E preprocessor to observations. LoRes4E stacks four
# egocentric frames together and downsamples them to 96x96.
env = gym.make('FindDupe-TestShape-LoRes4E-v0')
init_obs = env.reset()
print('Observation type:', type(obs))  # np.ndarray
print('Observation shape:', obs.shape)  # (96, 96, 3)
env.close()