# CarRacing-run_random_agent.py
import gym

render = True
n_episodes = 1
env = gym.make('CarRacing-v2')  

print(env.action_space)
print(env.observation_space)

rewards = []
for i_episode in range(n_episodes):
    observation = env.reset()
    sum_reward = 0
    for t in range(1000):
        if render:
            env.render()
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        sum_reward += reward
        if done or t == 999:
            print(f"Episode {i_episode} finished after {t+1} timesteps")
            print(f"Reward: {sum_reward}")
            rewards.append(sum_reward)
            break

env.close()