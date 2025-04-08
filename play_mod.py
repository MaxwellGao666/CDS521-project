# play_mod.py
import gym
import numpy as np  # 必须导入
from utils import FrameStack, preprocess_frame
from ppo import PPO

# 初始化环境
env = gym.make('CarRacing-v2', render_mode='human',new_step_api=False)

# 加载模型
model = PPO(
    input_shape=(84, 84, 4),
    num_actions=3,
    action_min=env.action_space.low,
    action_max=env.action_space.high,
    model_checkpoint="./models/CarRacing-v2/step280000.ckpt"  # 替换为实际路径
)

# 初始化帧栈
initial_frame = env.reset()
frame_stack = FrameStack(initial_frame, stack_size=4, preprocess_fn=preprocess_frame)  # 闭合括号

# 运行推理
total_reward = 0
for t in range(1000):
    env.render()
    state = frame_stack.get_state()
    action, _ = model.predict(state[np.newaxis, ...], use_old_policy=False, greedy=True)
    next_frame, reward, done, _ = env.step(action[0])
    frame_stack.add_frame(next_frame)
    total_reward += reward
    if done:
        print(f"Total Reward: {total_reward}")
        break

env.close()