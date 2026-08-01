# ffmpeg.exe 放在Path的路徑下
# mac: brew install ffmpeg
# 載入相關套件
import gymnasium as gym

# 載入環境
env = gym.make("CartPole-v0", render_mode="rgb_array")

# 錄影
env = gym.wrappers.RecordVideo(env, "recording")

# 實驗
for _ in range(50):
    total_reward = 0.0
    obs, info = env.reset()

    while True:
        env.render()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        if done:
            break

    print(f"報酬: {total_reward:.2f}")

env.close()
