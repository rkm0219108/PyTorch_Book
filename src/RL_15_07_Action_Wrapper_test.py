# 載入相關套件
import random

import gymnasium as gym


# 繼承 gym.ActionWrapper 基礎類別
class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, epsilon: float = 0.1) -> None:
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon  # 隨機行動的機率

    def action(self, action: int) -> int:
        # 隨機亂數小於 epsilon，採取隨機行動
        if random.random() < self.epsilon:
            print("Random!")
            return self.env.action_space.sample()
        return action


if __name__ == "__main__":
    env = RandomActionWrapper(gym.make("CartPole-v0", render_mode='human'))

    for _ in range(50):
        env.reset()
        total_reward = 0.0
        while True:
            env.render()
            # 固定往左走
            print("往左走!")
            obs, reward, terminated, truncated, _ = env.step(0)
            done = terminated or truncated
            total_reward += float(reward)
            if done:
                break

        print(f"報酬: {total_reward:.2f}")
    env.close()
