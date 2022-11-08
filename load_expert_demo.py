from stable_baselines3.ppo import PPO


def get_expert():
    return PPO.load("./experts/LunarLander-v2/lunarlander_expert")

def get_expert_performance(env, expert):
    Js = []
    for _ in range(100):
        obs = env.reset()
        J = 0
        done = False
        hs = []
        while not done:
            action, _ = expert.predict(obs)
            obs, reward, done, info = env.step(action)
            hs.append(obs[1])
            J += reward

        Js.append(J)
    ll_expert_performance = np.mean(Js)
    return ll_expert_performance
