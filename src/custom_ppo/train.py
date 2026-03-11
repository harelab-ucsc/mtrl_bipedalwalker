import os
import time
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from envrunner import EnvRunner
from model import PolicyNet, ValueNet
from agent import PPO

#Run an episode using the policy net
def play(policy_net, device=torch.device("cpu")):
    render_env = gym.make("BipedalWalker-v3")

    with torch.no_grad():
        state, _ = render_env.reset()
        total_reward = 0.0
        length = 0

        while True:
            render_env.render()
            state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32, device=device)
            action = policy_net.choose_action(state_tensor, deterministic=True).cpu().numpy()
            state, reward, done, info, _ = render_env.step(action[0])
            total_reward += float(reward)
            length += 1

            if done:
                print("[Evaluation] Total reward = {:.6f}, length = {:d}".format(total_reward, length), flush=True)
                break

    render_env.close()

#Train the policy net & value net using the agent
def train(env, runner, policy_net, value_net, agent, max_episode=5000, device=torch.device("cpu")):
    mean_total_reward = 0
    mean_length = 0
    save_dir = "../../save"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    start_time = time.time()
    last_i = 0
    for i in range(max_episode):
        # run and episode to collect data
        with torch.no_grad():
            mb_states, mb_actions, mb_old_a_logps, mb_values, mb_returns, mb_rewards = runner.run(env, policy_net, value_net)
            mb_advs = mb_returns - mb_values
            mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-6)
        
        # train the model using the collected data
        pg_loss, v_loss, ent = agent.train(mb_states, mb_actions, mb_values, mb_advs, mb_returns, mb_old_a_logps)
        mean_total_reward += mb_rewards.sum()
        mean_length += len(mb_states)
        print("[Episode {:4d}] total reward = {:.6f}, length = {:d}".format(i, mb_rewards.sum(), len(mb_states)))

        # show the current result & save the model
        if i % 200 == 0 and i != 0:            
            print("\n[{:5d} / {:5d}]".format(i, max_episode))
            print("----------------------------------")
            print("actor loss = {:.6f}".format(pg_loss))
            print("critic loss = {:.6f}".format(v_loss))
            print("entropy = {:.6f}".format(ent))
            print("mean return = {:.6f}".format(mean_total_reward / 200))
            print("mean length = {:.2f}".format(mean_length / 200))
            print("average episodes per second = {:.2f}".format((i - last_i) / (time.time() - start_time)))
            print("\nSaving the model ... ", end="")
            torch.save({
                "it": i,
                "PolicyNet": policy_net.state_dict(),
                "ValueNet": value_net.state_dict()
            }, os.path.join(save_dir, "model.pt"))
            print("Done.")
            print()
            # play(policy_net, device=device)
            
            mean_total_reward = 0
            mean_length = 0
            # reset performance
            start_time = time.time()
            last_i = i

if __name__ == "__main__":
    # detect device
    # device = torch.device(
    #     "cuda" if torch.cuda.is_available()
    #     else ("mps" if torch.backends.mps.is_available()
    #           else "cpu")
    # )
    device = torch.device("cpu") # realized half way through this isnt isaaclab
    
    # create the environment
    env = gym.make("BipedalWalker-v3")
    s_dim = env.observation_space.shape[0] # type: ignore
    a_dim = env.action_space.shape[0] # type: ignore
    print(s_dim)
    print(a_dim)

    # create the policy net & value net
    policy_net = PolicyNet(s_dim, a_dim).to(device)
    value_net = ValueNet(s_dim).to(device)
    print(policy_net)
    print(value_net)

    # create the runner
    runner = EnvRunner(s_dim, a_dim, device=device)

    # Create a PPO agent for training2
    agent = PPO(policy_net, value_net, device=device)

    # Train the network
    train(env, runner, policy_net, value_net, agent, device=device)
    env.close()
