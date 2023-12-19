import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from battle_env import MAgentBattle
from agent.agent_rl.agent_rl import AgentRL  # Import your AgentRL class here
from agent.ppo.ppo_agent import AgentPPO


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
                        help="total timesteps of the experiments")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=5,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=100,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=16,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


if __name__ == "__main__":

    args = parse_args()

    env = MAgentBattle(visualize=False, eval_mode=False, obs_flat=True)
    
    # Create two instances of AgentRL for self-play
    
    agent1 = AgentPPO(dim_obs=env.dim_obs, dim_action=env.dim_action)
    agent2 = AgentRL(dim_obs=env.dim_obs, dim_action=env.dim_action)

    num_rl_win = 0
    num_game = 0

    (obs1, obs2), done, (valid1, valid2) = env.reset()

    optimizer = optim.Adam(agent1.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs, env.dim_obs))
    actions = torch.zeros((args.num_steps, args.num_envs, env.dim_action))
    logprobs = torch.zeros((args.num_steps, args.num_envs))
    rewards = torch.zeros((args.num_steps, args.num_envs))
    dones = torch.zeros((args.num_steps, args.num_envs))
    values = torch.zeros((args.num_steps, args.num_envs))
    valids = torch.zeros((args.num_steps, args.num_envs))

    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(obs1[0])
    next_done = torch.zeros(args.num_envs)
    next_valid = torch.ones(args.num_envs)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            valids[step] = next_valid

            #### agent1: PPO
            with torch.no_grad():
                action1, logprob, _, value = agent1.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = torch.nn.functional.one_hot(action1, env.dim_action).to(torch.float32)
            logprobs[step] = logprob

            #### agent2: RL
            with torch.no_grad():
                action2, _, _, _ = agent2.get_action_and_value(next_obs)

            #### Environment step
            (obs1, obs2), (reward1, reward2), (done1, done2, done_env), (valid1, valid2) = \
                env.step(action1.cpu().numpy().astype(np.int32), action2.cpu().numpy().astype(np.int32))
            if done_env:
                num_game += 1

                if len(obs1[1]) > len(obs2[1]):
                    winstr = 'RL Win'
                    num_rl_win += 1
                elif len(obs1[1]) < len(obs2[1]):
                    winstr = 'Self-play Win'
                else:
                    winstr = 'Draw'
                print(f'ppo win: {num_rl_win / num_game:.3f} | rl win: {1 - num_rl_win / num_game:.3f}')
                # print(action1)
                (obs1, obs2), (_, _, _), (_, _) = env.reset()
            rewards[step][torch.tensor(valid1, dtype=torch.bool)] = torch.tensor(reward1).view(-1)
            next_obs, next_done, next_valid = torch.Tensor(obs1[0]), torch.Tensor(done1), torch.Tensor(valid1)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent1.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_valids = valids.reshape(-1).to(torch.bool)

        b_obs = obs.reshape(-1, env.dim_obs)[b_valids]
        b_logprobs = logprobs.reshape(-1)[b_valids]
        b_actions = actions.reshape(-1, env.dim_action)[b_valids]
        b_advantages = advantages.reshape(-1)[b_valids]
        b_returns = returns.reshape(-1)[b_valids]
        b_values = values.reshape(-1)[b_valids]

        batch_size_new = b_valids.sum()
        minibatch_size_new = int(batch_size_new // args.num_minibatches)
        b_inds = np.arange(batch_size_new)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size_new, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent1.get_action_and_value(b_obs[mb_inds],
                                                                               torch.argmax(b_actions.long()[mb_inds], dim=1))
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs
