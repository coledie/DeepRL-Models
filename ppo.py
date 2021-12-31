"""
Proximal policy optimization with a clipped objective function.
- on-policy
- model based
- continuous

Proximal policy optimization takes an agent's behavior policy
represented in a deep neural network and tunes it as a whole
via gradient descent. This is a model(neural network) based,
on-policy reinforcement learning approach to learning to play games.

In this, two separate neural networks are used, a policy network that
determines how the agent will react to the environment and a value
network that approximates the long term value of a given state assuming
the current policy is used.
The value network simply learns to approximate long term, potential
rewards that come after the state by using the current policy.
This is learned via gradient descent on the mean squared error
between its output and the calculated rewards-to-go. Based on the
approximated state values, the policy network's gradients are calculated
with respect to the advantage function between the state's predicted value
and the earned rewards-to-go. The advantage will be positive if the actions
taken in the current episode earned a higher total reward than expected,
otherwise negative.
Therefore if there is a positive change in the networks policy,
its behavior will be reinforced and further solidified.
Though these advantages can be pretty volatile and hard to learn off of.
Thus in standard PPO this factor is modulated by a policy change ratio
that determines how different the current policy is to the old. This
limits the speed of policy changes. In PPO-Clip(implemented here) the
advantages are further stabilized by normalizing and setting bounds
to them, preventing any sudden jumps in network parameters.

On-policy methods(PPO) have pros over off-policy methods(Q-Learning).
They consider the decisions made as a whole instead of separate unrelated
steps. This is crucial in scenarios where it is necessary to take actions
that are less optimal in the short run but will lead to long term gains.
Model based approaches using neural networks also have the benefit that
they can handle continuous state and action spaces. On top of this the
neural models are more capable of understanding the environment they
interact with. This allows them to carry knowledge on how certain actions
will play out over time, giving them a bit of memory and saving much time
unnecessarily retrying the same patterns.

https://spinningup.openai.com/en/latest/algorithms/ppo.html
https://medium.com/swlh/coding-ppo-from-scratch-with-pytorch-part-2-4-f9d8b8aa938a
https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py
"""
from copy import deepcopy
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim

from nn import NN

np.random.seed(0)
torch.manual_seed(0)


class ActorCritic(nn.Module):
    def __init__(self, actor, critic, **kwargs):
        super().__init__()

        self.actor = actor
        self.critic = critic
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.epoch = 0
        self.lengths, self.eps = [], []

    def play(self, env):
        """
        Compute actor trajectories by running current policy.
        """
        n_steps, e_count = 0, 0
        self.history, self.values = [], torch.zeros(self.epoch_steps)
        while n_steps < self.epoch_steps:
            e_count += 1
            state = env.reset()
            for l in range(self.episode_len):
                action = self.actor.act(state)
                state_next, reward, done, info = env.step(action)

                self.history.append((state, action, reward, state_next))
                self.values[n_steps] = self.critic(state)
                state = state_next

                n_steps += 1
                if done or n_steps == self.epoch_steps:
                    break
            self.lengths.append(l+1)
            if np.mean(self.lengths[-self.n_to_win:]) >= self.win_score:
                print("Win!")
        self.eps.append(e_count)
        print(f"{self.epoch}: {np.mean(self.lengths[-e_count:]):.0f}")
        self.epoch += 1

    def rtg(self):
        """
        Calculate rewards to go,
        Rhat_t = sum_T(gamma^(i-t) R(s_i))
        """
        rtg = torch.zeros(self.epoch_steps)
        split_values = np.cumsum(self.lengths[-self.eps[-1]:][::-1])
        for i, (_, _, r,_) in enumerate(self.history[::-1]):
            if i == 0 or i in split_values:
                rhat = 0
            rhat = r + self.gamma * rhat
            rtg[-i-1] = rhat
        return rtg

    def advantages(self, rtg=None):
        """
        Calculate general advantage estimates,
        Ahat = Q^pi(s, a) - V^pi(s)
        """
        rtg = self.rtg() if rtg is None else rtg
        advantages = rtg - self.values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        return advantages

    def step(self):
        """
        Update policy and value networks.
        """
        rtg = self.rtg()
        advantages = self.advantages(rtg)

        self.actor.step(self.history, advantages)
        self.critic.step(self.history, self.values, rtg)


class Actor(nn.Module):
    def __init__(self, layers, activations, alpha, action_space, epsilon_clip, epoch_steps):
        super().__init__()
        self.action_space = action_space
        self.epsilon_clip = epsilon_clip
        self.epoch_steps = epoch_steps
        self.nn = NN(layers, activations)
        self.nn_old = deepcopy(self.nn)
        self.opt = optim.Adam(self.nn.parameters(), lr=alpha)
        self.policy_probs = torch.zeros((self.epoch_steps, len(self.action_space)))
        self.time = 0

    def forward(self, x):
        return self.nn(x)

    def act(self, state):
        policy_choice = self.nn(state) + 1e-10
        action = np.random.choice(self.action_space, p=policy_choice.detach().numpy() / float(policy_choice.sum()))
        self.policy_probs[self.time] = policy_choice
        self.time += 1
        return action

    def step(self, history, advantages):
        """
        Update policy via maximize ppo-clip objective
        https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl
        argmax 1 / NT sum_N sum_T min(pi_theta / pi_theta_k * A^pi_theta_k, g(epsilon, A^pi_theta_k))
        -> w += gradient. Invert loss output
        PPO Clip adds clipped surrogate objective as replacement for
        policy gradient objective. This improves stability by limiting
        change you make to policy at each step.
        Vanilla pg uses log p of action to trace impact of actions,
        ppo clip uses prob of action under current policy / prob under prev policy
        r_t(theta) = pi_theta(a_t, s_t) / pi_theta_old(a_t | s_t)
        Thus L = E[r * advantages]
        r > 1 if action more probable with current policy, else 1 > r > 0.
        to prevent too large steps from too large r, uses clipped surrogate objective
        ie L_clip = E[min(r * advantages, clip(r, 1 - epsilon, 1 + epsilon) * advantages)]
        epsilon = 2.
        """
        self.opt.zero_grad()

        p, p_old = torch.zeros(len(self.policy_probs)), torch.zeros(len(self.policy_probs))
        for i, (s, a, _, _) in enumerate(history):
            a_idx = self.action_space.index(a)
            p[i] = self.policy_probs[i][a_idx]
            p_old[i] = self.nn_old(s)[a_idx]

        r_theta = p / p_old.detach()
        r_clip = torch.clamp(r_theta, 1 - self.epsilon_clip, 1 + self.epsilon_clip)
        loss = -torch.min(r_theta * advantages, r_clip * advantages).mean()

        loss.backward(retain_graph=True)
        self.opt.step()

        self.nn_old = deepcopy(self.nn)
        self.policy_probs = torch.zeros((self.epoch_steps, len(self.action_space)))
        self.time = 0


class Critic(nn.Module):
    def __init__(self, layers, activations, alpha):
        super().__init__()

        self.nn = NN(layers, activations)
        self.criterion = nn.MSELoss()
        self.opt = optim.Adam(self.nn.parameters(), lr=alpha)

    def forward(self, x):
        return self.nn(x)

    def step(self, history, values, rtg):
        """
        Fit value fn via regression on MSE.
        """
        self.opt.zero_grad()
        loss = self.criterion(values, rtg)
        loss.backward()
        self.opt.step()


if __name__ == '__main__':
    """
    PPO Learning the cartpole balancing problem.

    Sucess Metric
    -------------
    Bellman resisual approaching 0.
    Longer cartpole trials, >=200.
    """
    N_EPOCH = 50
    EPOCH_STEPS = 4800
    ALPHA = .005

    env = gym.make('CartPole-v0')
    action_space = [i for i in range(env.action_space.n)]

    N_INPUT = 4
    LATENT_SIZE = 64
    N_ACTION = env.action_space.n

    actor = Actor(
        [N_INPUT, LATENT_SIZE, LATENT_SIZE, N_ACTION],
        [nn.Tanh(), nn.Tanh(), nn.Softmax(dim=-1)],
        alpha=ALPHA,
        action_space=action_space,
        epsilon_clip=.2,
        epoch_steps=EPOCH_STEPS,
    )
    critic = Critic(
        [N_INPUT, LATENT_SIZE, LATENT_SIZE, 1],
        [nn.Tanh(), nn.Tanh(), lambda x: x],
        ALPHA,
    )

    ac = ActorCritic(
        actor,
        critic,
        gamma=.95,
        epoch_steps=EPOCH_STEPS,
        episode_len=400,
        win_score=200,
        n_to_win=100,
        action_space=action_space,
    )

    for e in range(N_EPOCH):
        ac.play(env)
        ac.step()

    import matplotlib.pyplot as plt
    import seaborn as sns
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    eps_scatter = []
    for i, v in enumerate(ac.eps):
        eps_scatter.extend([i for _ in range(v)])
    sns.violinplot(eps_scatter, ac.lengths, ax=ax1)
    X = np.arange(len(ac.history))
    ax2.plot(X, ac.values.detach().numpy(), label="v")
    ax2.plot(X, ac.rtg(), label="rtg")
    ax2.plot(X, ac.advantages().detach().numpy(), label="A")
    ax2.legend()
    plt.show()
