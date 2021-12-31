"""
Deep Q Network
Approximates a q table as neural network so it'll generalize to continuous state and action spaces.
Should generalize better as well.
- off-policy
- model based
- continuous
"""
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
        self.history, self.values = [], torch.zeros(200)#self.epoch_steps)
        while n_steps < self.epoch_steps:
            e_count += 1
            state = env.reset()
            for l in range(200):
                action = self.actor.act(state, self.critic)
                state_next, reward, done, info = env.step(action)

                self.history.append((state, action, reward, state_next, done))
                self.values[n_steps] = self.critic(state, action)
                state = state_next

                n_steps += 1
                if done:
                    break
            self.lengths.append(l+1)
            if np.mean(self.lengths[-self.n_to_win:]) >= self.win_score:
                print("Win!")    

        self.values = self.values[:l+1]
        self.eps.append(e_count)
        print(f"{self.epoch}: {np.mean(self.lengths[-e_count:]):.0f}")
        self.epoch += 1

    def targets(self):
        with torch.no_grad():
            targets = torch.zeros(self.lengths[-1])
            for i, (_, _, r,s_next, d) in enumerate(self.history):
                targets[i] = r + self.gamma * (1 - d) * self.critic.forward(s_next, self.actor.forward(s_next, self.critic))

        return targets

    def step(self):
        """
        Update policy and value networks.
        """
        y = self.targets()
        self.actor.step(self.history, None)
        self.critic.step(self.history, self.values, y)


class Actor(nn.Module):
    def __init__(self, layers, activations, alpha, action_space, epsilon_clip, epoch_steps):
        super().__init__()
        self.action_space = action_space
        self.epsilon_clip = epsilon_clip
        self.epoch_steps = epoch_steps
        self.time = 0

    def forward(self, state, critic):
        return int(torch.Tensor([critic(state, a) for a in self.action_space]).argmax())

    def act(self, state, critic):
        #print([critic(state, a) for a in self.action_space])
        action_probs = torch.Tensor([critic(state, a) for a in self.action_space])
        action = action_probs.argmax() if np.random.uniform() > .10 else torch.randint(action_probs.size()[0], (1,))
        return int(action)

    def step(self, history, advantages):
        return


class Critic(nn.Module):
    def __init__(self, layers, activations, alpha):
        super().__init__()

        self.nn = NN(layers, activations)
        self.criterion = nn.MSELoss()
        self.opt = optim.Adam(self.nn.parameters(), lr=alpha)

    def forward(self, x, a):
        return self.nn(torch.cat([torch.Tensor(x), torch.Tensor([a])], axis=0))

    def step(self, history, values, y):
        """
        Fit value fn via regression on MSE.
        """
        self.opt.zero_grad()
        loss = self.criterion(values, y)
        loss.backward()
        self.opt.step()


if __name__ == '__main__':
    """
    Sucess Metric
    -------------
    Critic loss goes down.
    Longer cartpole trials, >=200.
    """
    N_EPOCH = 1000
    EPOCH_STEPS = 1
    ALPHA = .001

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
        [N_INPUT + 1, LATENT_SIZE, LATENT_SIZE, 1],
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
    ax1 = fig.add_subplot(111)
    eps_scatter = []
    for i, v in enumerate(ac.eps):
        eps_scatter.extend([i for _ in range(v)])
    sns.violinplot(eps_scatter, ac.lengths, ax=ax1)
    #plt.show()
