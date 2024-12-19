import numpy as np
import contextlib
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally: 
        np.set_printoptions(**original)


class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        
        self.random_state = np.random.RandomState(seed)
        
    def p(self, next_state, state, action):
        raise NotImplementedError()
    
    def r(self, next_state, state, action):
        raise NotImplementedError()
        
    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)
        
        return next_state, reward

        
class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)
        
        self.max_steps = max_steps
        
        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1./n_states)
        
    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)
        
        return self.state
        
    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')
        
        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)
        
        self.state, reward = self.draw(self.state, action)
        
        return self.state, reward, done
    
    def render(self, policy=None, value=None):
        raise NotImplementedError()

        
class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
        lake =  [['&', '.', '.', '.'],
                ['.', '#', '.', '#'],
                ['.', '.', '.', '#'],
                ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)
        
        self.slip = slip
        
        n_states = self.lake.size + 1
        n_actions = 4
        
        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0
        
        self.absorbing_state = n_states - 1
        
        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed=seed)
        
    def step(self, action):
        state, reward, done = Environment.step(self, action)
        
        done = (state == self.absorbing_state) or done
        
        return state, reward, done
        
    def p(self, next_state, state, action):
        if state == self.absorbing_state:
            return 1.0 if next_state == self.absorbing_state else 0.0
        
        if self.lake_flat[state] in ['#', '$']:
            return 1.0 if next_state == self.absorbing_state else 0.0
        
        if self.random_state.rand() < self.slip:
            return 1.0 / self.n_states
        
        row, col = divmod(state, self.lake.shape[1])
        if action == 0:  # up
            row = max(row - 1, 0)
        elif action == 1:  # left
            col = max(col - 1, 0)
        elif action == 2:  # down
            row = min(row + 1, self.lake.shape[0] - 1)
        elif action == 3:  # right
            col = min(col + 1, self.lake.shape[1] - 1)
        
        next_state_expected = row * self.lake.shape[1] + col
        return 1.0 if next_state == next_state_expected else 0.0
    
    def r(self, next_state, state, action):
        if state == self.absorbing_state:
            return 0.0
        if self.lake_flat[state] == '$':
            return 1.0
        return 0.0
    
    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)
            
            if self.state < self.absorbing_state:
                lake[self.state] = '@'
                
            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['^', '<', '_', '>']
            
            print('Lake:')
            print(self.lake)
        
            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))
            
            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))
                
def play(env):
    actions = ['w', 'a', 's', 'd']
    
    state = env.reset()
    env.render()
    
    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid action')
            
        state, r, done = env.step(actions.index(c))
        
        env.render()
        print('Reward: {0}.'.format(r))
        
def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)
    
    for _ in range(max_iterations):
        delta = 0
        for state in range(env.n_states):
            v = value[state]
            action = policy[state]
            value[state] = sum([env.p(next_state, state, action) * 
                                (env.r(next_state, state, action) + gamma * value[next_state])
                                for next_state in range(env.n_states)])
            delta = max(delta, abs(v - value[state]))
        if delta < theta:
            break
    
    return value

def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)
    
    for state in range(env.n_states):
        q_values = np.zeros(env.n_actions)
        for action in range(env.n_actions):
            q_values[action] = sum([env.p(next_state, state, action) * 
                                    (env.r(next_state, state, action) + gamma * value[next_state])
                                    for next_state in range(env.n_states)])
        policy[state] = np.argmax(q_values)
    
    return policy

def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    
    for _ in range(max_iterations):
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        new_policy = policy_improvement(env, value, gamma)
        if np.array_equal(policy, new_policy):
            break
        policy = new_policy
    
    value = policy_evaluation(env, policy, gamma, theta, max_iterations)
    return policy, value

def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)
    
    for _ in range(max_iterations):
        delta = 0
        for state in range(env.n_states):
            v = value[state]
            q_values = np.zeros(env.n_actions)
            for action in range(env.n_actions):
                q_values[action] = sum([env.p(next_state, state, action) * 
                                        (env.r(next_state, state, action) + gamma * value[next_state])
                                        for next_state in range(env.n_states)])
            value[state] = np.max(q_values)
            delta = max(delta, abs(v - value[state]))
        if delta < theta:
            break
    
    policy = policy_improvement(env, value, gamma)
    return policy, value

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    q = np.zeros((env.n_states, env.n_actions))
    
    for i in range(max_episodes):
        s = env.reset()
        if random_state.rand() < epsilon[i]:
            a = random_state.choice(env.n_actions)
        else:
            a = random_state.choice(np.flatnonzero(q[s] == q[s].max()))
        
        done = False
        while not done:
            s_next, r, done = env.step(a)
            if random_state.rand() < epsilon[i]:
                a_next = random_state.choice(env.n_actions)
            else:
                a_next = random_state.choice(np.flatnonzero(q[s_next] == q[s_next].max()))
            
            q[s, a] += eta[i] * (r + gamma * q[s_next, a_next] - q[s, a])
            s, a = s_next, a_next
    
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
    return policy, value

def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    q = np.zeros((env.n_states, env.n_actions))
    
    for i in range(max_episodes):
        s = env.reset()
        done = False
        while not done:
            if random_state.rand() < epsilon[i]:
                a = random_state.choice(env.n_actions)
            else:
                a = random_state.choice(np.flatnonzero(q[s] == q[s].max()))
            
            s_next, r, done = env.step(a)
            q[s, a] += eta[i] * (r + gamma * q[s_next].max() - q[s, a])
            s = s_next
    
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
    return policy, value

class LinearWrapper:
    def __init__(self, env):
        self.env = env
        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states
        
    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0
        return features
    
    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)
        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)
            policy[s] = np.argmax(q)
            value[s] = np.max(q)
        return policy, value
        
    def reset(self):
        return self.encode_state(self.env.reset())
    
    def step(self, action):
        state, reward, done = self.env.step(action)
        return self.encode_state(state), reward, done
    
    def render(self, policy=None, value=None):
        self.env.render(policy, value)

def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        q = features.dot(theta)
        if random_state.rand() < epsilon[i]:
            a = random_state.choice(env.n_actions)
        else:
            a = random_state.choice(np.flatnonzero(q == q.max()))
        
        done = False
        while not done:
            next_features, reward, done = env.step(a)
            next_q = next_features.dot(theta)
            if random_state.rand() < epsilon[i]:
                next_a = random_state.choice(env.n_actions)
            else:
                next_a = random_state.choice(np.flatnonzero(next_q == next_q.max()))
            
            theta += eta[i] * (reward + gamma * next_q[next_a] - q[a]) * features[a]
            features, q, a = next_features, next_q, next_a
    
    return theta

def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        done = False
        while not done:
            q = features.dot(theta)
            if random_state.rand() < epsilon[i]:
                a = random_state.choice(env.n_actions)
            else:
                a = random_state.choice(np.flatnonzero(q == q.max()))
            
            next_features, reward, done = env.step(a)
            next_q = next_features.dot(theta)
            delta = reward - q[a]
            delta += gamma * next_q.max() if not done else 0
            theta += eta[i] * delta * features[a]
            features = next_features
    
    return theta

class FrozenLakeImageWrapper:
    def __init__(self, env):
        self.env = env
        lake = self.env.lake
        self.n_actions = self.env.n_actions
        self.state_shape = (4, lake.shape[0], lake.shape[1])
        
        lake_image = [(lake == c).astype(float) for c in ['&', '#', '$']]
        self.state_image = {self.env.absorbing_state: np.stack([np.zeros(lake.shape)] + lake_image)}
        
        for state in range(lake.size):
            agent_pos = np.zeros(lake.shape)
            agent_pos[np.unravel_index(state, lake.shape)] = 1.0
            self.state_image[state] = np.stack([agent_pos] + lake_image)
    
    def encode_state(self, state):
        return self.state_image[state]
    
    def decode_policy(self, dqn):
        states = np.array([self.encode_state(s) for s in range(self.env.n_states)])
        q = dqn(states).detach().numpy()
        
        policy = q.argmax(axis=1)
        value = q.max(axis=1)
        
        return policy, value
    
    def reset(self):
        return self.encode_state(self.env.reset())
    
    def step(self, action):
        state, reward, done = self.env.step(action)
        return self.encode_state(state), reward, done
    
    def render(self, policy=None, value=None):
        self.env.render(policy, value)

class DeepQNetwork(nn.Module):
    def __init__(self, env, learning_rate, kernel_size, conv_out_channels, fc_out_features, seed):
        super(DeepQNetwork, self).__init__()
        torch.manual_seed(seed)
        
        self.conv_layer = nn.Conv2d(in_channels=env.state_shape[0], out_channels=conv_out_channels, kernel_size=kernel_size, stride=1)
        h = env.state_shape[1] - kernel_size + 1
        w = env.state_shape[2] - kernel_size + 1
        self.fc_layer = nn.Linear(in_features=h * w * conv_out_channels, out_features=fc_out_features)
        self.output_layer = nn.Linear(in_features=fc_out_features, out_features=env.n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float)
        x = self.conv_layer(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        x = torch.relu(x)
        x = self.output_layer(x)
        return x
    
    def train_step(self, transitions, gamma, tdqn):
        states = np.array([transition[0] for transition in transitions])
        actions = np.array([transition[1] for transition in transitions])
        rewards = np.array([transition[2] for transition in transitions])
        next_states = np.array([transition[3] for transition in transitions])
        dones = np.array([transition[4] for transition in transitions])
        
        q = self(states)
        q = q.gather(1, torch.tensor(actions).view(len(transitions), 1).long())
        q = q.view(len(transitions))
        
        with torch.no_grad():
            next_q = tdqn(next_states).max(dim=1)[0] * (1 - dones)
        
        target = torch.tensor(rewards) + gamma * next_q
        
        loss = nn.MSELoss()(q, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class ReplayBuffer:
    def __init__(self, buffer_size, random_state):
        self.buffer = deque(maxlen=buffer_size)
        self.random_state = random_state
    
    def __len__(self):
        return len(self.buffer)
    
    def append(self, transition):
        self.buffer.append(transition)
    
    def draw(self, batch_size):
        indices = self.random_state.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

def deep_q_network_learning(env, max_episodes, learning_rate, gamma, epsilon, batch_size, target_update_frequency, buffer_size, kernel_size, conv_out_channels, fc_out_features, seed):
    random_state = np.random.RandomState(seed)
    replay_buffer = ReplayBuffer(buffer_size, random_state)
    
    dqn = DeepQNetwork(env, learning_rate, kernel_size, conv_out_channels, fc_out_features, seed=seed)
    tdqn = DeepQNetwork(env, learning_rate, kernel_size, conv_out_channels, fc_out_features, seed=seed)
    
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    for i in range(max_episodes):
        state = env.reset()
        done = False
        while not done:
            if random_state.rand() < epsilon[i]:
                action = random_state.choice(env.n_actions)
            else:
                with torch.no_grad():
                    q = dqn(np.array([state]))[0].numpy()
                qmax = max(q)
                best = [a for a in range(env.n_actions) if np.allclose(qmax, q[a])]
                action = random_state.choice(best)
            
            next_state, reward, done = env.step(action)
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            
            if len(replay_buffer) >= batch_size:
                transitions = replay_buffer.draw(batch_size)
                dqn.train_step(transitions, gamma, tdqn)
        
        if (i % target_update_frequency) == 0:
            tdqn.load_state_dict(dqn.state_dict())
    
    return dqn

def main():
    seed = 0
    
    # Big lake
    # lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
    #         ['.', '.', '.', '.', '.', '.', '.', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '.'],
    #         ['.', '.', '.', '.', '.', '#', '.', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '.'],
    #         ['.', '#', '#', '.', '.', '.', '#', '.'],
    #         ['.', '#', '.', '.', '#', '.', '#', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '$']]
    
    # Small lake
    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    gamma = 0.9
    
    print('# Model-based algorithms')

    print('')

    print('## Policy iteration')
    policy, value = policy_iteration(env, gamma, theta=0.001, max_iterations=128)
    env.render(policy, value)

    print('')

    print('## Value iteration')
    policy, value = value_iteration(env, gamma, theta=0.001, max_iterations=128)
    env.render(policy, value)

    print('')

    print('# Model-free algorithms')
    max_episodes = 4000

    print('')

    print('## Sarsa')
    policy, value = sarsa(env, max_episodes, eta=0.5, gamma=gamma,
                          epsilon=0.5, seed=seed)
    env.render(policy, value)

    print('')

    print('## Q-learning')
    policy, value = q_learning(env, max_episodes, eta=0.5, gamma=gamma,
                               epsilon=0.5, seed=seed)
    env.render(policy, value)

    print('')

    linear_env = LinearWrapper(env)

    print('## Linear Sarsa')

    parameters = linear_sarsa(linear_env, max_episodes, eta=0.5, gamma=gamma, 
                              epsilon=0.5, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print('')

    print('## Linear Q-learning')

    parameters = linear_q_learning(linear_env, max_episodes, eta=0.5, gamma=gamma, 
                                   epsilon=0.5, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print('')

    image_env = FrozenLakeImageWrapper(env)

    print('## Deep Q-network learning')

    dqn = deep_q_network_learning(image_env, max_episodes, learning_rate=0.001, 
                                  gamma=gamma,  epsilon=0.2, batch_size=32,
                                  target_update_frequency=4, buffer_size=256,
                                  kernel_size=3, conv_out_channels=4,
                                  fc_out_features=8, seed=4)
    policy, value = image_env.decode_policy(dqn)
    image_env.render(policy, value)
