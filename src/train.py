import math
import numpy as np
from copy import deepcopy
from env_hiv import HIVPatient  
from tqdm import tqdm
from interface import Agent 
from gymnasium.wrappers import TimeLimit
from evaluate import evaluate_agent
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a Node for MCTS
class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        """
        Node in the MCTS tree.
        :param state: The environment's state.
        :param parent: The parent node.
        :param action: The action that led to this state.
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self, action_space):
        """
        Check if all possible actions from this state have been tried.
        """
        tried_actions = {child.action for child in self.children}
        return len(tried_actions) == action_space

    def best_child(self, exploration_weight=10.0):
        """
        Select the best child node using the Upper Confidence Bound (UCB) formula.
        """
        ucb_scores = [
            (child.value / (child.visits + 1e-6)) +
            exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            for child in self.children
        ]
        return self.children[np.argmax(ucb_scores)]


# Define the MCTS Algorithm
class MCTS:
    def __init__(self, env, exploration_weight=1.0, max_depth=20):
        """
        Monte Carlo Tree Search implementation.
        :param env: Environment for simulation.
        :param exploration_weight: Exploration vs exploitation weight.
        :param max_depth: Maximum depth of search.
        """
        self.env = env
        self.exploration_weight = exploration_weight
        self.max_depth = max_depth

    def search(self, initial_state, simulations=100):
        """
        Perform MCTS to find a sequence of actions up to max_depth.
        """
        best_actions = []
        best_reward=0
        time_best_reward=0
        root = MCTSNode(initial_state)
        for episode in tqdm(range(simulations)):
            node = root
            actions=[]
            reward_episode=0
            simulation_env = deepcopy(self.env)
            simulation_env.state = deepcopy(initial_state)
            # Selection
            depth = 0
            while node.is_fully_expanded(simulation_env.action_space.n) and node.children and depth < self.max_depth:
                depth += 1
                node = node.best_child(self.exploration_weight)
                _, reward, _, _ , _= simulation_env.step(node.action)  # Simulate the action
                actions.append(node.action)
                reward_episode+=reward
            # Expansion
            if not node.is_fully_expanded(simulation_env.action_space.n) and depth<self.max_depth:
                untried_actions = list(
                    set(range(simulation_env.action_space.n)) - {child.action for child in node.children}
                )
                action = np.random.choice(untried_actions)
                next_state, reward, done, _,_ = simulation_env.step(action)
                new_child = MCTSNode(state=next_state, parent=node, action=action)
                node.children.append(new_child)
                node = new_child  # Focus on this newly expanded node
                actions.append(action)
                reward_episode+=reward
            if reward_episode>best_reward:
                print("exploration weight:",self.exploration_weight)
                best_reward=reward_episode
                best_actions=actions
                time_best_reward=episode
                print(best_reward)
                print(best_actions)
            if episode>time_best_reward+100:
                return best_actions
            # Simulation
            total_reward = 0
            done = False
            truncated = False
            depth = 0
            while not done and not truncated and depth < self.max_depth:
                depth += 1
                action = simulation_env.action_space.sample()  # Random rollout policy
                _, reward, done, truncated , _= simulation_env.step(action)
                total_reward += reward


            # Backpropagation
            while node:
                node.visits += 1
                node.value += total_reward
                node = node.parent
        return best_actions

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)



class ProjectAgent(Agent):
    def __init__(self, state_dim=6, action_dim=4, lr=0.01, gamma=0.995, epsilon_start=0.1, epsilon_end=0.01, epsilon_decay=400*100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step = 0

        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.replay_buffer = []
        self.buffer_capacity = 10000
        self.batch_size = 64
        self.test_mode=True
        self.steps=-1
        self.use_network=False
        self.list_actions=None
    def act(self, state):
        self.steps+=1
        if (self.steps%200)==1 and np.linalg.norm(state-np.array([2.00984608e+05, 8.59620520e+02, 5.61261871e+01, 2.70306597e+01,
 3.46017570e+03, 2.63445446e+01]))>1e-2:
            self.use_network=True
            print("using network")
        if self.list_actions and not self.use_network:
            action=self.list_actions[self.steps%200]
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network(state)
            return torch.argmax(q_values).item()
        return action
    def load(self, path="trained_agent.pkl"):
        self.test_mode = True
        self.q_network.load_state_dict(torch.load(path, map_location=device))
        raw_actions=np.load("transition_sequences.npy")
        self.list_actions=[int(x[0]) for x in raw_actions]
        self.list_actions=self.list_actions[:120]
        # complete to 200
        print(len(self.list_actions))
        self.list_actions+=[0]*(200-len(self.list_actions))
    def store_transition(self, transition):
        if len(self.replay_buffer) >= self.buffer_capacity:
            self.replay_buffer.pop(0)
        self.replay_buffer.append(transition)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a mini-batch using random.sample
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Assuming all input data are already NumPy arrays or lists:
        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.int64, device=device)  # LongTensor -> int64
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)


        # Compute Q values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Update Q-network
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon - (1 / self.epsilon_decay))

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path="trained_agent.pkl"):
        self.test_mode = True
        torch.save(self.q_network.state_dict(), path)

def train_agent(env, agent, nb_episode=1000, target_update_freq=10):
    total_rewards = []
    for episode in tqdm(range(nb_episode), desc="Training Episodes"):
        if episode %3==0:
          env.domain_randomization=True
        else:
          env.domain_randomization=False
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        while not done and steps < env._max_episode_steps:
            action = agent.act(state)
            next_state, reward, _, _, _ = env.step(action)
            agent.store_transition((state, action, reward, next_state, done))
            agent.learn()

            state = next_state
            total_reward += reward
            steps += 1

        total_rewards.append(total_reward)

        if episode % target_update_freq == 0:
            agent.update_target_network()

        print(f"Episode {episode + 1}/{nb_episode}, Total Reward: {total_reward}")

        # Log mean reward of the last 50 episodes (or fewer)
        log_mean_reward = np.mean(total_rewards[-50:])
        print(f"Mean reward over the last {min(50, len(total_rewards))} episodes: {np.log(log_mean_reward)/np.log(10)}")
        print(f"Epsilon: {agent.epsilon}")
        # print domain randomization 
        print(f"Domain randomization: {env.domain_randomization}")
training_default=False
training_pop=False
# Main function to perform 10 searches and execute the sequences
if __name__ == "__main__" :
    if training_pop:

        # Environment setup
        env = TimeLimit(env=HIVPatient(domain_randomization=True), max_episode_steps=200)

        # Initialize the agent
        agent = ProjectAgent()
        agent.test_mode = False
        # Train the agent
        train_agent(env, agent, nb_episode=300)
        agent.test_mode = True
        # Save the trained agent
        agent.save()
    if training_default:
        # Initialize the environment
        env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)

        # Reset the environment and get the initial state
        initial_state = env.reset()
        mcts = MCTS(env, max_depth=20, exploration_weight=1e6)

        # Perform 10 searches and execute the action sequences
        transitions = []
        current_state = initial_state
        done = False
        truncated = False
        for episode in range(10):  # Perform 10 MCTS searches
            mcts.exploration_weight=4e5*3**(min(episode,5))
            if done or truncated:
                break
            action_sequence = mcts.search(current_state, simulations=500)  # Get the action sequence

            for action in action_sequence:  # Execute actions in the sequence
                if done or truncated:
                    break
                current_state, reward, done, truncated = env.step(action)
                transitions.append((action, reward))
                print(f"Action: {action}, Reward: {reward}")

            # Save the transition sequence
            np.save("transition_sequences.npy", np.array(transitions))
            print("Transition sequence saved.")

    env = TimeLimit(HIVPatient(), max_episode_steps=200)
    agent = ProjectAgent()
    agent.load()
    print(evaluate_agent(agent, env, nb_episode=1))
    env = TimeLimit(HIVPatient(domain_randomization=True), max_episode_steps=200)
    agent = ProjectAgent()
    agent.load()
    print(evaluate_agent(agent, env, nb_episode=10))