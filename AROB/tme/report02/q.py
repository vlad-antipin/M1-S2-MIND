import os
import time
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import gymnasium as gym

from sklearn.metrics import auc
from scipy.stats import ttest_ind

from bbrl_gymnasium.envs.maze_mdp import MazeMDPEnv

from mazemdp.toolbox import egreedy, egreedy_loc, sample_categorical, softmax
from mazemdp import random_policy

import os
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
import matplotlib.pyplot as plt
import gymnasium as gym

from bbrl_gymnasium.envs.maze_mdp import MazeMDPEnv

from mazemdp.toolbox import egreedy, egreedy_loc, sample_categorical, softmax
from mazemdp import random_policy

from mazemdp.mdp import Mdp


def make_mdp(width=3, height=3,ratio=0):
    """
    Return the MazeMDP and an unwrapped version, used to access the attributes and function without raising a warning
    We also need the not unwrapped env for calling steps otherwise truncation beyond time limit will not apply
    """
    # Environment with 20% of walls and no negative reward when hitting a wall
    mdp = gym.make(
        "MazeMDP-v0",
        kwargs={"width": width, "height": height, "ratio": ratio, "hit": 0.0, "start_states": [0]},
        render_mode="human",
    )
    env = mdp.unwrapped  # the .unwrapped removes a warning from gymnasium when accessing an attribute or function
    return mdp, env


# ================ Q-learning ===================

def get_policy_from_q(q: np.ndarray) -> np.ndarray:
    return q.argmax(axis=1)

def q_learning_soft(
    mdp: MazeMDPEnv,
    alpha: float = 0.2,  # alpha: learning rate
    beta: float = 0.6,
    nb_episodes: int = 20,
    render: bool = True,
):

    # Run learning cycle
    env = mdp.unwrapped

    # Initialize the state-action value function
    # alpha is the learning rate
    q = np.zeros((env.nb_states, env.action_space.n))
    q_min = np.zeros((env.nb_states, env.action_space.n))
    q_list = [0]
    time_list = [0]
    sample_list = [0]
    start_time = time.process_time()
    nb_samples = 0

    if render:
        env.init_draw("Q Learning (Softmax)")

    for _ in range(nb_episodes):
        # Draw the first state of episode i using a uniform distribution over all the states
        s, _ = mdp.reset(options={"uniform": True})
        cpt = 0

        terminated = False
        truncated = False
        while not (terminated or truncated):
            if render:
                env.draw_v_pi(q, q.argmax(axis=1))

            # To be completed...

            # Draw an action using a soft-max policy
            a = sample_categorical(
                softmax(q, s, beta)
            )  # (here, call the softmax function)
            # assert False, "Not implemented yet"

            # To be completed...

            # Copy-paste the rest from q_learning_eps
            y, r, terminated, truncated, _ = mdp.step(a)
            
            delta = r + env.gamma * q[y, :].max() * (1 - terminated) - q[s, a]
            q[s, a] = q[s, a] + alpha * delta

            s = y
            cpt = cpt + 1
            nb_samples += 1

        q_list.append(np.linalg.norm(np.maximum(q, q_min)))
        # time_list.append(cpt)
        time_list.append(time.process_time() - start_time)
        sample_list.append(nb_samples)

    if render:
        env.current_state = 0
        env.draw_v_pi(q, get_policy_from_q(q))

    return np.array([q_list, time_list, sample_list])


# ============= DYNA-Q =================


def calculate_iqm(all_steps, min_bound, max_bound):
      """
      Calculate the Interquartile Mean (IQM) for the given data.
      :param all_steps: Array of steps from multiple runs
      :return: IQM array
      """
      # Calculate the min and max percentiles
      steps_min = np.percentile(all_steps, min_bound, axis=0)
      steps_max = np.percentile(all_steps, max_bound, axis=0)

      # IQM calculation: mean of values between min and max percentiles
      steps_iqm = np.mean(np.clip(all_steps, steps_min, steps_max), axis=0)

      return steps_iqm, steps_min, steps_max

def plot_quartiles(q_iqm, q_low, q_high, label):
    plt.plot(range(len(q_iqm)), q_iqm, label=label)
    plt.fill_between(range(len(q_iqm)),
                     q_low,
                     q_high,
                     alpha=0.2)
    

class TransitionModel():
   def __init__(self, nb_states, nb_actions):
      self.nb_states = nb_states
      self.nb_actions = nb_actions

   def predict_next_state(self, state, action) -> int:
      pass

   def add_transition(self, state, action, next_state) -> None:
      pass

   # To monitor the accuracy of the model of the transition function, we build an evaluate function.
   # This function draws sample_size random transitions from the real maze and checks whether the model outputs the same next state.
   # This way to proceed is only adequate if the real maze is deterministic.
   # Otherwise, we should tell the distance between two probability distributions.

   def evaluate_random(self, sample_size: int=100) -> int:
      success = 0
      for _ in range(sample_size):
         state, action, next_state = env.sample_transition()
         next_state_model = self.predict_next_state(state, action)
         if next_state == next_state_model:
            success = success + 1
      return success / 100

   def is_accurate(self) -> bool:
      pass

   def display(self) -> None:
      pass
   
class StochasticTransitionModel(TransitionModel):
   def __init__(self,env, nb_states, nb_actions, threshold=0.01):
      super().__init__(nb_states, nb_actions)
      self.probas = np.ones(
            (self.nb_states, self.nb_actions, self.nb_states)
         ) / self.nb_states
      self.I = np.array(range(self.nb_states))
      self.count = np.zeros((self.nb_states, self.nb_actions))
      self.threshold = threshold

   def predict_next_state(self, state, action) -> int:
      # To be completed...
      next_action_distr = self.probas[state, action, :]
      return sample_categorical(next_action_distr)


   def add_transition(self, state, action, next_state) -> None:
      self.count[state, action] = self.count[state, action] + 1
      self.probas[state, action, :] = (1-1/self.count[state, action]) * (self.probas[state, action, :]).reshape(self.nb_states) + 1/self.count[state, action] * np.transpose((self.I==next_state).astype(int))

   # In the probabilistic case, sampling the right next state is not enough, as it may happen by chance
   # we compare the probabilities

   def is_accurate(self) -> bool:
      """
      This function assumes access to the true probabilities of the mdp,
      it should not be used by an agent learning from interactions with
      the environment. To be used only for evaluation purposes
      """
      # To be completed...
      # compare probability distributions
      return np.linalg.norm(self.probas - env.P) < self.threshold

   
   # Useful for debug
   
   def display(self) -> None:
      print("probas :", self.probas)

class DeterministicTransitionModel(TransitionModel):
   def __init__(self,env, nb_states, nb_actions):
      super().__init__(nb_states, nb_actions)
      self.count = np.zeros(
            (self.nb_states, self.nb_actions, self.nb_states)
         )

   def predict_next_state(self, state, action) -> int:
      # To be completed...
      return int(self.count[state, action, :].argmax())


   def add_transition(self, state, action, next_state) -> None:
      self.count[state, action, next_state] = self.count[state, action, next_state] + 1

   # This function draws all transitions from the real maze and checks whether the model outputs the same next state
   # This way to proceed is only adequate if the real maze is deterministic.

   def is_accurate(self) -> bool:
      """
      This function assumes access to the true probabilities of the mdp,
      it should not be used by an agent learning from interactions with
      the environment. To be used only for evaluation purposes
      """
      # To be completed...
      return bool(np.all((self.count > 0) == env.P))


   # Useful for debug
   
   def display(self) -> None:
      print("count :", self.count)

class RewardModel():
   def __init__(self,env, nb_states, nb_actions):
      self.nb_states = nb_states
      self.nb_actions = nb_actions
      self.reward_model = np.zeros((self.nb_states, self.nb_actions))

   def predict_reward(self, state, action) -> float:
      return self.reward_model[state, action]

   def add_reward(self, state, action, reward) -> None:
      self.reward_model[state, action] = reward

   # This function draws all transitions from the real maze and checks whether the model outputs the same reward.
   # This way to proceed is only adequate if the real maze is deterministic.

   def is_accurate(self) -> bool:
      """
      This function assumes access to the true probabilities of the mdp,
      it should not be used by an agent learning from interactions with
      the environment. To be used only for evaluation purposes
      """
      # To be completed...
      # return bool(np.all(self.reward_model == env.r))
   
      for state in range(self.nb_states):
         for action in range(self.nb_actions):
            reward = env.r[state, action]
            if reward != self.predict_reward(state, action):
               return False
      return True

   # Useful for debug
   
   def display_reward(self) -> None:
      print("reward model", self.reward_model)

class TerminationModel():
   def __init__(self,env, nb_states, nb_actions):
      self.nb_states = nb_states
      self.nb_actions = nb_actions
      self.termination_model = np.zeros((self.nb_states, self.nb_actions))
   
   def predict_termination(self, state, action) -> bool:
      return self.termination_model[state, action]

   def add_termination(self, state, action, termination) -> None:
      self.termination_model[state, action] = termination


   # This function draws all transitions from the real maze and checks whether the model outputs the same termination
   # This way to proceed is only adequate if the real maze is deterministic.

   def is_accurate(self) -> bool:
      """
      This function assumes access to the true probabilities of the mdp,
      it should not be used by an agent learning from interactions with
      the environment. To be used only for evaluation purposes
      """
      # To be completed...
      # _, _, terminated, *_ = mdp.step(action)

      for state in range(self.nb_states):
         for action in range(self.nb_actions):
            terminated = env.P[state, action].argmax() in env.terminal_states
            if terminated != self.predict_termination(state, action):
               return False
      return True


   # Useful for debug

   def display_termination(self) -> None:
      print("terminated model", self.termination_model)

class FullModel():
   def __init__(self,env, deterministic=True, threshold=0.01):
      self.nb_states = env.nb_states 
      self.nb_actions = env.action_space.n
      if deterministic:
         self.transition_model = DeterministicTransitionModel(env,self.nb_states, self.nb_actions)
      else:
         self.transition_model = StochasticTransitionModel(env,self.nb_states, self.nb_actions, threshold=threshold)
      self.reward_model = RewardModel(env, self.nb_states, self.nb_actions)
      self.termination_model = TerminationModel(env,self.nb_states, self.nb_actions)

   def is_accurate(self) -> bool:
      """
      This function assumes access to the true probabilities of the mdp,
      it should not be used by an agent learning from interactions with
      the environment. To be used only for evaluation purposes
      """
      trans_ok = self.transition_model.is_accurate()
      rew_ok = self.reward_model.is_accurate()
      term_ok = self.termination_model.is_accurate()
      return trans_ok and rew_ok and term_ok

   def add_sample(self, state, action, reward, next_state, terminated) -> None:
      self.transition_model.add_transition(state, action, next_state)
      self.reward_model.add_reward(state, action, reward)
      self.termination_model.add_termination(state, action, terminated)

   def sample_state_action(self) -> Tuple[int, int]:
      """
      This function draws a random state, action pair from the model.
      """
      state = random.randint(0, self.nb_states - 1)
      action = random.randint(0, self.nb_actions - 1)
      return state, action

   def predict_full_sample(self, state, action)-> Tuple[int, int, float, int, bool]: 
      """
      This function draws a full sample from the models.
      in BBRL, this is renamed into forward
      """
      # To be completed...

      reward = self.reward_model.predict_reward(state, action)
      next_state = self.transition_model.predict_next_state(state, action)
      terminated = self.termination_model.predict_termination(state, action)

      return state, action, reward, next_state, terminated
   
   def display_all(self) -> None:
      self.transition_model.display()
      self.reward_model.display_reward()
      self.termination_model.display_termination()

class QAgent():
   def __init__(self, env, alpha):
      self.nb_states = env.nb_states
      self.nb_actions = env.action_space.n
      self.gamma = env.gamma
      self.alpha = alpha
      self.init_Q()

   def choose_action(self, state):
      pass

   def init_Q(self):
       self.Q = np.zeros((self.nb_states, self.nb_actions))
       
   # Beware that the efficiency is highly dependent on the error threshold
   def is_accurate(self, q_func, threshold=0.01) -> bool:
      """
      This function assumes access to the ground truth optimal $Q$ function
      it should not be used by an agent learning from interactions with
      the environment. To be used only for evaluation purposes
      """
      error = np.linalg.norm(self.Q - q_func)
      if error > threshold:
         return False
      return True

   # Do not forget to deal with the case where the episode is terminated
   def updateQ(self, state, action, reward, next_state, terminated) -> None:
      """
      Performs a Bellman back-up over the $Q$ function of the agent
      :return: nothing
      """
      # To be completed...
      delta = reward + (1-terminated) * self.gamma * self.Q[next_state, :].max() - self.Q[state, action]
      self.Q[state, action] += self.alpha * delta


   # The function below should use the above function
   def update_Q_from_model(self, full_model:FullModel, nb_updates: int) -> None:
      """
      Updates the $Q$ function of the agent using randomly sampled transitions (i.e. the agent is learning in imagination)
      It does so nb_updates times
      :param full_model: the model used to sample updates
      :param nb_updates: the number of performed updates
      :return: nothing
      """
      # To be completed...
      for _ in range(nb_updates):
         state, action = full_model.sample_state_action()
         state, action, reward, next_state, terminated = full_model.predict_full_sample(state, action)
         self.updateQ(state, action, reward, next_state, terminated)


   # Learn the Q function in imagination from a model of the MDP
   def learn_Q_from_model(self, full_model: FullModel, nb_steps: int, nb_repeats: int) -> Tuple[int, np.array]: 
      # To be completed...
      q_list = []
      for _ in range(nb_steps):
         self.update_Q_from_model(full_model, nb_repeats) 
         q_list.append(np.linalg.norm(self.Q))
      return q_list
   
class SoftmaxQAgent(QAgent):
   def __init__(self,env, alpha: float = 0.5, beta: float = 6.0):
      super().__init__(env,alpha)
      self.beta = beta

   def choose_action(self, state) -> int:
      # To be completed...
      action = sample_categorical(softmax(self.Q, state ,self.beta)) 
      return action

class EgreedyQAgent(QAgent):
   def __init__(self,env, alpha: float = 0.5, epsilon: float = 0.02):
      super().__init__(env,alpha)
      self.epsilon = epsilon 

   def choose_action(self, state) -> int:
      # To be completed...
      action = egreedy(self.Q, state,self.epsilon)
      return action
   
def dyna_q_soft(
    mdp: MazeMDPEnv,
    alpha,
    beta,
    nb_steps: int,
    nb_updates: int = 5,
    uniform: bool = True,
):
    env = mdp.unwrapped

    agent = SoftmaxQAgent(env, alpha=alpha, beta=beta)
   # Run learning cycle
    full_model = FullModel(env)

    state, _ = mdp.reset(options={"uniform": uniform})
    truncated = terminated = False

    q_list = [0]
    time_list = [0]
    sample_list = [0]
    start_time = time.process_time()
    nb_samples = 0

    steps = 0
    while steps < nb_steps: 
        # To be completed...
            
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = mdp.step(action)
        full_model.add_sample(state, action, reward, next_state, terminated)
        

        agent.update_Q_from_model(full_model, nb_updates)
        if (truncated or terminated):
            state, _ = mdp.reset(options={"uniform": uniform}) 
        else:
            state = next_state
        steps += 1
        nb_samples += 1
        q_list.append(np.linalg.norm(agent.Q)) 
        time_list.append(time.process_time() - start_time)
        sample_list.append(nb_samples)

    return np.array([q_list, time_list, sample_list])


# ===== DP TO CHECK Q VALUE ======

def get_policy_from_q(q: np.ndarray) -> np.ndarray:
    # Outputs a policy given the action values
    return q.argmax(axis=1)

def evaluate_one_step_q(
    mdp: MazeMDPEnv, q: np.ndarray, policy: np.ndarray
) -> np.ndarray:
    # Outputs the state value function after one step of policy evaluation
    qnew = np.zeros(
        (mdp.nb_states, mdp.action_space.n)
    )  # initial action values are set to 0
    for s in range(mdp.nb_states):  # for each state x
        # Compute the value of the state x for each action u of the MDP action space
        for a in range(mdp.action_space.n):
            if s in mdp.terminal_states:
                qnew[s, a] = mdp.r[s, a]
            else:
                # Process sum of the values of the neighbouring states
                summ = 0
                for y in range(mdp.nb_states):
                    # [[STUDENT]]...
                    summ += mdp.P[s,a, y] * q[y, policy[y]]

                # [[STUDENT]]...
                qnew[s, a] = mdp.r[s,a] + mdp.gamma * summ

    return qnew
def evaluate_q(mdp: MazeMDPEnv, policy: np.ndarray) -> np.ndarray:
    # Outputs the state value function of a policy
    q = np.zeros(
        (mdp.nb_states, mdp.action_space.n)
    )  # initial action values are set to 0
    stop = False
    while not stop:
        qold = q.copy()
        # [[STUDENT]]...
        q = evaluate_one_step_q(mdp, q, policy)


        # Test if convergence has been reached
        if (np.linalg.norm(q - qold)) < 0.01:
            stop = True
    return q
def policy_iteration_q(
    mdp: MazeMDPEnv, render: bool = True
) -> Tuple[np.ndarray, List[float]]:
    """policy iteration over the q function."""
    q = np.zeros(
        (mdp.nb_states, mdp.action_space.n)
    )  # initial action values are set to 0
    q_list = []
    policy = random_policy(mdp)

    stop = False

    if render:
        mdp.init_draw("Policy iteration Q")

    while not stop:
        qold = q.copy()
        if render:
            mdp.draw_v(q)

        # Step 1 : Policy evaluation

        q = evaluate_q(mdp, policy)

        # Step 2 : Policy improvement

        policy = get_policy_from_q(q)


        # Check convergence
        if (np.linalg.norm(q - qold)) <= 0.01:
            stop = True
        q_list.append(np.linalg.norm(q))

    if render:
        mdp.draw_v_pi(q, get_policy_from_q(q))
    return q, q_list