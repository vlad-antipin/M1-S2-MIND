import os
import time
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import gymnasium as gym

import optuna

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




def tune_params(objective, search_space):
   n_trials = 1
   for item in search_space.items():
      n_trials *= len(item[1])
   study_grid = optuna.create_study(direction='maximize', sampler=optuna.samplers.GridSampler(search_space))
   study_grid.optimize(objective, n_trials=n_trials, show_progress_bar=True)

   study_grid_analyse = study_grid.trials_dataframe(attrs=('params', 'value'))
   best_grid = study_grid.best_params
   print ('The best parameters founded using Grid search are: ', best_grid, '\n\n')
   return study_grid_analyse

def plot_optuna(study_grid_analyse, x, y):
   plt.figure(figsize=(8, 4))
   plt.scatter(study_grid_analyse["params_"+x], 
               study_grid_analyse["params_"+y],
               c=study_grid_analyse['value'], cmap="RdYlGn_r")
   plt.title('Value function norms (Grid search)')
   plt.xlabel(x)
   plt.ylabel(y)
   plt.grid(False)
   plt.colorbar()
   # plt.savefig("grid_search.pdf")
   plt.show()


def plot_and_compare_auc(alpha,
                         beta,
                        max_episodes,
                        nb_steps,
                        nb_updates,
                        nb_repeats,
                        mdp_sizes,
                        alpha_plot = 0.7, 
                        colors = ("#888888","#1f77b4", "#9467bd") ):

   aucs_ql_all = {}
   for mdp_size in mdp_sizes:
      mdp, env = make_mdp(*mdp_size)
      q, q_list = policy_iteration_q(env.unwrapped, render=False)
      q_ref = q_list[-1]

      log_ql = np.empty((nb_repeats, 3, max_episodes+1))
      log_dyna = np.empty((nb_repeats, 3, nb_steps+1))

      for i in tqdm(range(nb_repeats)):
         log_ql[i] = q_learning_soft(mdp, alpha, beta, nb_episodes=max_episodes, render=False)
         log_dyna[i] = dyna_q_soft(mdp,alpha,beta, nb_steps=nb_steps, nb_updates=nb_updates)

      for i, measure_type in enumerate(("times", "samples")):
         measures_ql=log_ql[:,i+1,:]
         perfs_ql=log_ql[:,0,:]
         measures_dyna=log_dyna[:,i+1,:]
         perfs_dyna=log_dyna[:,0,:]
         perf_optim=q_ref
         fig, axes = plt.subplots(1,2, figsize=(12,5))
         axes = axes.flatten()
         axes[0].set_title("Q-value evolution during training")
         axes[0].set_ylabel("Q norm")

         if measure_type == "samples":
            axes[0].set_xlabel("nb of used samples")
         else:
            axes[0].set_xlabel("training time (s)")

         
         axes[0].axhline(y=perf_optim, linestyle="--", linewidth=2, c=colors[0],alpha=alpha_plot)
         
         nb_repeats = len(perfs_ql)

         min_measure = np.maximum(measures_ql[:, 0].max(), measures_dyna[:, 0].max())
         max_measure = np.minimum(measures_ql[:, -1].min(), measures_dyna[:, -1].min())
         grid = np.linspace(min_measure, max_measure, 100)
         aucs_ql = np.empty(nb_repeats)
         aucs_dyna = np.empty(nb_repeats)

         axes[0].set_xlim(min_measure, max_measure)

         for i in range(nb_repeats):

            interp_ql = np.interp(grid, measures_ql[i], perfs_ql[i])
            interp_dyna = np.interp(grid, measures_dyna[i], perfs_dyna[i])
            aucs_ql[i] = auc(grid, interp_ql)
            aucs_dyna[i] = auc(grid, interp_dyna)
            

            axes[0].plot(measures_ql[i], perfs_ql[i], c=colors[1], alpha=alpha_plot)
            axes[0].plot(measures_dyna[i], perfs_dyna[i], c=colors[2], alpha=alpha_plot)
         
         
         
         legend_elements = [
            Line2D([0], [0], color=colors[0], lw=2, alpha=alpha_plot, linestyle='--', label='optimal Q'),
            Line2D([0], [0], color=colors[1], lw=2, alpha=alpha_plot, label='Q-learning'),
            Line2D([0], [0], color=colors[2], lw=2, alpha=alpha_plot, label='DYNA-Q')
         ]
         axes[0].legend(handles=legend_elements)

         test_res = ttest_ind(aucs_ql, aucs_dyna, equal_var=False)
         pval = test_res.pvalue

         axes[1].set_ylabel("AUC")
         axes[1].set_title("AUC comparison across runs")
         positions = [1, 2]

         # Boxplots with black medians
         axes[1].boxplot(
            [aucs_ql, aucs_dyna],
            positions=positions,
            medianprops=dict(color='black')
         )

         # Scatter individual points
         axes[1].scatter(np.ones(len(aucs_ql))*1, aucs_ql, alpha=alpha_plot, c=colors[1])
         axes[1].scatter(np.ones(len(aucs_dyna))*2, aucs_dyna, alpha=alpha_plot, c=colors[2])
         axes[1].set_xticks([1, 2], ["Q-Learning", "Dyna-Q"])

         # Add p-value annotation
         y_max = max(max(aucs_ql), max(aucs_dyna))  # top of the data
         y_min = min(min(aucs_ql), min(aucs_dyna))
         y, h, col = y_max + 0.02*(y_max-y_min), 0.01*(y_max-y_min), 'k'  # position above boxes

         axes[1].plot([positions[0], positions[0], positions[1], positions[1]], [y, y+h, y+h, y], lw=1.5, c=col)
         axes[1].text((positions[0]+positions[1])*0.5, y+h, f"p = {pval:.3g}", ha='center', va='bottom', color=col)

         if measure_type == "samples":
            measure_name = "Sample"
         else:
            measure_name = "Time"
         fig.suptitle(f"{measure_name} efficiency for maze of size {mdp_size[0]} by {mdp_size[1]}")
         plt.tight_layout()
         plt.show()
         

# ALTERNATIVE:

def train_models(alpha,
                  beta,
               max_episodes,
               nb_steps,
               nb_updates,
               nb_repeats,
               mdp_sizes):
      log_ql_all = {}
      log_dyna_all = {}
      q_ref_all = {}
      for mdp_size in mdp_sizes:
         mdp, env = make_mdp(*mdp_size)
         q, q_list = policy_iteration_q(env.unwrapped, render=False)
         q_ref_all[mdp_size] = q_list[-1]

         log_ql_all[mdp_size] = np.empty((nb_repeats, 3, max_episodes+1))
         log_dyna_all[mdp_size] = np.empty((nb_repeats, 3, nb_steps+1))

         for i in tqdm(range(nb_repeats)):
            log_ql_all[mdp_size][i] = q_learning_soft(mdp, alpha, beta, nb_episodes=max_episodes, render=False)
            log_dyna_all[mdp_size][i] = dyna_q_soft(mdp,alpha,beta, nb_steps=nb_steps, nb_updates=nb_updates)
      return log_ql_all, log_dyna_all, q_ref_all

def pval_to_stars(pval):
    if pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval < 0.05:
        return "*"
    else:
        return ""
    
def plot_and_compare_auc2(log_ql_all, 
                          log_dyna_all, 
                          q_ref_all,
                        nb_repeats,
                        mdp_sizes,
                        alpha_plot = 0.7, 
                        colors = ("#888888","#1f77b4", "#9467bd") ):

   aucs_ql_all = {}
   aucs_dyna_all = {}
   # log_ql_all = {}
   # log_dyna_all = {}
   # q_ref_all = {}
   # for mdp_size in mdp_sizes:
   #    mdp, env = make_mdp(*mdp_size)
   #    q, q_list = policy_iteration_q(env.unwrapped, render=False)
   #    q_ref_all[mdp_size] = q_list[-1]

   #    log_ql_all[mdp_size] = np.empty((nb_repeats, 3, max_episodes+1))
   #    log_dyna_all[mdp_size] = np.empty((nb_repeats, 3, nb_steps+1))

   #    for i in tqdm(range(nb_repeats)):
   #       log_ql_all[mdp_size][i] = q_learning_soft(mdp, alpha, beta, nb_episodes=max_episodes, render=False)
   #       log_dyna_all[mdp_size][i] = dyna_q_soft(mdp,alpha,beta, nb_steps=nb_steps, nb_updates=nb_updates)

   for mdp_size in mdp_sizes:
      log_ql = log_ql_all[mdp_size]
      log_dyna = log_dyna_all[mdp_size]
      q_ref = q_ref_all[mdp_size]
      fig, axes = plt.subplots(1,2, figsize=(12,5))
      axes = axes.flatten()

      for i, measure_type in enumerate(("times", "samples")):
         
         min_measure = - np.inf
         max_measure = np.inf
         for ms in mdp_sizes:
            measures_ql=log_ql_all[ms][:,i+1,:]
            measures_dyna=log_dyna_all[ms][:,i+1,:]
            min_measure = np.maximum(min_measure, np.maximum(measures_ql[:, 0].max(), measures_dyna[:, 0].max()))
            max_measure = np.minimum(max_measure, np.minimum(measures_ql[:, -1].min(), measures_dyna[:, -1].min()))

         measures_ql=log_ql[:,i+1,:]
         perfs_ql=log_ql[:,0,:]
         measures_dyna=log_dyna[:,i+1,:]
         perfs_dyna=log_dyna[:,0,:]
         perf_optim=q_ref

         if measure_type == "samples":
            axes[i].set_title("Sample efficiency")
            axes[i].set_xlabel("nb of used samples")
         else:
            axes[i].set_title("Time efficiency")
            axes[i].set_xlabel("training time (s)")
         
         axes[i].set_ylabel(r"$||Q||_2$")
        
         axes[i].axhline(y=perf_optim, linestyle="--", linewidth=2, c=colors[0],alpha=0.7)
         
         nb_repeats = len(perfs_ql)

         grid = np.linspace(min_measure, max_measure, 100)
         aucs_ql = np.empty(nb_repeats)
         aucs_dyna = np.empty(nb_repeats)

         axes[i].set_xlim(min_measure, max_measure)

         for j in range(nb_repeats):

            interp_ql = np.interp(grid, measures_ql[j], perfs_ql[j] / q_ref_all[mdp_size])
            interp_dyna = np.interp(grid, measures_dyna[j], perfs_dyna[j]  / q_ref_all[mdp_size])
            aucs_ql[j] = auc(grid, interp_ql)
            aucs_dyna[j] = auc(grid, interp_dyna)
            
            # Normalize height (so that curves all converge to the same value (1))
            # and AUC are comparable
            axes[i].plot(measures_ql[j], perfs_ql[j] , c=colors[1], alpha=alpha_plot)
            axes[i].plot(measures_dyna[j], perfs_dyna[j], c=colors[2], alpha=alpha_plot)
         
         grid_smooth = np.linspace(0, max_measure, 100)
         smoothed_ql = np.empty(len(grid_smooth)-1)
         smoothed_dyna = np.empty(len(grid_smooth)-1)
         for bin in range(len(grid_smooth)-1):
            edge_left, edge_right = grid_smooth[bin], grid_smooth[bin+1]
            mask_ql = (measures_ql > edge_left) & (measures_ql < edge_right) 
            mask_dyna = (measures_dyna > edge_left) & (measures_dyna < edge_right) 
            smoothed_ql[bin] = perfs_ql[mask_ql].mean()
            smoothed_dyna[bin] = perfs_dyna[mask_dyna].mean()
         grid_smooth = grid_smooth[:-1]
         
         window_size = mdp_size[0]*2+1
         pad_width = window_size // 2
         smoothed_ql_padded = np.pad(smoothed_ql, pad_width, mode='edge')
         smoothed_dyna_padded = np.pad(smoothed_dyna, pad_width, mode='edge')
         smoothed_ql = np.convolve(smoothed_ql_padded, np.ones(window_size)/window_size, mode='valid')
         smoothed_dyna = np.convolve(smoothed_dyna_padded, np.ones(window_size)/window_size, mode='valid')

         axes[i].plot(grid_smooth, smoothed_ql , c=colors[1], alpha=0.7, linewidth=3)
         axes[i].plot(grid_smooth, smoothed_dyna, c=colors[2], alpha=0.7, linewidth=3)


         aucs_ql_all[(mdp_size, measure_type)] = aucs_ql
         aucs_dyna_all[(mdp_size, measure_type)] = aucs_dyna
         
         legend_elements = [
            Line2D([0], [0], color=colors[0], lw=2,  linestyle='--', label='optimal Q', alpha=0.7),
            Line2D([0], [0], color=colors[1], lw=2,  label='Q-learning', alpha=0.7),
            Line2D([0], [0], color=colors[2], lw=2, label='DYNA-Q', alpha=0.7)
         ]
         axes[i].legend(handles=legend_elements)

         if measure_type == "samples":
            measure_name = "Sample"
         else:
            measure_name = "Time"
      # fig.suptitle(f"Maze of size {mdp_size[0]} by {mdp_size[1]}")
      plt.tight_layout()
      plt.savefig(f"curves-maze_{mdp_size[0]}.pdf")
      plt.show()
   alpha_plot = 0.4
   fig, axes = plt.subplots(1,2, figsize=(12,5))
   axes = axes.flatten()
   for j, measure_type in enumerate(("times", "samples")):
      for i, mdp_size in enumerate(mdp_sizes):
      
         aucs_ql = aucs_ql_all[(mdp_size, measure_type)]
         aucs_dyna = aucs_dyna_all[(mdp_size, measure_type)]

         test_res = ttest_ind(aucs_ql, aucs_dyna, equal_var=False)
         pval = test_res.pvalue

         if measure_type == "times":
            axes[j].set_ylabel(f"AUC")
            axes[j].set_title("Time efficiency comparison")
         else:
            axes[j].set_ylabel(f"AUC")
            axes[j].set_title("Sample efficiency comparison")
         positions = [2*i+1, 2*i+2]

         # Boxplots with black medians
         axes[j].boxplot(
            [aucs_ql, aucs_dyna],
            positions=positions,
            medianprops=dict(color='black')
         )

         # Scatter individual points
         axes[j].scatter(np.ones(len(aucs_ql))*positions[0], aucs_ql, alpha=alpha_plot, c=colors[1])
         axes[j].scatter(np.ones(len(aucs_dyna))*positions[1], aucs_dyna, alpha=alpha_plot, c=colors[2])
         # axes[j].xticks([positions[0], positions[1]], ["Q-Learning", "Dyna-Q"])
         

         # Add p-value annotation
         y_max = max(max(aucs_ql), max(aucs_dyna))  # top of the data
         y_min = min(min(aucs_ql), min(aucs_dyna))
         y, h, col = y_max + 0.02*(y_max-y_min), 0.01*(y_max-y_min), 'k'  # position above boxes

         axes[j].plot([positions[0], positions[0], positions[1], positions[1]], [y, y+h, y+h, y], lw=1.5, c=col)
         axes[j].text((positions[0]+positions[1])*0.5, y+h,pval_to_stars(pval), #f"p = {pval:.3g}",
                       ha='center', va='bottom', color=col)
      axes[j].set_xticks([1.5, 3.5, 5.5], mdp_sizes)
      axes[j].set_xlabel("maze size")

      aucs_ql0 = aucs_ql_all[(mdp_sizes[0], measure_type)]
      aucs_ql1 = aucs_ql_all[(mdp_sizes[1], measure_type)]
      aucs_ql2 = aucs_ql_all[(mdp_sizes[2], measure_type)]

      aucs_dyna0 = aucs_dyna_all[(mdp_sizes[0], measure_type)]
      aucs_dyna1 = aucs_dyna_all[(mdp_sizes[1], measure_type)]
      auc_dyna2 = aucs_dyna_all[(mdp_sizes[2], measure_type)]

      # 0 vs 1
      test_res = ttest_ind(aucs_ql0, aucs_ql1, equal_var=False)
      pval_ql01 = test_res.pvalue
      test_res = ttest_ind(aucs_dyna0, aucs_dyna1, equal_var=False)
      pval_dyna01 = test_res.pvalue

      # 1 vs 2
      test_res = ttest_ind(aucs_ql1, aucs_ql2, equal_var=False)
      pval_ql12 = test_res.pvalue
      test_res = ttest_ind(aucs_dyna1, auc_dyna2, equal_var=False)
      pval_dyna12 = test_res.pvalue

      # 0 vs 2
      test_res = ttest_ind(aucs_ql0, aucs_ql2, equal_var=False)
      pval_ql02 = test_res.pvalue
      test_res = ttest_ind(aucs_dyna0, auc_dyna2, equal_var=False)
      pval_dyna02 = test_res.pvalue

      print(f"{pval_ql01=}")
      print(f"{pval_ql12=}")
      print(f"{pval_ql02=}")
      print(f"{pval_dyna01=}")
      print(f"{pval_dyna12=}")
      print(f"{pval_dyna02=}")

      legend_elements = [
      Line2D([0], [0],
            marker='o',
            linestyle='None',
            color=colors[1],
            alpha=alpha_plot,
            label='Q-learning'),
      Line2D([0], [0],
            marker='o',
            linestyle='None',
            color=colors[2],
            alpha=alpha_plot,
            label='DYNA-Q')
   ]
      axes[j].legend(handles=legend_elements)
   plt.tight_layout()
   plt.savefig(f"comparison-all.pdf")
   plt.show()
         