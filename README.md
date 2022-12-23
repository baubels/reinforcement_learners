# RL Agents Library
A mini RL library implementing RL learners on OpenAI environments.

The current environments supported are OpenAI's [Classic Control](https://www.gymlibrary.dev/environments/classic_control/) : `CartPole`, `Acrobot`, and `MountainCar` environments.

On my computer only Cartpole is relatively quick, the others take more time to run.

Learners supported: `DQN`, `DDQN`, `REINFORCE`, `ACTOR_CRITIC`. Examples can be found in `usage_example.ipynb`. Trying different nets is a matter of using a different single-line function.

This is run successfully on Python 3.10.8, with libraries used found in the `requirements.txt` file.

#### Install

```
python3 -m venv rl_testing
source rl_testing/bin/activate
git clone https://github.com/baubels/reinforcement_learners.git
cd reinforcement_learners
pip install -r requirements.txt
```

#### Sample Usage

```python
import general_utils, nets, nets_utils, accessory
run, trained_net = nets.train_DQN('DDQN', env_name='CartPoleEnv', n_runs=1, starting_eps=1., network_layers=[4,32,2],
                                  episode_print_thresh=1, n_episodes=600, buffer_size=100000, batch_size=64,
                                  update_when=1, learning_rate=0.01, decay=0.99, max_episode_steps=500, record=False, lengths_to_consider=1)
accessory.print_results(run, n_episodes=600)                                      # print results of the sample runs
```