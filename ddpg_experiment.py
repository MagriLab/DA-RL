import os
import sys
from pathlib import Path

import orbax.checkpoint
from absl import app, flags
from ml_collections import config_flags

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

from datetime import datetime

import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import orbax
from flax.training import orbax_utils
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit

import utils.file_processing as fp
import utils.flags as myflags
import wandb
from ddpg import DDPG
from envs.KS_environment import KSenv
from replay_buffer import ReplayBuffer

# system config
FLAGS = flags.FLAGS
myflags.DEFINE_path("experiment_path", None, "Directory to store experiment results.")
flags.DEFINE_bool("log_wandb", False, "Use --log_wandb to log the experiment to wandb.")
flags.DEFINE_bool(
    "log_offline", False, "Use --log_offline to log the experiment to local."
)
flags.DEFINE_bool(
    "save_checkpoints",
    False,
    "Use --save_checkpoints to save intermediate model weights.",
)
_CONFIG = config_flags.DEFINE_config_file(
    "config", "configs/config.py", "Contains configs to run the experiment"
)
_WANDB = config_flags.DEFINE_config_file(
    "wandb_config", "configs/wandb_config.py", "Contains configs to log to wandb."
)
_ENV = config_flags.DEFINE_config_file(
    "env_config", "configs/KS_config.py", "Contains configs for the environment."
)

# flags.mark_flags_as_required(['config'])


def save_config():
    with open(FLAGS.experiment_path / "config.yml", "w") as f:
        FLAGS.config.to_yaml(stream=f)


def log_metrics_wandb(wandb_run, metrics, step):
    wandb_run.log(data=metrics, step=step)


def log_metrics_offline(logs, metrics):
    for metric_name, metric_value in metrics.items():
        if metric_name in logs.keys():
            logs[metric_name].append(metric_value)
        else:
            logs[metric_name] = [metric_value]


def add_gaussian_noise(key, x, stddev):
    noise = stddev * jax.random.normal(key, shape=x.shape)
    return x + noise


def evaluate(config, env, actor, params):
    # if env in jax, vmap over eval_episodes and jax.lax.scan over episode_steps
    last_reward_arr = []
    for _ in range(config.eval_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            # get action from the target actor network
            action = actor.apply(params, state)
            state, reward, terminated, truncated, _ = env.step(action)
        last_reward_arr.append(reward)

    return_queue = np.asarray(env.return_queue).flatten()[-config.eval_episodes :]
    length_queue = np.asarray(env.length_queue).flatten()[-config.eval_episodes :]
    average_return = return_queue.mean()
    average_reward = (return_queue / length_queue).mean()
    average_last_reward = np.asarray(last_reward_arr).mean()
    return average_return, average_reward, average_last_reward


def evaluate_KS_for_plotting(env, actor, params, eval_episodes=2):
    # this should go in a different file
    x = np.arange(env.unwrapped.N) * 2 * np.pi / env.unwrapped.N

    fig, axs = plt.subplots(
        eval_episodes,
        3,
        width_ratios=[1, 0.55, 1],
        figsize=(10, 3 * eval_episodes),
        constrained_layout=True,
    )

    # fig2, axs2 = plt.subplots(
    #     env.unwrapped.num_observations,
    #     eval_episodes,
    #     width_ratios=[1, 1],
    #     figsize=(6 * eval_episodes, 10),
    #     constrained_layout=True,
    # )

    for i in range(eval_episodes):
        state, _ = env.reset()
        full_state = env.unwrapped.u

        state_arr = np.array(state)
        full_state_arr = np.array(full_state)
        reward_arr = np.zeros((1,))

        terminated = False
        truncated = False
        while not terminated and not truncated:
            # get action from the target actor network
            action = actor.apply(params, state)
            state, reward, terminated, truncated, info = env.step(action)
            full_state = env.unwrapped.u
            state_arr = np.vstack((state_arr, state))
            full_state_arr = np.vstack((full_state_arr, full_state))
            reward_arr = np.vstack((reward_arr, reward))

        # plot the full state and reward
        im = axs[i, 0].imshow(
            full_state_arr.T,
            extent=[0, len(full_state_arr), x[0], x[-1]],
            origin="lower",
            aspect="auto",
        )
        axs[i, 0].set_xlabel("t")
        axs[i, 0].set_ylabel("x")
        cbar = fig.colorbar(im, ax=[axs[i, 0]], location="left")
        cbar.ax.set_title("u")
        axs[i, 0].set_title(
            f"Return={info['episode']['r'][0]:.2f}, Ave. Reward={info['episode']['r'][0]/info['episode']['l'][0]:.2f}"
        )

        axs[i, 1].plot(env.unwrapped.target, x)
        axs[i, 1].plot(full_state_arr[-1, :], x, "--")
        axs[i, 1].set_yticks(axs[i, 0].get_yticks())
        axs[i, 1].set_yticklabels([])
        axs[i, 1].set_ylim(axs[i, 0].get_ylim())
        axs[i, 1].set_title(f"Last Reward={reward:.2f}")
        axs[i, 1].grid()
        axs[i, 1].set_xlabel("u")
        axs[i, 1].legend(["Target", "Last"])

        err = np.abs(env.unwrapped.target[:, None] - full_state_arr.T)
        im = axs[i, 2].imshow(
            err,
            extent=[0, len(full_state_arr), x[0], x[-1]],
            origin="lower",
            aspect="auto",
            cmap="Reds",
        )
        axs[i, 2].set_xlabel("t")
        axs[i, 2].set_yticks(axs[i, 0].get_yticks())
        axs[i, 2].set_yticklabels([])
        axs[i, 2].set_ylim(axs[i, 0].get_ylim())
        cbar = fig.colorbar(im, ax=[axs[i, 2]], location="right")
        axs[i, 2].set_title("|Target - u|")

        # plot the measurements
        # for j in range(env.unwrapped.num_observations):
        #     axs2[j, i].plot(full_state_arr[:,env.unwrapped.observation_inds[j]])
        #     axs2[j, i].plot(state_arr[:,j], 'o--')
        #     if j < env.unwrapped.num_observations-1:
        #         axs2[j,i].set_xticklabels([])
        #     else:
        #         axs2[j,i].set_xlabel('t')
        #     if i == 0:
        #         axs2[j,i].set_ylabel(f'x={x[env.unwrapped.observation_inds[j]]:0.3f}')
    return fig


def run_experiment(
    config, env, eval_env, agent, wandb_run=None, logs=None, checkpoint_dir=None
):
    # random seed for initialization
    key = jax.random.PRNGKey(config.seed)

    # initialize networks
    # sample state and action to get the correct shape
    state_0 = jnp.array([env.observation_space.sample()])
    action_0 = jnp.array([env.action_space.sample()])
    actor_state, critic_state = agent.initial_network_state(key, state_0, action_0)

    # checkpoint the initial weights
    if checkpoint_dir is not None:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(
            save_interval_steps=config.total_steps // 5
        )
        checkpoint_manager = orbax.checkpoint.CheckpointManager(
            checkpoint_dir, orbax_checkpointer, options
        )
        checkpoint = {"actor": actor_state, "critic": critic_state}
        save_args = orbax_utils.save_args_from_target(checkpoint)
        checkpoint_manager.save(0, checkpoint, save_kwargs={"save_args": save_args})

    # initialize buffer
    replay_buffer = ReplayBuffer(config.replay_buffer.capacity)

    # initialize environment
    state, info = env.reset()
    replay_buffer.push(state=state, action=None, reward=None, terminated=None)

    # standard deviation of the exploration scales with the range of actions in the environment
    exploration_stddev = (
        env.action_space.high - env.action_space.low
    ) * config.train.exploration_stddev

    global_step = 0
    n_episode = 0
    while global_step < config.total_steps:
        key, _ = jax.random.split(key)
        if global_step < config.learning_starts:
            # collect samples before starting to train
            action = env.action_space.sample()
        else:
            # get action from the learning actor network
            action = agent.actor.apply(actor_state.params, state)

            # add exploration noise on the action
            # original paper adds Ornstein-Uhlenbeck process noise
            # but in other papers this is deemed unnecessary
            action = add_gaussian_noise(key, action, stddev=exploration_stddev)
            # clip the action so that it obeys the limits set by the environment
            action = jnp.clip(
                action, min=env.action_space.low, max=env.action_space.high
            )

        # get the next state and reward with this action
        state, reward, terminated, truncated, info = env.step(action)
        replay_buffer.push(state, action, reward, terminated)

        global_step += 1
        metrics = {"train": {}, "eval": {}}
        if "episode" in info:  # RecordEpisodeStatistics wrapper from gymnasium
            n_episode += 1
            print(
                f'Episode {n_episode}, Step={global_step}/{config.total_steps}, Return={info["episode"]["r"][0]}',
                flush=True,
            )
            # episode statistics to log
            metrics["train"]["episode_return"] = info["episode"]["r"][0]
            metrics["train"]["episode_length"] = info["episode"]["l"][0]
            metrics["train"]["episode_average_reward"] = (
                info["episode"]["r"][0] / info["episode"]["l"][0]
            )
            metrics["train"]["episode_last_reward"] = reward

        if truncated or terminated:
            # reset the environment
            state, info = env.reset()
            replay_buffer.push(state=state, action=None, reward=None, terminated=None)

        # if there is enough data in the buffer optimize
        if global_step >= config.learning_starts and replay_buffer.is_ready(
            config.train.batch_size
        ):
            (
                sampled_state,
                sampled_action,
                sampled_next_state,
                sampled_reward,
                sampled_terminated,
            ) = replay_buffer.sample(config.train.batch_size)
            critic_state, q_loss = agent.update_critic(
                actor_state,
                critic_state,
                sampled_state,
                sampled_action,
                sampled_next_state,
                sampled_reward,
                sampled_terminated,
            )

            actor_state, policy_loss = agent.update_actor(
                actor_state, critic_state, sampled_state
            )

            actor_state, critic_state = agent.update_target_networks(
                actor_state, critic_state
            )

            # losses to log
            metrics["train"]["q_loss"] = q_loss
            metrics["train"]["policy_loss"] = policy_loss

            if global_step % config.eval_freq == 0:
                # evaluate the performance of the target parameters ?
                eval_ave_return, eval_ave_reward, eval_ave_last_reward = evaluate(
                    config, eval_env, agent.actor, actor_state.target_params
                )
                print(
                    f"\n Evaluation, Step={global_step}/{config.total_steps}, Average Return={eval_ave_return} \n",
                    flush=True,
                )
                metrics["eval"]["average_return"] = eval_ave_return
                metrics["eval"]["average_reward"] = eval_ave_reward
                metrics["eval"]["average_last_reward"] = eval_ave_last_reward

        if wandb_run is not None:
            log_metrics_wandb(wandb_run, metrics, step=global_step)

        if logs is not None:
            log_metrics_offline(logs, metrics)

        # checkpoint the model
        if checkpoint_dir is not None:
            checkpoint = {"actor": actor_state, "critic": critic_state}
            save_args = orbax_utils.save_args_from_target(checkpoint)
            checkpoint_manager.save(
                global_step, checkpoint, save_kwargs={"save_args": save_args}
            )

    return actor_state, critic_state, logs


def main(_):
    config = FLAGS.config

    if not FLAGS.experiment_path:
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        FLAGS.experiment_path = (
            Path.cwd() / "local_results" / config.env_name / f"run_{dt_string}"
        )

    # setup the experiment path
    FLAGS.experiment_path.mkdir(parents=True, exist_ok=True)

    # redirect print to text file
    orig_stdout = sys.stdout
    f = open(FLAGS.experiment_path / "out.txt", "w")
    sys.stdout = f
    print(f"Experiment will be saved to {FLAGS.experiment_path}", flush=True)

    # add environment config to the config
    config.env = FLAGS.env_config

    save_config()

    # initialize wandb logging
    if FLAGS.log_wandb:
        wandb_run = wandb.init(config=config.to_dict(), **FLAGS.wandb_config)
    else:
        wandb_run = None

    # initialize offline logging
    if FLAGS.log_offline:
        logs = {}
    else:
        logs = None

    # create model directory if checkpoint saving
    if FLAGS.save_checkpoints:
        checkpoint_dir = FLAGS.experiment_path / "models"
    else:
        checkpoint_dir = None

    # create environment
    if config.env_name == "KS":
        env = KSenv(**config.env)
        env = TimeLimit(env, max_episode_steps=config.episode_steps)
        env = RecordEpisodeStatistics(env)

        eval_env = KSenv(**config.env)
        eval_env = TimeLimit(eval_env, max_episode_steps=config.episode_steps)
        eval_env = RecordEpisodeStatistics(eval_env)
    else:
        env = gym.make(config.env_name)
        env = RecordEpisodeStatistics(env)

        eval_env = gym.make(config.env_name)
        eval_env = RecordEpisodeStatistics(eval_env)

    # create agent and run
    agent = DDPG(config, env)
    print("Starting experiment.", flush=True)
    actor_state, critic_state, logs = run_experiment(
        config, env, eval_env, agent, wandb_run, logs, checkpoint_dir
    )

    # save the final model weights
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    final_checkpoint = {"actor": actor_state, "critic": critic_state}
    save_args = orbax_utils.save_args_from_target(final_checkpoint)
    orbax_checkpointer.save(
        FLAGS.experiment_path / "final_model", final_checkpoint, save_args=save_args
    )

    # finish logging
    if FLAGS.log_wandb:
        wandb_run.finish()

    if FLAGS.log_offline:
        print(f"Saving logs to {FLAGS.experiment_path}.")
        fp.pickle_file(FLAGS.experiment_path / "logs.pickle", logs)

    # evaluate with the final weights and plot episode
    if config.env_name == "KS":
        fig = evaluate_KS_for_plotting(eval_env, agent.actor, actor_state.target_params)
        fig.savefig(FLAGS.experiment_path / "final_evaluation.png")
        plt.show()

    # close text
    sys.stdout = orig_stdout
    f.close()
    return


if __name__ == "__main__":
    app.run(main)
