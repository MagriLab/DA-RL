import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.experiment = None

    config.env_name = "KS"

    config.seed = 41

    config.total_steps = 50000
    config.episode_steps = 1000
    config.learning_starts = 5000
    config.eval_freq = 5000
    config.plot_freq = 5000
    config.eval_episodes = 5

    config.env = ml_collections.ConfigDict()

    config.esn = ml_collections.ConfigDict()

    config.network = ml_collections.ConfigDict()
    config.network.actor_hidden_units = [256, 256]
    config.network.critic_hidden_units = [256, 256]
    config.network.activation_function = "relu"

    config.train = ml_collections.ConfigDict()
    config.train.actor_learning_rate = 3e-4
    config.train.critic_learning_rate = 3e-4
    config.train.discount_factor = 0.99  # denoted by gamma
    config.train.soft_update_rate = 0.005  # denoted by tau
    config.train.batch_size = 256
    config.train.exploration_stddev = 0.1  # to be scaled by the environment

    config.replay_buffer = ml_collections.ConfigDict()
    config.replay_buffer.capacity = 100000

    config.enKF = ml_collections.ConfigDict()
    config.enKF.std_init = 0.2
    config.enKF.m = 50
    config.enKF.std_obs = 0.2
    config.enKF.low_order_N = 20
    config.enKF.observation_starts = 100
    config.enKF.wait_steps = 10
    config.enKF.use_reward = "model"
    config.enKF.inflation_factor = 1.05
    return config
