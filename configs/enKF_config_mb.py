import ml_collections

# LT = 500, nu = 0.08
# LT = 350, nu = 0.05
# LT = 250, nu = 0.03
# LT = 250, nu = 0.02
# LT = 235, nu = 0.01

LT = 250
def get_config():
    config = ml_collections.ConfigDict()

    config.experiment = None

    config.env_name = "KS"

    config.seed = 41

    config.total_steps = (1000 + LT) * 100
    config.episode_steps = 1000 + LT
    config.learning_starts = (1000 + LT) * 5
    config.eval_freq = (1000 + LT) * 5
    config.plot_freq = (1000 + LT) * 5
    config.eval_episodes = 5

    config.env = ml_collections.ConfigDict()

    config.esn = ml_collections.ConfigDict()

    config.network = ml_collections.ConfigDict()
    config.network.actor_hidden_units = [256,256]
    config.network.critic_hidden_units = [256,256]
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
    config.enKF.std_init = 0.1
    config.enKF.m = 50
    config.enKF.std_obs = 0.1
    config.enKF.cov_type = "max" # "const", "max", "prop"
    config.enKF.low_order_N = 0
    config.enKF.observation_starts = LT
    config.enKF.wait_steps = 10
    config.enKF.use_reward = "model"
    config.enKF.inflation_factor = 1.02
    return config
