import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    # seed for reproducibility
    config.seed = 41

    # model configuration
    config.model = ml_collections.ConfigDict()

    config.model.which_state = "true_state"  # true_state, true_obs, obs
    config.model.which_control = "action"  # action, forcing

    config.model.network_dt = 5e-2
    config.model.washout_time = 5

    config.model.reservoir_size = 1000
    config.model.connectivity = 3
    config.model.r2_mode = False
    config.model.input_weights_mode = "random_sparse"
    config.model.reservoir_weights_mode = "erdos_renyi1"
    config.model.normalize_input = True
    config.model.input_bias = True

    # training configuration
    config.train_episodes = 40
    config.val_episodes = 5
    config.test_episodes = 5
    config.episode_type = "random_action"
    config.tikhonov = 1e-7

    # validation configuration
    config.validate = False

    config.val = ml_collections.ConfigDict()

    config.val.fold_time = 25
    config.val.n_folds = 3
    config.val.n_realisations = 3
    config.val.n_calls = 10
    config.val.n_initial_points = 5
    config.val.error_measure = "rel_L2"

    # Ranges for the hyperparameters

    # WARNING: when passing a scale other than uniform,
    # the min and max should be the original min and max you want
    # the scaling is done in the script using the scalers in the utils

    config.val.hyperparameters = ml_collections.ConfigDict()

    # SPECTRAL RADIUS
    config.val.hyperparameters.spectral_radius = ml_collections.ConfigDict()
    config.val.hyperparameters.spectral_radius.min = 0.01
    config.val.hyperparameters.spectral_radius.max = 1.0
    config.val.hyperparameters.spectral_radius.scale = "log10"

    # INPUT SCALING
    config.val.hyperparameters.input_scaling = ml_collections.ConfigDict()
    config.val.hyperparameters.input_scaling.min = 0.01
    config.val.hyperparameters.input_scaling.max = 5.0
    config.val.hyperparameters.input_scaling.scale = "log10"

    # LEAK FACTOR
    config.val.hyperparameters.leak_factor = ml_collections.ConfigDict()
    config.val.hyperparameters.leak_factor.min = 0.01
    config.val.hyperparameters.leak_factor.max = 1.0
    config.val.hyperparameters.leak_factor.scale = "log10"

    # TIKHONOV
    config.val.hyperparameters.tikhonov = ml_collections.ConfigDict()
    config.val.hyperparameters.tikhonov.min = 1e-8
    config.val.hyperparameters.tikhonov.max = 1e-1
    config.val.hyperparameters.tikhonov.scale = "log10"

    # PARAMETER NORMALIZATION
    # config.val.hyperparameters.parameter_normalization_mean = (
    #     ml_collections.ConfigDict()
    # )
    # config.val.hyperparameters.parameter_normalization_mean.min = -10.0
    # config.val.hyperparameters.parameter_normalization_mean.max = 10.0
    # config.val.hyperparameters.parameter_normalization_mean.scale = "uniform"

    config.val.hyperparameters.parameter_normalization_var = ml_collections.ConfigDict()
    config.val.hyperparameters.parameter_normalization_var.min = 0.01
    config.val.hyperparameters.parameter_normalization_var.max = 10.0
    config.val.hyperparameters.parameter_normalization_var.scale = "log10"

    return config
