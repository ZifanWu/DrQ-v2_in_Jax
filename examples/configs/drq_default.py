import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.algo = 'drq'

    config.n_step_trgt = 1

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.cnn_features = (32, 32, 32, 32)
    config.cnn_strides = (2, 1, 1, 1)
    config.cnn_padding = 'VALID'
    config.latent_dim = 50
    config.batch_size = 512 # NOTE 256 in DrQ-v2

    config.batch_size_statistics = 256
    config.dead_neurons_threshold = 0.025
    config.dormancy_logging_period = 2000

    config.discount = 0.99

    config.tau = 0.005
    config.target_update_period = 1

    config.init_temperature = 0.1
    config.target_entropy = None

    config.replay_buffer_size = 100_000

    config.gray_scale = False
    config.image_size = 84

    return config