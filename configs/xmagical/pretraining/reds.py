from base_configs.pretrain import get_config as _get_config

def get_config():
    """REDS config."""

    config = _get_config()

    config.algorithm = "reds_reward"
    config.optim.train_max_iters = 10000
    config.frame_sampler.strategy = "uniform"
    config.frame_sampler.uniform_sampler.offset = 0
    config.frame_sampler.num_frames_per_sequence = 50

    # Model settings
    config.model.model_type = "reds_model"  
    config.model.embedding_size = 32
    config.model.fusion = "concat"
    config.model.gpt2_layers = 2

    # Loss settings for REDS
    config.loss.reds.lambda_epic = 1.0
    config.loss.reds.lambda_supcon = 1.0
    config.loss.reds.supcon_temperature = 0.1
    config.loss.reds.epic_eps = 5e-2
    config.loss.reds.lambda_epic_reg = 1.0

    return config