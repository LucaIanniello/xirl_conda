from base_configs.pretrain import get_config as _get_config


def get_config():
  """TCC config."""

  config = _get_config()

  config.algorithm = "holdr"
  config.optim.train_max_iters = 2
  config.frame_sampler.strategy = "uniform"
  config.frame_sampler.uniform_sampler.offset = 0
  config.frame_sampler.num_frames_per_sequence = 40
  config.model.model_type = "resnet18_linear"
  config.model.embedding_size = 32
  config.model.normalize_embeddings = False
  config.model.learnable_temp = False
  
  config.loss.holdr.temperature = 0.1

  return config
