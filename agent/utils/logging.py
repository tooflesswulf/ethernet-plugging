import wandb 

class NoOpLogger:
    """A drop-in replacement for wandb that does nothing."""
    def init(self, **kwargs): pass
    def log(self, *args, **kwargs): pass
    def log_config(self, config): pass
    def finish(self, **kwargs): pass


def _log_config(config):
    """Write config to the wandb run's overview page."""
    wandb.config.update(config)

def setup_logger(use_wandb: bool, **wandb_kwargs):
    """Returns either real wandb or a no-op logger."""
    if use_wandb:
        wandb.init(**wandb_kwargs)
        # Expose a uniform log_config method matching NoOpLogger's interface.
        wandb.log_config = _log_config
        return wandb
    return NoOpLogger()
