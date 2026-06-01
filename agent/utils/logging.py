import wandb 

class NoOpLogger:
    """A drop-in replacement for wandb that does nothing."""
    def init(self, **kwargs): pass
    def log(self, *args, **kwargs): pass
    def finish(self, **kwargs): pass

def setup_logger(use_wandb: bool, **wandb_kwargs):
    """Returns either real wandb or a no-op logger."""
    if use_wandb:
        wandb.init(**wandb_kwargs)
        return wandb
    return NoOpLogger()