
import torch
import torch.nn as nn
from tqdm import tqdm
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from agent.model.networks import  ConditionalUnet1D, get_resnet, replace_bn_with_gn

def build_diffusion_policy(
    # train parameters
    num_training_steps,
    lr=1e-4,
    weight_decay=1e-6,
    num_warmup_steps=500,
    num_diffusion_iters=100,
    # model parameters
    obs_horizon=1,
    vision_feature_dim = 512,
    state_dim = 7,
    action_dim = 7,
    device = None
):

    # construct ResNet18 encoder
    # if you have multiple camera views, use seperate encoder weights for each view.
    vision_encoder = get_resnet('resnet18') # output dim of 512 for each

    # IMPORTANT!
    # replace all BatchNorm with GroupNorm to work with EMA
    # performance will tank if you forget to do this!
    vision_encoder = replace_bn_with_gn(vision_encoder)
    obs_dim = vision_feature_dim + state_dim

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )

    # the final arch has 2 parts
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })

    if device is not None: 
        nets.to(device)
    
    ema = EMAModel(parameters=nets.parameters(), power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW( params=nets.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    return nets, ema, optimizer, lr_scheduler, noise_scheduler

if __name__ == '__main__':
    import os
    from agent.pretrain.train import batch_to_device
    from agent.dataset.sequence import StitchedSequenceDataset
    device = 'cuda'
    task = 'ethernet_unplug'
    dataset_dir = '/zfsauton/scratch/yiqiw2/100%/datasets'
    dataset_path = os.path.join(dataset_dir, task)
    dataset = StitchedSequenceDataset(dataset_path, device = device)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True,
        # accelerate cpu-gpu transfer
        # pin_memory=True,
        # don't kill worker process afte each epoch
        # persistent_workers=True
    )
    nets, ema, opt, lr_scheduler, noise_scheduler = build_diffusion_policy( len(dataloader), device=device )
    for batch in tqdm( dataloader ):
        batch = batch_to_device(batch, device)
        images, states, actions, B = batch.conditions['rgb'], batch.conditions['state'], batch.actions, len(batch.actions)
        # BxTxCxHxW -> (B T)xCxHxW -> (B T) x d -> BxTxd
        image_features = nets['vision_encoder'](images.flatten(end_dim=1)).reshape(*images.shape[:2],-1)
        obs_features = torch.cat([image_features, states], dim=-1)
        obs_cond = obs_features.flatten(start_dim=1) 
        noise = torch.randn(actions.shape, device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()

        # add noise to the clean images according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)

        # predict the noise residual
        noise_pred = nets['noise_pred_net']( noisy_actions, timesteps, global_cond=obs_cond)

        # L2 loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        # optimize
        loss.backward()
        opt.step()
        opt.zero_grad()
        # step lr scheduler every batch
        # this is different from standard pytorch behavior
        lr_scheduler.step()

        # update Exponential Moving Average of the model weights
        ema.step(nets.parameters())

        # logging
        loss_cpu = loss.item()
    assert False,f"Maybe put all images into shared ram is a bad idea ..., regardless of numpy/torch or not?"
    # https://discuss.pytorch.org/t/cuda-initialization-error-when-dataloader-with-cuda-tensor/43390/2