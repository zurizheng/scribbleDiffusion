"""
Exponential Moving Average (EMA) for model parameters.
"""

import torch
import torch.nn as nn
from typing import Iterable


class EMAModel:
    """
    Exponential Moving Average of model parameters.
    Keeps track of a moving average of model weights for better sampling.
    """
    
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: float = 1.0,
        power: float = 2/3,
    ):
        """
        Initialize EMA model.
        
        Args:
            parameters: Model parameters to track
            decay: EMA decay rate
            min_decay: Minimum decay rate
            update_after_step: Start EMA updates after this step
            use_ema_warmup: Whether to use EMA warmup
            inv_gamma: Inverse gamma for warmup
            power: Power for warmup
        """
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]
        self.collected_params = []
        
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = 0
    
    @property
    def averaged_model(self):
        """Return a model with EMA weights."""
        # This is a simplified version - in practice you'd need to
        # create a proper model instance and load the EMA weights
        return self
    
    def get_decay(self, optimization_step: int) -> float:
        """Get decay rate for current step."""
        if optimization_step < self.update_after_step:
            return 0.0
        
        if self.use_ema_warmup:
            value = 1 - (1 + optimization_step / self.inv_gamma) ** -self.power
            return max(value, self.min_decay)
        else:
            return self.decay
    
    def step(self, parameters: Iterable[torch.nn.Parameter]):
        """Update EMA parameters."""
        parameters = list(parameters)
        self.optimization_step += 1
        
        decay = self.get_decay(self.optimization_step)
        
        if decay == 0.0:
            return
        
        one_minus_decay = 1.0 - decay
        
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                s_param.sub_(one_minus_decay * (s_param - param))
    
    def copy_to(self, parameters: Iterable[torch.nn.Parameter]):
        """Copy EMA parameters to model."""
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)
    
    def to(self, device=None, dtype=None):
        """Move EMA parameters to device/dtype."""
        self.shadow_params = [
            p.to(device=device, dtype=dtype) for p in self.shadow_params
        ]
        return self
