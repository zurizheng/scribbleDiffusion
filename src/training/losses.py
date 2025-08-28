"""
Loss functions for diffusion training.
"""

import torch
import torch.nn.functional as F


class DiffusionLoss:
    """
    Loss function for diffusion model training.
    Supports different weighting schemes and objectives.
    """
    
    def __init__(self, config):
        self.config = config
        self.loss_type = config.get("loss_type", "mse")
        self.snr_gamma = config.get("snr_gamma", None)
    
    def __call__(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute diffusion loss.
        
        Args:
            model_pred: Model prediction (noise or v-prediction)
            target: Ground truth target (noise or v-target)
            timesteps: Diffusion timesteps
            
        Returns:
            Loss tensor
        """
        if self.loss_type == "mse":
            loss = F.mse_loss(model_pred, target, reduction="none")
        elif self.loss_type == "l1":
            loss = F.l1_loss(model_pred, target, reduction="none")
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        # Apply SNR weighting if specified
        if self.snr_gamma is not None:
            # Compute SNR weighting based on timesteps
            snr_weights = self._compute_snr_weights(timesteps)
            loss = loss * snr_weights.view(-1, 1, 1, 1)
        
        return loss.mean()
    
    def _compute_snr_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute signal-to-noise ratio based weighting."""
        # This is a simplified SNR weighting
        # In practice, you'd compute based on the noise schedule
        snr = 1.0 / (timesteps.float() / 1000.0 + 1e-8)
        weights = torch.clamp(snr / self.snr_gamma, max=1.0)
        return weights
