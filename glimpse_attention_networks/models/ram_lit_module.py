import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.distributions import Normal

from glimpse_attention_networks.models.baseline_network import BaselineNetwork
from glimpse_attention_networks.models.classification_network import ActionNetwork
from glimpse_attention_networks.models.glimpse_network import GlimpseNetwork
from glimpse_attention_networks.models.location_network import LocationNetwork
from glimpse_attention_networks.models.sequence_network import CoreNetwork


class RecurrentAttentionModel(pl.LightningModule):
    """
    Complete Recurrent Attention Model implementation using PyTorch Lightning.
    """
    def __init__(
        self, 
        glimpse_size: int = 8,
        num_patches: int = 1,
        num_glimpses: int = 6,
        num_classes: int = 10,
        channels: int = 1,
        hidden_size: int = 256,
        glimpse_hidden: int = 256,
        location_std: float = 0.17,
        learning_rate: float = 1e-3,
        baseline_coeff: float = 0.5
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.num_glimpses = num_glimpses
        self.location_std = location_std
        self.baseline_coeff = baseline_coeff
        
        # Initialize networks
        self.glimpse_network = GlimpseNetwork(
            glimpse_size=glimpse_size,
            num_patches=num_patches,
            channels=channels,
            hidden_size=128,
            glimpse_hidden=glimpse_hidden
        )
        
        self.core_network = CoreNetwork(
            glimpse_hidden=glimpse_hidden,
            hidden_size=hidden_size
        )
        
        self.location_network = LocationNetwork(
            hidden_size=hidden_size,
            std=location_std
        )
        
        self.action_network = ActionNetwork(
            hidden_size=hidden_size,
            num_classes=num_classes
        )
        
        self.baseline_network = BaselineNetwork(hidden_size=hidden_size)
        
        # Initialize first location (center of image)
        self.register_buffer('init_location', torch.zeros(1, 2))
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list, list, list]:
        """
        Forward pass through the RAM model.
        
        Args:
            x: input image (B, C, H, W)
            
        Returns:
            action_logits: final action predictions (B, num_classes)
            locations: list of sampled locations
            location_log_probs: list of location log probabilities
            baselines: list of baseline predictions
        """
        batch_size = x.size(0)
        
        # Initialize location and hidden state
        location = self.init_location.expand(batch_size, -1)
        hidden_state = None
        
        # Storage for REINFORCE
        locations = []
        location_log_probs = []
        baselines = []
        
        # Take glimpses
        for t in range(self.num_glimpses):
            # Process glimpse
            glimpse_repr = self.glimpse_network(x, location)
            
            # Update core network
            h, hidden_state = self.core_network(glimpse_repr, hidden_state)
            
            # Predict baseline
            baseline = self.baseline_network(h)
            baselines.append(baseline)
            
            # Predict next location (except for last step)
            if t < self.num_glimpses - 1:
                location_mean, location = self.location_network(h)
                
                # Calculate log probability for REINFORCE
                location_dist = Normal(location_mean, self.location_std)
                location_log_prob = location_dist.log_prob(location).sum(dim=1)
                
                locations.append(location)
                location_log_probs.append(location_log_prob)
        
        # Final action prediction
        action_logits = self.action_network(h)
        
        return action_logits, locations, location_log_probs, baselines
    
    def compute_loss(self, action_logits: torch.Tensor, locations: list, 
                    location_log_probs: list, baselines: list, 
                    targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the complete loss function including classification loss,
        REINFORCE loss, and baseline loss.
        """
        batch_size = action_logits.size(0)
        
        # Classification loss (supervised)
        classification_loss = F.cross_entropy(action_logits, targets, reduction='mean')
        
        # Reward: 1 if correct classification, 0 otherwise
        predicted = action_logits.argmax(dim=1)
        rewards = (predicted == targets).float()
        
        # REINFORCE loss for location network
        reinforce_loss = 0
        if len(location_log_probs) > 0:
            for t, (log_prob, baseline) in enumerate(zip(location_log_probs, baselines[:-1])):
                # Use baseline for variance reduction
                advantage = rewards.unsqueeze(1) - baseline.detach()
                reinforce_loss += -(log_prob.unsqueeze(1) * advantage).mean()
        
        # Baseline loss (MSE between baseline and actual reward)
        baseline_loss = 0
        for baseline in baselines:
            baseline_loss += F.mse_loss(baseline, rewards.unsqueeze(1))
        baseline_loss /= len(baselines)
        
        return classification_loss, reinforce_loss, baseline_loss
    
    def training_step(self, batch, batch_idx):
        x, targets = batch
        
        # Forward pass
        action_logits, locations, location_log_probs, baselines = self(x)
        
        # Compute losses
        classification_loss, reinforce_loss, baseline_loss = self.compute_loss(
            action_logits, locations, location_log_probs, baselines, targets
        )
        
        # Total loss
        total_loss = classification_loss + reinforce_loss + self.baseline_coeff * baseline_loss
        
        # Logging
        self.log('train_classification_loss', classification_loss)
        self.log('train_reinforce_loss', reinforce_loss)
        self.log('train_baseline_loss', baseline_loss)
        self.log('train_total_loss', total_loss)
        
        # Accuracy
        predicted = action_logits.argmax(dim=1)
        accuracy = (predicted == targets).float().mean()
        self.log('train_accuracy', accuracy)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, targets = batch
        
        # Forward pass
        action_logits, locations, location_log_probs, baselines = self(x)
        
        # Compute losses
        classification_loss, reinforce_loss, baseline_loss = self.compute_loss(
            action_logits, locations, location_log_probs, baselines, targets
        )
        
        # Total loss
        total_loss = classification_loss + reinforce_loss + self.baseline_coeff * baseline_loss
        
        # Logging
        self.log('val_classification_loss', classification_loss)
        self.log('val_reinforce_loss', reinforce_loss)
        self.log('val_baseline_loss', baseline_loss)
        self.log('val_total_loss', total_loss)
        
        # Accuracy
        predicted = action_logits.argmax(dim=1)
        accuracy = (predicted == targets).float().mean()
        self.log('val_accuracy', accuracy)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        x, targets = batch
        
        # Forward pass
        action_logits, locations, location_log_probs, baselines = self(x)
        
        # Compute losses
        classification_loss, reinforce_loss, baseline_loss = self.compute_loss(
            action_logits, locations, location_log_probs, baselines, targets
        )
        
        # Total loss
        total_loss = classification_loss + reinforce_loss + self.baseline_coeff * baseline_loss
        
        # Logging
        self.log('test_classification_loss', classification_loss)
        self.log('test_reinforce_loss', reinforce_loss)
        self.log('test_baseline_loss', baseline_loss)
        self.log('test_total_loss', total_loss)
        
        # Accuracy
        predicted = action_logits.argmax(dim=1)
        accuracy = (predicted == targets).float().mean()
        self.log('test_accuracy', accuracy)
        
        return total_loss
    
    def configure_optimizers(self):
        # Use separate optimizers for different components if needed
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
