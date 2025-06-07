import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.distributions import Normal
from pathlib import Path
from typing import List, Dict, Any

from glimpse_attention_networks.models.baseline_network import BaselineNetwork
from glimpse_attention_networks.models.classification_network import ActionNetwork
from glimpse_attention_networks.models.context_network import ContextNetwork
from glimpse_attention_networks.models.glimpse_network import GlimpseNetwork
from glimpse_attention_networks.models.location_network import LocationNetwork
from glimpse_attention_networks.models.sequence_network import CoreNetwork, TransformerEncoderDecoder
from glimpse_attention_networks.visualization.attention_visualizer import AttentionVisualizer


class GlimpseAttentionModel(pl.LightningModule):
    """
    Complete Recurrent Attention Model implementation using PyTorch Lightning.
    """
    def __init__(
        self, 
        coarse_wh: tuple[int, int] = (16, 16),
        glimpse_size: int = 8,
        num_patches: int = 2,
        num_glimpses: int = 6,
        num_classes: int = 10,
        channels: int = 1,
        hidden_size: int = 256,
        glimpse_hidden: int = 256,
        location_std: float = 0.17,
        learning_rate: float = 1e-3,
        baseline_coeff: float = 0.5,
        num_visualization_samples: int = 5,
        visualization_dir: str = "diags",
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.num_glimpses = num_glimpses
        self.location_std = location_std
        self.baseline_coeff = baseline_coeff
        self.num_visualization_samples = num_visualization_samples
        self.visualization_dir = visualization_dir
        self._hidden_size = hidden_size
        
        # Initialize networks
        self.glimpse_network = GlimpseNetwork(
            glimpse_size=glimpse_size,
            num_patches=num_patches,
            channels=channels,
            hidden_size=128,
            glimpse_hidden=glimpse_hidden
        )
        
        self.core_network = TransformerEncoderDecoder(
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

        self.context_network = ContextNetwork(out_channels=hidden_size, input_wh=coarse_wh)
        
        self.baseline_network = BaselineNetwork(hidden_size=hidden_size)
        
        # Initialize first location (center of image)
        self.register_buffer('init_location', torch.zeros(1, 2))
        
        # Store test samples for visualization
        self.test_samples = None
        self.test_labels = None
        
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
        
        # Storage for REINFORCE
        locations = []
        location_log_probs = []
        baselines = []
        action_logits = []
        
        # Initialize location and hidden state
        init_state = self.context_network(x)

        glimpse_context = [self.core_network.glimpse_start_token.expand(batch_size, -1)]
        locations = [self.core_network.location_start_token.expand(batch_size, -1)]
        where_vec, _ = self.core_network(
            glimpse_context=glimpse_context,
            previous_queries=locations,
            global_context=init_state,
        )

        # Take glimpses
        for t in range(self.num_glimpses):
            # Predict next location (except for last step)
            location_mean, location = self.location_network(where_vec)
            
            # Process glimpse
            glimpse_vector = self.glimpse_network(x, location)
            glimpse_context.append(glimpse_vector)

            # Calculate log probability for REINFORCE
            location_dist = Normal(location_mean, self.location_std)
            location_log_prob = location_dist.log_prob(location).sum(dim=1)
            
            locations.append(location)
            location_log_probs.append(location_log_prob)

            # Update core network
            where_vec, what_vec = self.core_network(
                glimpse_context=glimpse_context,
                global_context=init_state,
                previous_queries=locations,
            )

            # action prediction
            action_logit = self.action_network(what_vec)
            action_logits.append(action_logit)

            # Predict baseline
            baseline = self.baseline_network(where_vec)
            baselines.append(baseline)
        
        return action_logits, locations[1:], location_log_probs, baselines
    
    def compute_loss(
        self,
        action_logits: torch.Tensor,
        locations: list,
        location_log_probs: list,
        baselines: list,
        targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the complete loss function including classification loss,
        REINFORCE loss, and baseline loss.
        """
        # N, C, G
        action_logits = torch.stack(action_logits, dim=2)
        # N, G
        sequence_targets = targets.unsqueeze(1).expand(-1, self.num_glimpses)

        # Classification loss (supervised)
        classification_loss = F.cross_entropy(action_logits, sequence_targets, reduction='mean')
        
        # Reward: 1 if correct classification, 0 otherwise
        predicted = action_logits.argmax(dim=1)     # N, G
        rewards = (predicted == sequence_targets).float()    #  N, G

        # N, G
        baselines = torch.cat(baselines, dim=1)

        advantage = rewards - baselines.detach()
        location_log_probs = torch.stack(location_log_probs, dim=1)     # N,G

        reinforce_loss = -(location_log_probs * advantage).mean()

        baseline_loss = F.mse_loss(baselines, rewards)

        return classification_loss, reinforce_loss, baseline_loss

    def training_step(self, batch, batch_idx):
        x, targets = batch
        
        # Forward pass
        action_logits, locations, location_log_probs, baselines = self(x)
        
        # Compute losses
        classification_loss, reinforce_loss, baseline_loss = self.compute_loss(
            action_logits, locations, location_log_probs, baselines, targets
        )

        total_loss = classification_loss + reinforce_loss * 0.1 + baseline_loss
        # total_loss = classification_loss + reinforce_loss + baseline_loss
        
        # Logging
        self.log('train/classification_loss', classification_loss)
        self.log('train/reinforce_loss', reinforce_loss)
        self.log('train/baseline_loss', baseline_loss)
        self.log('train/total_loss', total_loss)
        
        # Accuracy
        # TODO: check if this is correct
        predicted = action_logits[-1].argmax(dim=1)
        accuracy = (predicted == targets).float().mean()
        self.log('train_accuracy', accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
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
        self.log('val/classification_loss', classification_loss)
        self.log('val/reinforce_loss', reinforce_loss)
        self.log('val/baseline_loss', baseline_loss)
        self.log('val/total_loss', total_loss, on_step=False, on_epoch=True)
        
        # Accuracy
        # TODO: check if this is correct
        predicted = action_logits[-1].argmax(dim=1)
        accuracy = (predicted == targets).float().mean()
        self.log('val_accuracy', accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
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
        self.log('test/classification_loss', classification_loss)
        self.log('test/reinforce_loss', reinforce_loss)
        self.log('test/baseline_loss', baseline_loss)
        self.log('test/total_loss', total_loss)
        
        # Accuracy
        # TODO: check if this is correct
        predicted = action_logits[-1].argmax(dim=1)
        accuracy = (predicted == targets).float().mean()
        self.log('test_accuracy', accuracy)
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def on_test_batch_end(self, outputs, batch, batch_idx):
        """Store first test batch for visualization."""
        if batch_idx == 0:  # Only store samples from first batch
            images, labels = batch
            self.test_samples = images
            self.test_labels = labels

    def on_test_epoch_end(self):
        """Create attention visualizations after testing."""
        if self.test_samples is None:
            return

        # Create visualization directory
        save_dir = Path(self.trainer.logger.log_dir) / "diags"
        save_dir.mkdir(exist_ok=True)
        
        # Limit to num_visualization_samples
        images = self.test_samples[:self.num_visualization_samples]
        labels = self.test_labels[:self.num_visualization_samples]
        
        # Create visualizer
        visualizer = AttentionVisualizer()
        
        # Visualize attention for each sample
        for i, (image, label) in enumerate(zip(images, labels)):
            # Add batch dimension
            image = image.unsqueeze(0)
            # Get model prediction
            with torch.no_grad():
                logits, _, _, _ = self(image)
                pred = logits[-1].argmax(dim=1).item()

            # Create visualization
            name = f"attention_sample_{i}_true_{label.item()}_pred_{pred}.png"
            save_path = save_dir / name
            visualizer.visualize_attention_sequence(
                self, 
                image, 
                save_path=str(save_path),
                glimpse_size=self.hparams.glimpse_size,
                num_patches=self.hparams.num_patches
            )
            print(f"Saved attention visualization to {save_path}")
        

