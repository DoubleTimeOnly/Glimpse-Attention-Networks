import torch


class AttentionVisualizer:
    """Utility class for visualizing attention patterns."""
    
    @staticmethod
    def visualize_attention_sequence(model, image, save_path=None):
        """
        Visualize the sequence of attention locations for a given image.
        
        Args:
            model: trained RAM model
            image: input image tensor (1, C, H, W)
            save_path: optional path to save the visualization
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        model.eval()
        with torch.no_grad():
            # Get attention sequence
            action_logits, locations, _, _ = model(image)
            
            # Convert image to numpy for visualization
            img_np = image.squeeze().cpu().numpy()
            
            # Create figure
            fig, axes = plt.subplots(1, len(locations) + 1, figsize=(15, 3))
            
            # Show original image
            axes[0].imshow(img_np, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Show each glimpse location
            for i, location in enumerate(locations):
                loc = location.cpu().numpy()[0]  # First sample in batch
                
                # Convert from [-1, 1] to pixel coordinates
                h, w = img_np.shape
                x = int((loc[0] + 1) * w / 2)
                y = int((loc[1] + 1) * h / 2)
                
                # Show image with attention location
                axes[i + 1].imshow(img_np, cmap='gray')
                
                # Add attention window
                glimpse_size = 8
                rect = patches.Rectangle(
                    (x - glimpse_size // 2, y - glimpse_size // 2),
                    glimpse_size, glimpse_size,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                axes[i + 1].add_patch(rect)
                axes[i + 1].set_title(f'Glimpse {i + 1}')
                axes[i + 1].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
            plt.show()
            
            # Print prediction
            predicted_class = action_logits.argmax(dim=1).item()
            confidence = torch.softmax(action_logits, dim=1).max().item()
            print(f"Predicted class: {predicted_class} (confidence: {confidence:.3f})")
