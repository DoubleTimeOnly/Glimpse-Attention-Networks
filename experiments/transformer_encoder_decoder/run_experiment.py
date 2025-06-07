from glimpse_attention_networks.train import train_ram_model
from glimpse_attention_networks.utils.config import load_config

def run_experiment():
    config = load_config(
        config_name='train_gam', 
        # config_name='train', 
        overrides=[
        ]
    )
    print("Starting GAM training...")
    experiment_name = f"transformer_encoder_decoder_testing"
    train_ram_model(config, experiment_name)
    print("Training completed!")


if __name__ == "__main__":
    run_experiment()
