from glimpse_attention_networks.train import train_ram_model
from glimpse_attention_networks.utils.config import load_config


if __name__ == "__main__":
    # Train the model
    config = load_config(
        config_name='train', 
        overrides=[
            "model.num_glimpses=5",
        ]
    )
    print("Starting RAM training...")
    experiment_name = "baseline_with_extra_train"
    train_ram_model(config, experiment_name)
    print("Training completed!")