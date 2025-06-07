from glimpse_attention_networks.train import train_ram_model
from glimpse_attention_networks.utils.config import load_config

def run_experiment(num_glimpses, glimpse_size):
    config = load_config(
        config_name='train', 
        overrides=[
            f"model.num_glimpses={num_glimpses}",
            f"model.glimpse_size={glimpse_size}"
        ]
    )
    print("Starting RAM training...")
    experiment_name = f"glimpse_{num_glimpses}_baseline_with_extra_train_glimpse_size_{glimpse_size}"
    train_ram_model(config, experiment_name)
    print("Training completed!")

def run_experiment_lr(learning_rate):
    config = load_config(
        config_name='train', 
        overrides=[
            f"model.learning_rate={learning_rate}"
        ]
    )
    print("Starting RAM training...")
    experiment_name = f"learning_rate_{learning_rate}"
    train_ram_model(config, experiment_name)
    print("Training completed!")

def run_experiment_gam(num_glimpses, glimpse_size):
    config = load_config(
        config_name='train_gam', 
        overrides=[
            f"model.num_glimpses={num_glimpses}",
            f"model.glimpse_size={glimpse_size}",
            f"trainer.max_steps=10000",
        ]
    )

    print("Starting GAM training...")
    experiment_name = f"gam_baseline"
    train_ram_model(config, experiment_name)
    print("Training completed!")

if __name__ == "__main__":
    # Train the model
    # for glimpse_size in [12, 16, 8]:
    # for glimpse_size in [8]:
    #     for num_glimpses in [3, 6]:
    #         run_experiment(
    #             num_glimpses=num_glimpses, 
    #             glimpse_size=glimpse_size
    #         )
    
    # run_experiment(num_glimpses=3, glimpse_size=12)
    # run_experiment_gam(num_glimpses=3, glimpse_size=12)
    for learning_rate in [1e-2, 1e-3]:
        run_experiment_lr(learning_rate)
