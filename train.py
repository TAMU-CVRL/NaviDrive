from driverEngine import driverEngine
from utils.data_utils import load_config
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a samll-size driver LLM")
    parser.add_argument("--config", type=str, default="default.yaml", help="Path to the configuration YAML file")
    
    args = parser.parse_args()
    config = load_config(args.config)
    trainer = driverEngine(config)
    trainer.train()
    