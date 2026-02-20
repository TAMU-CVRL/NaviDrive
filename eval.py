from driverEngine import driverEngine
from utils.data_utils import load_config
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a samll-size driver LLM")
    parser.add_argument("--config", type=str, default="default.yaml", help="Path to the configuration YAML file")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the checkpoint to evaluate")
    parser.add_argument("--inference_path", type=str, help="Path to inference dataset (JSONL)")
    parser.add_argument("--eval_path", type=str, help="Path to evaluation dataset (JSONL)")
    parser.add_argument("--eval_L2", type=bool, default=False, help="Whether to evaluate L2 distance for predicted waypoints")
    parser.add_argument("--eval_video", type=bool, default=False, help="Whether to generate evaluation video")
    parser.add_argument("--eval_images", type=bool, default=False, help="Whether to generate evaluation images")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for evaluation")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for evaluation")
    
    args = parser.parse_args()
    config = load_config(args.config)
    trainer = driverEngine(config)
    # Load trained model
    trainer.load_model_from_checkpoint(args.ckpt_path)
    # Inference: Generate predicted waypoints and save to JSONL
    if args.inference_path:
        print(f"Running inference on dataset: {args.inference_path}")
        trainer.inference(inference_path=args.inference_path)
    # Evaluation
    if args.eval_L2:
        trainer.eval_L2(eval_path=args.eval_path)
    else:
        print("L2 evaluation skipped. Set --eval_L2 to True to enable L2 distance evaluation.")
    if args.eval_video:
        trainer.eval_video(eval_path=args.eval_path, start_idx=args.start_idx, end_idx=args.end_idx)
    else:
        print("Video evaluation skipped. Set --eval_video to True to enable video generation.")
    if args.eval_images:
        trainer.eval_images(eval_path=args.eval_path, start_idx=args.start_idx, end_idx=args.end_idx)
    else:
        print("Image evaluation skipped. Set --eval_images to True to enable image generation.")
