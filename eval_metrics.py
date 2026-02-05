import json
import numpy as np
import os
import argparse
from datetime import datetime

def calculate_metrics(gt, pred, threshold=2.0):
    """
    gt: np.array (N, 2)
    pred: np.array (N, 2)
    """
    gt = np.array(gt)
    pred = np.array(pred)
    
    min_len = min(len(gt), len(pred))
    gt, pred = gt[:min_len], pred[:min_len]
    
    # L2 Distance
    errors = np.linalg.norm(gt - pred, axis=1)
    
    # 1s->idx 1, 2s->idx 3, 3s->idx 5, 6s->idx 11
    metrics = {
        "l2_1s": errors[1] if min_len > 1 else np.nan,
        "l2_2s": errors[3] if min_len > 3 else np.nan,
        "l2_3s": errors[5] if min_len > 5 else np.nan,
        "l2_6s": errors[11] if min_len > 11 else errors[-1], # 6s即为FDE
        "ade": np.mean(errors)
    }
    
    # Failure rate
    metrics['is_failure'] = 1 if metrics['l2_6s'] > threshold else 0
    
    return metrics

def format_results(avg_metrics, input_file, total_samples, threshold):
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    width = 85
    double_line = "=" * width
    single_line = "-" * width

    header_section = (
        f"{double_line}\n"
        f"  TRAJECTORY EVALUATION REPORT  |  {date_str}\n"
        f"{double_line}\n"
        f"  [Input File]  : {input_file}\n"
        f"  [Sample Count]: {total_samples}\n"
        f"  [Failure Thresh] : {threshold} m\n"
        f"{single_line}\n"
    )
    
    table_header = (
        f"  {'Metric':<15} | {'1.0s':<10} | {'2.0s':<10} | {'3.0s':<10} | {'6.0s (FDE)':<12} | {'Avg (ADE)':<10}\n"
        f"  {'-'*15}-|-{'-'*10}-|-{'-'*10}-|-{'-'*10}-|-{'-'*12}-|-{'-'*10}\n"
    )
    
    table_row = (
        f"  {'L2 Error (m)':<15} | "
        f"{avg_metrics['L2_1s']:<10.3f} | "
        f"{avg_metrics['L2_2s']:<10.3f} | "
        f"{avg_metrics['L2_3s']:<10.3f} | "
        f"{avg_metrics['L2_6s']:<12.3f} | "
        f"{avg_metrics['ADE_avg']:<10.3f}\n"
    )
    
    summary_section = (
        f"{single_line}\n"
        f"  {'OVERALL PERFORMANCE':<15}\n"
        f"  > Failure Rate : {avg_metrics['Failure_Rate']:>6.2f} %\n"
        f"  > Reliability  : {100 - avg_metrics['Failure_Rate']:>6.2f} % (within {threshold}m)\n"
        f"{double_line}\n"
    )
    
    return header_section + table_header + table_row + summary_section

def process_eval(input_file, threshold=2.0):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return

    all_results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                res = calculate_metrics(data['gt_waypoints'], data['pred_waypoints'], threshold=threshold)
                all_results.append(res)
            except Exception as e:
                print(f"Skipping malformed line: {e}")

    if not all_results:
        print("No valid data processed.")
        return

    avg_metrics = {
        "L2_1s": np.nanmean([r['l2_1s'] for r in all_results]),
        "L2_2s": np.nanmean([r['l2_2s'] for r in all_results]),
        "L2_3s": np.nanmean([r['l2_3s'] for r in all_results]),
        "L2_6s": np.nanmean([r['l2_6s'] for r in all_results]),
        "ADE_avg": np.mean([r['ade'] for r in all_results]),
        "Failure_Rate": np.mean([r['is_failure'] for r in all_results]) * 100
    }

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    date_str = datetime.now().strftime("%Y%m%d")
    # file_base = os.path.splitext(os.path.basename(input_file))[0]
    output_path = os.path.join(output_dir, f"{date_str}_results.txt")

    result_text = format_results(avg_metrics, input_file, len(all_results), threshold)
    
    with open(output_path, 'a', encoding='utf-8') as f_out:
        f_out.write(result_text)

    print(result_text)
    print(f"Results saved to: {output_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate evaluation metrics (ADE/FDE)")
    parser.add_argument("--eval_data", type=str, default="eval_results_8B.jsonl", help="Path to the evaluation data JSONL file")
    parser.add_argument("--threshold", type=float, default=2.0, help="Failure threshold in meters")
    args = parser.parse_args()
    process_eval(args.eval_data, threshold=args.threshold)
    