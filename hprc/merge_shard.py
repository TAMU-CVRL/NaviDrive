import json
from pathlib import Path

def merge_nuscenes_shards(data_dir, output_filename, num_shards=6):
    base_path = Path(data_dir)
    final_output_path = base_path / output_filename
    grand_total_samples = 0
    
    print(f"Starting merge of {num_shards} shards into: {output_filename}")
    
    # Using 'w' mode to create a fresh file for the final combined dataset
    with open(final_output_path, 'w', encoding='utf-8') as outfile:
        for shard_id in range(num_shards):
            # Matches the naming convention used in your SLURM script [cite: 8, 12]
            shard_name = f"nuscenes_reasons_32B_{shard_id}.jsonl"
            shard_path = base_path / shard_name
            
            if not shard_path.exists():
                print(f"[Warning] Shard file missing: {shard_name}. Skipping...")
                continue
                
            shard_samples_count = 0
            with open(shard_path, 'r', encoding='utf-8') as infile:
                for line_number, line in enumerate(infile, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        # Verify the line is valid JSON before writing to the final file
                        json.loads(line)
                        outfile.write(line + '\n')
                        shard_samples_count += 1
                    except json.JSONDecodeError:
                        print(f"[Error] Corrupt JSON on line {line_number} in {shard_name}. Skipping line.")
            
            print(f" - {shard_name}: Successfully merged {shard_samples_count} samples.")
            grand_total_samples += shard_samples_count

    print("-" * 50)
    print(f"Merge Complete!")
    print(f"Final file: {final_output_path}")
    print(f"Grand Total: {grand_total_samples} samples merged.")

if __name__ == "__main__":
    # Path configuration for your HPRC scratch directory [cite: 9, 12]
    SCRATCH_DATA_DIR = "data"
    FINAL_JSONL_NAME = "nuscenes_reasons_32B.jsonl"
    
    merge_nuscenes_shards(
        data_dir=SCRATCH_DATA_DIR, 
        output_filename=FINAL_JSONL_NAME,
        num_shards=6
    )