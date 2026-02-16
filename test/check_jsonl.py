import json
import argparse
import sys
from pathlib import Path

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def check_structure(file_path, verbose=False):
    print(f"Checking file: {file_path} ...\n")
    
    total_count = 0
    valid_count = 0
    incomplete_count = 0
    empty_count = 0
    
    problematic_samples = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                total_count += 1
                try:
                    data = json.loads(line)
                    token = data.get('token', f'Line {line_num}')
                    reasons = data.get('reasons', [])

                    if not reasons or not reasons[0]:
                        empty_count += 1
                        problematic_samples.append((token, "Empty Response"))
                        if verbose:
                            print(f"{RED}[Empty]{RESET} Token: {token}")
                        continue

                    content = reasons[0]
                    
                    content_lower = content.lower()
                    has_perception = "perception" in content_lower
                    has_action = "action" in content_lower
                    has_reasoning = "reasoning" in content_lower or "reason" in content_lower

                    if has_perception and has_action and has_reasoning:
                        valid_count += 1
                    else:
                        incomplete_count += 1
                        missing = []
                        if not has_perception: missing.append("Perception")
                        if not has_action: missing.append("Action")
                        if not has_reasoning: missing.append("Reasoning")
                        
                        problematic_samples.append((token, f"Missing: {', '.join(missing)}"))
                        if verbose:
                            print(f"{YELLOW}[Incomplete]{RESET} Token: {token} | Missing: {missing}")

                except json.JSONDecodeError:
                    print(f"{RED}[Error]{RESET} Line {line_num} is not valid JSON.")
                    continue

    except FileNotFoundError:
        print(f"{RED}Error: File not found: {file_path}{RESET}")
        sys.exit(1)

    print("\n" + "="*40)
    print(f"  **Validation Report**")
    print("="*40)
    print(f"Total Samples:      {total_count}")
    print(f"{GREEN}Valid Samples:    {valid_count} ({valid_count/total_count*100:.1f}%){RESET}")
    print(f"{YELLOW}Incomplete:       {incomplete_count} ({incomplete_count/total_count*100:.1f}%){RESET}")
    print(f"{RED}Empty/Error:      {empty_count}{RESET}")
    print("="*40)

    if incomplete_count > 0:
        print(f"\n Tip: Found {incomplete_count} incomplete samples.")
        print("   Likely causes: 'max_output_tokens' too low or safety filters triggered.")
        
        print("\n First 5 Problematic Tokens:")
        for t, issue in problematic_samples[:5]:
            print(f"   - {t}: {issue}")
            
    if valid_count == total_count and total_count > 0:
        print(f"\n{GREEN} Perfect! All data looks good.{RESET}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check JSONL for completeness of Perception/Action/Reasoning.")
    parser.add_argument("--file_path", type=str, help="Path to the .jsonl file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print every bad sample token")
    
    args = parser.parse_args()
    
    check_structure(args.file_path, args.verbose)
    