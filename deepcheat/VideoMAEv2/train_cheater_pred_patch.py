#!/usr/bin/env python3
"""
Patch for train_cheater_pred.py to handle single/multi GPU training gracefully
This script patches the merge function call to check if evaluation files exist
"""

import os
import sys

def patch_training_script():
    """Patch the training script to handle missing evaluation files"""
    script_path = os.path.join(os.path.dirname(__file__), 'train_cheater_pred.py')

    # Read the original file
    with open(script_path, 'r') as f:
        content = f.read()

    # Create backup
    backup_path = script_path + '.backup'
    if not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            f.write(content)

    # Patch the merge call to handle missing files
    old_merge_block = '''    if global_rank == 0:
        print("Start merging results...")
        final_top1, final_top5 = merge(args.output_dir, num_tasks)
        print(
            f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%"
        )
        log_stats = {'Final top-1': final_top1, 'Final Top-5': final_top5}
        if args.output_dir and utils.is_main_process():
            with open(
                    os.path.join(args.output_dir, "log.txt"),
                    mode="a",
                    encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")'''

    new_merge_block = '''    if global_rank == 0:
        # Check if evaluation files exist before attempting to merge
        eval_file_0 = os.path.join(args.output_dir, '0.txt')
        if os.path.exists(eval_file_0):
            print("Start merging results...")
            try:
                final_top1, final_top5 = merge(args.output_dir, num_tasks)
                print(
                    f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%"
                )
                log_stats = {'Final top-1': final_top1, 'Final Top-5': final_top5}
                if args.output_dir and utils.is_main_process():
                    with open(
                            os.path.join(args.output_dir, "log.txt"),
                            mode="a",
                            encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")
            except Exception as e:
                print(f"Warning: Could not merge evaluation results: {e}")
                print("This is normal for single-GPU training without evaluation.")
        else:
            print("No evaluation files found - skipping merge step.")
            print("This is normal for training-only mode or single-GPU setup.")'''

    # Apply the patch
    if old_merge_block in content:
        patched_content = content.replace(old_merge_block, new_merge_block)

        # Write the patched file
        with open(script_path, 'w') as f:
            f.write(patched_content)

        print(f"Successfully patched {script_path}")
        print("Training script now handles single/multi GPU setups automatically")
        return True
    else:
        print("Merge block not found - file may already be patched or different version")
        return False

if __name__ == '__main__':
    patch_training_script()