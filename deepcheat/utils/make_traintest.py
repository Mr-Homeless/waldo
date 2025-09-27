import os
import argparse
import random

def main(directory):
    train_csv_lines = []
    test_csv_lines = []

    for subdir, _, files in os.walk(directory):
        if subdir == directory:  # Skip the root directory
            continue

        num_files = len(files)
        
        if num_files == 0:
            continue

        line_format = f"{subdir} 0 {num_files - 1}"

        # Randomly decide whether to include in train or test set
        if random.random() <= 0.9:
            train_csv_lines.append(line_format)
        else:
            test_csv_lines.append(line_format)

    with open("train.csv", "w") as train_csv:
        train_csv.write("\n".join(train_csv_lines))

    with open("test.csv", "w") as test_csv:
        test_csv.write("\n".join(test_csv_lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate train and test CSV files.")
    parser.add_argument("directory", help="Path to the directory containing subdirectories.")

    args = parser.parse_args()
    main(args.directory)
