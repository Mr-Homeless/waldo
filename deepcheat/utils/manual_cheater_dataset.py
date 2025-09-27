import os
import csv
import argparse

def create_dataset_csv(txt_dir, img_dir, output_csv):
    # List to store rows for csv
    dataset = []
    
    # 1. Iterate over the directory containing the .mp4.txt files.
    for txt_file in os.listdir(txt_dir):
        if txt_file.endswith('.mp4.txt'):
            file_base_name = txt_file.replace('.mp4.txt', '')  # e.g., 'filename' for 'filename.mp4.txt'
            
            # Check for "Aim-Assist ON" in the filename
            cheater = 1 if "Aim-Assist ON" in txt_file else 0

            # Get the corresponding directory for this txt_file in img_dir
            corresponding_dir = os.path.join(img_dir, file_base_name)
            
            # Check if the directory exists
            if os.path.exists(corresponding_dir):
                # Read the indices from the txt file
                with open(os.path.join(txt_dir, txt_file), 'r') as f:
                    indices = f.readline().strip().split(',')
                    # 2. For each index, create a corresponding entry in the dataset
                    for idx in indices:
                        img_name = f"img_{int(idx):010}.jpg"  # Format the index to 10 digits
                        dataset.append([corresponding_dir, img_name, idx, cheater])
    
    # 3. Write the dataset to the CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Directory Path', 'Filename', 'Index'])  # Writing the headers
        csvwriter.writerows(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Center crop videos in a directory.")
    parser.add_argument("--txt_dir", required=True, help="Source directory containing the videos.")
    parser.add_argument("--img_dir", required=True, help="Destination directory to save the cropped videos.")
    parser.add_argument("--output_csv_file", type=str, default="simplecheater.csv", help="Size of the center crop.")

    args = parser.parse_args()
    create_dataset_csv(args.txt_dir, args.img_dir, args.output_csv_file)
