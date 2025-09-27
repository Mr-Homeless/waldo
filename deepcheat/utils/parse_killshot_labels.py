import os
import csv
from sklearn.model_selection import train_test_split

def extract_frame_index(frame_id):
    """
    Extract the integer part from the frame ID.
    Example: "img_0000002450" -> 2450
    """
    return int(frame_id.split('_')[-1])

def parse_labels(label_dir, video_dir):
    """
    Parse the annotation CSV files and returns the required dataset.
    """
    all_data = []

    # List all directories under the killshot_labels directory
    for dir_name in os.listdir(label_dir):
        if "train.csv" in dir_name: continue
        if "test.csv" in dir_name: continue
        #if not os.path.isdir(dir_name): continue
        annotation_file = os.path.join(label_dir, dir_name, "annotations", "default-annotations-human-imagelabels.csv")

        # Parse the CSV
        with open(annotation_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip the header
            for row in reader:
                frame_id = row[0]
                frame_index = extract_frame_index(frame_id)
                frame_of_kill = frame_id + ".jpg"
                path_to_frame_dir = os.path.join(video_dir, dir_name)
                all_data.append([path_to_frame_dir, frame_of_kill, frame_index])

    return all_data

def save_to_csv(data, filename):
    """
    Save the data to a CSV file.
    """
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["path_to_frame_dir", "frame_of_kill", "index_of_frame"])
        writer.writerows(data)

def main(label_dir, video_dir):
    # Parse the labels
    data = parse_labels(label_dir, video_dir)

    # Split the data into train and test
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

    # Save the datasets to CSV files
    save_to_csv(train_data, os.path.join(label_dir, 'train.csv'))
    save_to_csv(test_data, os.path.join(label_dir, 'test.csv'))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('label_dir', help='Path to the killshot_labels directory')
    parser.add_argument('video_dir', help='Path to the video frames directory')
    
    args = parser.parse_args()

    main(args.label_dir, args.video_dir)

