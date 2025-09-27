import cv2
import os
import argparse
from pathlib import Path
from tqdm import tqdm

def center_crop(frame, crop_size):
    y, x, c = frame.shape
    start_x = x // 2 - (crop_size // 2)
    start_y = y // 2 - (crop_size // 2)
    return frame[start_y:start_y + crop_size, start_x:start_x + crop_size]

def process_videos(src_dir, dest_dir, crop_size, make_train_file=True):
    # Create destination directory if it doesn't exist
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    if crop_size > 240:
        raise ValueError("Crop size cannot be greater than 240.")
    #max_frames = max([ cv2.VideoCapture(os.path.join(src_dir, filename)).get(cv2.CAP_PROP_FRAME_COUNT) for filename in os.listdir(src_dir)])
    # Create a file to hold the annotations
    with open(os.path.join(dest_dir,"train.csv"), "w") as train_csv:
        for filename in tqdm(os.listdir(src_dir)):
            video_path = os.path.join(src_dir, filename)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Skipped {filename}: Could not open the file.")
                continue

            # Create a directory for each video's frames
            frame_dir = os.path.join(dest_dir, filename.split('.')[0])
            frame_dir = os.path.abspath(frame_dir)
            Path(frame_dir).mkdir(parents=True, exist_ok=True)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            annotation = f"{frame_dir} {15} 1\n"

            # Write annotation to train.csv
            if make_train_file: train_csv.write(annotation)

            frame_number = 0
            save_frame_number = 0  # This will be used to label frames as 0-15
            pbar = tqdm(total=total_frames)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if 85 <= frame_number <= 100:
                    # Center crop
                    cropped_frame = center_crop(frame, crop_size)

                    # Save frame as image
                    zz = 10
                    img_filename = f"img_{save_frame_number:0{zz}d}.jpg"
                    img_path = os.path.join(frame_dir, img_filename)
                    cv2.imwrite(img_path, cropped_frame)

                    save_frame_number += 1  # Increment the save frame number only for frames in the desired range

                frame_number += 1  # Increment the overall frame number regardless
                pbar.update(1)

            cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Center crop videos in a directory.")
    parser.add_argument("--src_dir", required=True, help="Source directory containing the videos.")
    parser.add_argument("--dest_dir", required=True, help="Destination directory to save the cropped videos.")
    parser.add_argument("--crop_size", type=int, required=True, help="Size of the center crop.")

    args = parser.parse_args()
    process_videos(args.src_dir, args.dest_dir, args.crop_size)
