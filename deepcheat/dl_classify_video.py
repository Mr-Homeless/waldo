import os

import csv
import argparse
import uuid
import os
import subprocess
import numpy as np



from utils.download_links import download_videos_from_csv
from utils.bb_killshot import write_bb_kills
from utils.combo_cheater_detection import make_test_set
from utils.new_crop import process_videos


# Step 1: Parse command-line argument to identify the type of input
parser = argparse.ArgumentParser(description='Process a YouTube URL or a CSV of YouTube URLs.')
parser.add_argument('--url', type=str, help='A single YouTube URL')
parser.add_argument('--csv', type=str, help='Path to the CSV file containing YouTube URLs')
parser.add_argument('--dl_dir', type=str, help='Path to download the directory')
parser.add_argument('--ischeater', type=int, choices=[0,1])

args = parser.parse_args()

# Step 2: Function to generate temporary CSV from a single URL
def generate_csv_from_url(url, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "id", "url", "game"])
        writer.writerow([0, str(uuid.uuid4()), url, "CSG"])



# Step 4: Identify the type of input and call the respective function

if args.url:
    filename = "temp.csv"
    generate_csv_from_url(args.url, filename)
elif args.csv: 
    filename = args.url
else:
    print("Please provide either a URL or a CSV file.")


if not os.path.exists(args.dl_dir):
    os.makedirs(args.dl_dir)

download_videos_from_csv(filename, args.dl_dir)

# List all the .mp4 filenames in args.dl_dir
mp4_files = [] 
for f in os.listdir(args.dl_dir):
	if args.url:
		video_id = os.path.splitext(f)[0]
		if video_id not in args.url:
			continue
		else:
			mp4_files = [os.path.abspath(os.path.join(args.dl_dir,f))]
			break
	elif f.endswith('.mp4'):
		mp4_files.append(os.path.abspath(os.path.join(args.dl_dir,f)))


process_videos(args.dl_dir, args.dl_dir, 240, make_train_file=False)

for mp4_file in mp4_files:
	video_dir = os.path.splitext(mp4_file)[0]

	#killshot finder
	bb_file = os.path.join(video_dir, "bb_labels.txt")
	write_bb_kills(mp4_file, bb_file)


	os.chdir('./VideoMAEv2')
	subprocess.run(['./run_killshot.sh', '--pred_video', video_dir])

	# combine both killshots into 1 file
	make_test_set(bb_file, video_dir, args.ischeater, os.path.join(video_dir,"test.csv"))

	subprocess.run(['./eval_cheater.sh', video_dir])
	preds = np.loadtxt(os.path.join(video_dir,"cheater_preds.txt"))
	print("cheating suspicion: ", (preds > .5).sum() / len(preds) )

	os.chdir('..')

