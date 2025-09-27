import argparse
import os
from pathlib import Path
from common import ensure_dir_exists
from paddleocr import PaddleOCR
# from default_video_cropper import DefaultVideoCropper
import cv2

parser = argparse.ArgumentParser(
    description="Segment video frames for using timeframes", add_help=False
)
args = vars(parser.parse_args())

# set directory path
frames_path = Path('./')
ensure_dir_exists(frames_path)


ocr = PaddleOCR(use_angle_cls=True)

# loop through each file in the directory
for file in os.listdir(frames_path):
    # check if the file extension is .jpg or .jpeg
    if file.endswith('.jpg'):
        # print the filename if it is a JPG file
        image = cv2.imread(file)
        height, width = image.shape[:2]
        crop_width = int(width * 0.2)
        crop_height = int(height * 0.2)
        left = crop_width
        top = 0
        right = width
        bottom = crop_height
        cropped_image = image[top:crop_height, crop_width:width]
        #cv2.imwrite(f"{output_path}/{os.path.basename(file).split('/')[-1]}", cropped_image)
        result = ocr.ocr(cropped_image)
        for line in result:
            for word in line:
                if word[1][1] >= 0.85:
                    print(word[1])