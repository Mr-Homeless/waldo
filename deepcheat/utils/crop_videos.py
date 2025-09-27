"""
Takes a given video and outputs cropped frames of the video.
"""
import argparse
import pathlib
from utils.default_video_cropper import DefaultVideoCropper

parser = argparse.ArgumentParser(
    description="Crop a video and store frames locally", add_help=False
)
parser.add_argument("-i", "--input", help="Input Video", required=True, type=str)
parser.add_argument("-w", "--width", help="Cropping width", required=True, type=int)
parser.add_argument("-h", "--height", help="Cropping height", required=True, type=int)
parser.add_argument(
    "-o", "--output", help="Folder To Store Output", required=True, type=str
)
parser.add_argument(
    "-x", "--x_position", help="Crop x position", required=False, default=(0), type=int
)
parser.add_argument(
    "-y", "--y_position", help="Crop y position", required=False, default=(0), type=int
)

args = vars(parser.parse_args())


def extract_vid_uuid(vid_path: str) -> str:
    """Extracts uuid from video path"""
    vid_path = pathlib.Path(vid_path)
    uuid = vid_path.with_suffix("").name  # Remove file extension (.mp4,.webm)
    return uuid


def main() -> None:
    """Crops an input video and saves its frames"""
    cropper = DefaultVideoCropper(
        output_dir=args["output"],
        cropping_dimensions=(args["width"], args["height"]),
        crop_position=(args["x_position"], args["y_position"]),
    )
    uuid = extract_vid_uuid(args["input"])
    cropper.process_video(args["input"], uuid)


if __name__ == "__main__":
    main()
