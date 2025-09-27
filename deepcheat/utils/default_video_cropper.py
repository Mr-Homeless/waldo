"""
Contains the default video cropper class.
This class can be used to convert a video into cropped frames.
"""
from typing import Any, Tuple
import cv2
from utils.common import ensure_dir_exists


class DefaultVideoCropper:
    # pylint: disable=too-few-public-methods
    # pylint: disable=no-member
    """
    Creates a video cropper object which can crop and save
    the frames of a video to a specified output directory
    """
    def __init__(
        self,
        output_dir: str,
        cropping_dimensions: Tuple[int, int],
        crop_position: Tuple[int, int] = (0, 0),
    ) -> None:
        self._output_dir = output_dir
        ensure_dir_exists(self._output_dir)
        self._crop_width, self._crop_height = cropping_dimensions
        self._crop_position_x, self._crop_position_y = crop_position

    def _crop_frame(self, video_frame: Any) -> Any:
        """Crops a given frame"""
        video_frame = video_frame[
            self._crop_position_y : self._crop_position_y + self._crop_height,
            self._crop_position_x : self._crop_position_x + self._crop_width,
        ]
        return video_frame

    def process_video(self, vid_path: str, video_uuid: str) -> None:
        """Crops an input video and saves its frames"""
        vidcap = cv2.VideoCapture(vid_path)
        count = 0
        was_read, frame = vidcap.read()
        while was_read:
            frame = self._crop_frame(frame)
            cv2.imwrite(f"{self._output_dir}/vid_{video_uuid}_frame_{count}.jpg", frame)
            was_read, frame = vidcap.read()
            count += 1
