import math
from pathlib import Path
from moviepy import VideoFileClip, concatenate_videoclips
from moviepy.video.fx import CrossFadeIn, CrossFadeOut, Crop
from .utils import _resolve_input, _build_output_path


class VideoProcessor:
    def __init__(self):
        pass

    def remove_audio(self, output_path=None):
        input_path = _resolve_input(self.video_path)
        if not input_path.exists():
            raise FileNotFoundError(str(input_path))

        output_path = _build_output_path(input_path, "no_audio", output_path)

        with VideoFileClip(str(input_path)) as video:
            video_without_audio = video.without_audio()
            video_without_audio.write_videofile(
                str(output_path),
                codec="libx264",
                audio=False,
                logger=None,
            )

        return str(output_path)

    def crop_to_4k(self, input_path, output_path=None):
        # crop video to 3840x2160 (4k resolution). If the size is small, zoom in to fit.
        input_path = _resolve_input(input_path)
        if not input_path.exists():
            raise FileNotFoundError(str(input_path))   
        output_path = _build_output_path(input_path, "4k", output_path)
        with VideoFileClip(str(input_path)) as video:
            video_width, video_height = video.size
            target_width, target_height = 3840, 2160

            scale_factor = max(target_width / video_width, target_height / video_height)
            new_width = math.ceil(video_width * scale_factor)
            new_height = math.ceil(video_height * scale_factor)

            resized_video = video.resized(width=new_width, height=new_height)

            x1 = (new_width - target_width) // 2
            y1 = (new_height - target_height) // 2
            x2 = x1 + target_width
            y2 = y1 + target_height

            cropped_video = resized_video.with_effects([Crop(x1=x1, y1=y1, x2=x2, y2=y2)])

            cropped_video.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                logger=None,
            )
        return str(output_path)



if __name__ == "__main__":
    video_path = "/home/gnikesh/projects/lofi-gen/output_4k_50fps.mp4"
    processor = VideoProcessor()    
    # processor.remove_audio()
    # processor.crop_to_4k(video_path)
    # print("Done")
    