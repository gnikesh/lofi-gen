
from pathlib import Path
from moviepy import VideoFileClip, concatenate_videoclips
from moviepy.video.fx import CrossFadeIn, CrossFadeOut, Crop
from ..utils import _resolve_input, _build_output_path
from ..models import LTX2Model

class LongVideoGenerator(LTX2Model):
    def __init__(self, model_name: str = "LTX-2"):
        self.model_name = model_name
        super().__init__()

    def generate(self):
        

    def loop_video(self, duration_in_sec, input_path, output_path=None, crossfade_duration=1):
        input_path = _resolve_input(input_path)
        if not input_path.exists():
            raise FileNotFoundError(str(input_path))
        output_path = _build_output_path(input_path, f"looped_{duration_in_sec}s", output_path)

        with VideoFileClip(str(input_path)) as video:
            loop_duration = video.duration - crossfade_duration
            loops = int(duration_in_sec // loop_duration) + 1

            clips = []
            for i in range(loops):
                clip = VideoFileClip(str(input_path))
                if i > 0:
                    clip = clip.with_effects([CrossFadeIn(crossfade_duration)])
                if i < loops - 1:
                    clip = clip.with_effects([CrossFadeOut(crossfade_duration)])
                clips.append(clip)

            final_clip = concatenate_videoclips(clips, method="compose", padding=-crossfade_duration)
            final_clip = final_clip.subclipped(0, duration_in_sec)
            final_clip.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                logger=None,
            )

            for clip in clips:
                clip.close()

        return str(output_path)

