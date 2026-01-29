import random
from pathlib import Path
from moviepy import VideoFileClip, AudioFileClip


def merge_audio_video(audio_path, video_path, output_path=None):
    """
    Merge audio and video files into a single video file.

    Args:
        audio_path: Path to the audio file.
        video_path: Path to the video file.
        output_path: Path to save the merged file. If None, saves as
                     '<video_name>_merged.mp4' in the same directory.

    Returns:
        Path to the saved merged video.
    """
    audio_path = Path(audio_path).expanduser().resolve()
    video_path = Path(video_path).expanduser().resolve()

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_path is None:
        output_path = video_path.with_name(f"{video_path.stem}_merged.mp4")
    else:
        output_path = Path(output_path).expanduser().resolve()

    with VideoFileClip(str(video_path)) as video, AudioFileClip(str(audio_path)) as audio:
        final_duration = min(video.duration, audio.duration)

        trimmed_video = video.subclipped(0, final_duration)
        trimmed_audio = audio.subclipped(0, final_duration)

        final_video = trimmed_video.with_audio(trimmed_audio)

        final_video.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            logger=None,
        )

    return str(output_path)


def take_screenshot(video_path, time_in_sec=None, output_path=None):
    """
    Take a screenshot from a video at a specific time.

    Args:
        video_path: Path to the video file.
        time_in_sec: Time in seconds to capture the screenshot.
                     If None, randomly selects a time between 20-80% of the video duration.
        output_path: Path to save the screenshot. If None, saves as
                     '<video_name>_screenshot.png' in the same directory.

    Returns:
        Path to the saved screenshot.
    """
    video_path = Path(video_path).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    with VideoFileClip(str(video_path)) as video:
        duration = video.duration

        if time_in_sec is None:
            time_in_sec = random.uniform(0.2 * duration, 0.8 * duration)

        if time_in_sec < 0 or time_in_sec > duration:
            raise ValueError(f"time_in_sec ({time_in_sec}) must be between 0 and {duration}")

        if output_path is None:
            output_path = video_path.with_name(f"{video_path.stem}_screenshot.png")
        else:
            output_path = Path(output_path).expanduser().resolve()

        video.save_frame(str(output_path), t=time_in_sec)

    return str(output_path)



if __name__ == "__main__":
    video_file = "/home/gnikesh/projects/lofi-gen/output_4k_50fps_4k.mp4"
    # screenshot_path = take_screenshot(video_file)
    # print(f"Screenshot saved at: {screenshot_path}")
    audio_file = '/home/gnikesh/projects/lofi-gen/relaxing_lofi_beat.wav'
    video_file = '/home/gnikesh/projects/lofi-gen/output_4k_50fps.mp4'

    merged_path = merge_audio_video(audio_file, video_file)
    print(f"Merged video saved at: {merged_path}")
