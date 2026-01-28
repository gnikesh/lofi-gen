from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
from ltx_pipelines.utils.helpers import get_device

from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

device = get_device()

from lofi_gen.video.models.base import BaseVideoGen
from .constants import (
    NUM_INFERENCE_STEPS,
    WIDTH,
    HEIGHT,
    NUM_FRAMES,
    FRAME_RATE,
    SEED,
    CFG_GUIDANCE_SCALE,
    DEFAULT_NEGATIVE_PROMPT
)


class LTX2Model(BaseVideoGen):
    def __init__(self, model_name: str = "LTX2", config: dict = None):
        super().__init__(model_name=model_name, config=config)

    def generate(self, prompt, **kwargs):
        pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path=kwargs.get("checkpoint_path", None),
            distilled_lora=kwargs.get("distilled_lora", None),
            spatial_upsampler_path=kwargs.get("spatial_upsampler_path", None),
            gemma_root=kwargs.get("gemma_root", None),
            loras=kwargs.get("loras", []),
        )

        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(kwargs.get('num_frames', NUM_FRAMES), tiling_config)
        video, audio = pipeline(
            prompt=prompt,
            negative_prompt=kwargs.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
            seed=kwargs.get("seed", SEED),
            height=kwargs.get("height", HEIGHT),
            width=kwargs.get("width", WIDTH),
            num_frames=kwargs.get("num_frames", NUM_FRAMES),
            frame_rate=kwargs.get("frame_rate", FRAME_RATE),
            num_inference_steps=kwargs.get("num_inference_steps", NUM_INFERENCE_STEPS),
            cfg_guidance_scale=kwargs.get("cfg_guidance_scale", CFG_GUIDANCE_SCALE),
            images=kwargs.get("images", None),
            tiling_config=tiling_config,
        )

        encode_video(
            video=video,
            fps=kwargs.get("frame_rate", FRAME_RATE),
            audio=audio,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            output_path=kwargs.get("output_path", "output_ltx2.mp4"),
            video_chunks_number=video_chunks_number,
        )

