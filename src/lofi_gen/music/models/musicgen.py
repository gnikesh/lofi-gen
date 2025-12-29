from .base import BaseMusicGenModel
from typing import Dict, Any, Optional
from transformers import pipeline



class MusicGenModel(BaseMusicGenModel):
    """MusicGen model wrapper for text-to-music generation.
    """
    
    def __init__(self, 
                 model_name: str = "facebook/musicgen-large",
                 device: Optional[int] = 0,
                 **kwargs):
        super().__init__(model_name=model_name, config=kwargs)
        self.device = device
        self.load_model()

    def load_model(self):
        if self._model is None:
            self._model = pipeline("text-to-audio", self.model_name, device=self.device)

    def generate_music(self, prompt: str, duration: Optional[int] = 30, **kwargs) -> Any:
        music = self._model(prompt, forward_params={"do_sample": True})
        return music
    