import os
from .models.musicgen import musicgen

class MusicGenerator:
    def __init__(self, config):
        self.config = config

    def generate_music(self, prompt):
        return musicgen(prompt)