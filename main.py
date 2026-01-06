import os
import random
from datetime import datetime
from src.lofi_gen.music.models import MusicGenModel
from src.lofi_gen.music.pipelines import LongMusicGenerator


prompts = {
    'sleep': [
        "nepali ambient sleep music, slow bansuri, no drums, soft pads, himalayan drone, 60 bpm, peaceful",
        "deep sleep nepali lofi, gentle sarangi, minimal, spacious reverb, night ambience, meditative, ultra calm, 50 bpm",
        "calming sleep music with nepali flute, soft tanpura drone, no percussion, dreamy, slow, 55 bpm",
        "himalayan sleeping music, soft flute melody, ambient soundscape, ethnic nepali, dreamy, lullaby, 50 bpm"
    ],

    'study': [
        "nepali lofi study beats, soft tungna plucks, subtle madal rhythm, ambient textures, focus music, 85 bpm, instrumental",
        "chill nepali lofi for studying, gentle sarangi loops, soft piano chords, vinyl warmth, atmospheric, 80 bpm",
        "relaxing study music with nepali folk instruments, soft bansuri melodies, mellow synths, lo-fi texture, 75 bpm",
        "himalayan study beats, dreamy pads, subtle percussion, ethereal atmosphere, nostalgic vibe, 70 bpm"
    ],

    'relax': [
        "chill nepali lofi, soft sarangi loops, madal rhythm, lo-fi piano, vinyl warmth, himalayan nostalgia, rainy kathmandu evening, instrumental, relaxing study beats",
        "nepali lofi chill beat with soft sarangi melody, flute (bansuri) phrases, gentle madal drums, relaxing ambient atmosphere, himalayan vibes, peaceful meditation music, warm analog sound, vinyl crackle, slow tempo 70 bpm",
        "relaxing nepali lofi with soft tungna plucks, subtle madal rhythm, ambient textures, focus music, 85 bpm, instrumental",
        "calming nepali lofi with gentle bansuri melodies, soft pads, minimal percussion, dreamy atmosphere, perfect for unwinding after a long day"
    ]

}



model = MusicGenModel()
audio, sample_rate = model.generate_music(
    prompt="lofi hip hop beat with soft piano",
    duration_seconds=60
)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"lofi_{timestamp}.wav"

model.save_audio(audio, sample_rate, filename)

extender = LongMusicGenerator()

extender.generate(
    input_file=filename,
    output_file=f"seamless_{filename}",
    target_duration_mins=5
)
