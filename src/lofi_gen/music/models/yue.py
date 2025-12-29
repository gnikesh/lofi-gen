#!/usr/bin/env python3
"""
Lofi Nepali Music Generator with Vocals using YuE
Generates complete songs with singing from lyrics using the YuE model.

YuE GitHub: https://github.com/multimodal-art-projection/YuE
"""

import os
import subprocess
import sys
from pathlib import Path


def setup_yue():
    """Clone and setup YuE repository if not present."""
    if not Path("YuE").exists():
        print("Cloning YuE repository...")
        subprocess.run([
            "git", "clone", 
            "https://github.com/multimodal-art-projection/YuE.git"
        ], check=True)
    
    # Install dependencies
    print("Installing dependencies...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "torch", "transformers", "accelerate", "flash-attn",
        "einops", "scipy", "numpy", "--quiet"
    ], check=True)


def create_genre_prompt(output_path: str = "genre.txt"):
    """
    Create genre prompt file for lofi Nepali style.
    
    Format: [Genre] genre instrument mood gender timbre
    """
    genre_prompt = """[Genre] lofi hip hop acoustic guitar female soft vocal warm vocal mellow dreamy ambient nepali folk fusion slow tempo"""
    
    with open(output_path, "w") as f:
        f.write(genre_prompt)
    
    print(f"Genre prompt saved to: {output_path}")
    return output_path


def create_lyrics_prompt(
    lyrics: str = None, 
    output_path: str = "lyrics.txt"
):
    """
    Create lyrics prompt file with structure labels.
    
    Structure labels: [intro], [verse], [chorus], [bridge], [outro]
    Each section separated by double newlines.
    """
    if lyrics is None:
        # Default Nepali lofi lyrics (romanized)
        # Note: YuE officially supports EN, ZH, JA, KO
        # Nepali may work but with varying quality
        lyrics = """[verse]
Himalko chhahaari ma
Sapana haru udchan
Baadal sanga khelda
Mann shanta hunchha

[chorus]
Yo raatko shanti ma
Tara haru chamkinchan
Samjhana ko dhun ma
Dil bhari runchha

[verse]
Purano galli ma hidda
Yaad aauchan ti din
Gham ra jhari sangai
Biteka ti chin

[chorus]
Yo raatko shanti ma
Tara haru chamkinchan
Samjhana ko dhun ma
Dil bhari runchha

[outro]
La la la la la
Hmm hmm hmm"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(lyrics)
    
    print(f"Lyrics saved to: {output_path}")
    return output_path


def generate_with_yue(
    genre_file: str,
    lyrics_file: str,
    output_dir: str = "./output",
    num_segments: int = 2,
    stage2_batch_size: int = 4,
    max_new_tokens: int = 3000,
    cuda_idx: int = 0,
    use_icl: bool = False,
    audio_prompt_path: str = None,
):
    """
    Generate music using YuE inference script.
    
    Args:
        genre_file: Path to genre prompt file
        lyrics_file: Path to lyrics file
        output_dir: Output directory for generated audio
        num_segments: Number of lyric sections to generate
        stage2_batch_size: Batch size for stage 2 (adjust based on VRAM)
        max_new_tokens: Max tokens per segment (~3000 = 30s)
        cuda_idx: GPU index to use
        use_icl: Whether to use in-context learning with audio prompt
        audio_prompt_path: Path to reference audio for style transfer
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, "YuE/inference/infer.py",
        "--stage1_model", "m-a-p/YuE-s1-7B-anneal-en-cot",
        "--stage2_model", "m-a-p/YuE-s2-1B-general",
        "--genre_txt", genre_file,
        "--lyrics_txt", lyrics_file,
        "--run_n_segments", str(num_segments),
        "--stage2_batch_size", str(stage2_batch_size),
        "--output_dir", output_dir,
        "--cuda_idx", str(cuda_idx),
        "--max_new_tokens", str(max_new_tokens),
        "--repetition_penalty", "1.1",
    ]
    
    # Add ICL (in-context learning) if using audio prompt
    if use_icl and audio_prompt_path:
        cmd.extend([
            "--use_audio_prompt",
            "--audio_prompt_path", audio_prompt_path,
            "--prompt_start_time", "0",
            "--prompt_end_time", "30",
        ])
    
    print("\n" + "=" * 60)
    print("Starting YuE generation...")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60 + "\n")
    
    subprocess.run(cmd, check=True)
    
    print(f"\nGeneration complete! Check {output_dir} for output files.")


def generate_with_transformers(
    genre_prompt: str,
    lyrics: str,
    output_path: str = "lofi_nepali_yue.wav",
):
    """
    Alternative: Generate using HuggingFace Transformers directly.
    This is a simplified version - the full inference script is recommended.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Installing required packages...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch", "transformers", "accelerate"
        ], check=True)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Loading YuE Stage-1 model (this may take a while)...")
    
    # Note: This is a simplified approach
    # The full YuE pipeline requires both Stage-1 and Stage-2 models
    # plus the audio tokenizer/decoder
    
    model_name = "m-a-p/YuE-s1-7B-anneal-en-cot"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Combine genre and lyrics
    prompt = f"{genre_prompt}\n\n{lyrics}"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=3000,
            temperature=0.93,
            top_p=0.95,
            repetition_penalty=1.1,
            do_sample=True,
        )
    
    # Note: This gives you audio tokens, not audio
    # You need the Stage-2 model and vocoder to convert to audio
    print("Generated tokens. Full audio conversion requires Stage-2 + vocoder.")
    
    return outputs


# ============================================================
# EXAMPLE LYRICS TEMPLATES
# ============================================================

NEPALI_LOFI_LYRICS_EXAMPLES = {
    "romantic": """[verse]
Timi bina yo dil ritto
Yaadai yaad timro
Ratbhari nindra chaina
Sapana ma timi chau

[chorus]
Timi ho mero maya
Timi ho mero sansar
Sadhai sangai rahaula
Yo prem amara

[verse]
Aakaash ma tara haru
Timrai naam lekhi
Chandra ma timro photo
Sadhai herchhu ma rakhi

[chorus]
Timi ho mero maya
Timi ho mero sansar
Sadhai sangai rahaula
Yo prem amara""",

    "nostalgic": """[verse]
Purano ghar tyo pahaad ma
Aamako maya yaad aayo
Bachpan ko ti din haru
Mann bhitra runchha aaja

[chorus]
Ghar jaane bato ma
Yaad haru sangai hidchan
Pahad ra kholaa sangai
Dil bhari bhidchan

[verse]
Dhulo sukaile bato
Harek paila yaad chha
Saathi haru ko haaso
Aaja pani sunin chha

[outro]
La la la la la
Hmm hmm hmm""",

    "peaceful": """[verse]
Bihanako surya udyo
Pahad ma ujyaalo chha
Chara haru geet gaauchan
Yo din ramro chha

[chorus]
Shanti shanti yo mann ma
Prakriti ko sangit
Himalko hawaa sangai
Aatma shanta bhit

[bridge]
Om mani padme hum
Shanti milosh sabai lai
Yo sansar ramro hos
Prem failosh sadhai

[outro]
Hmm hmm hmm hmm
La la la la la""",

    "english_nepali_mix": """[verse]
Walking through the mountain trails
Himalaya calling my name
Feeling peace in every breath
Nothing here is the same

[chorus]
Yo mann khushi chha aaja
Heart is feeling so free
Prakriti sangai nachda
This is where I want to be

[verse]
Morning sun on temple bells
Ancient wisdom in the air
Nepali soul and western beats
A fusion beyond compare

[chorus]
Yo mann khushi chha aaja
Heart is feeling so free
Prakriti sangai nachda
This is where I want to be"""
}


def main():
    """Main function to run the lofi Nepali music generator with YuE."""
    print("=" * 60)
    print("  Lofi Nepali Music Generator with Vocals")
    print("  Using YuE (乐) Foundation Model")
    print("=" * 60)
    print()
    print("NOTE: YuE officially supports English, Chinese, Japanese, Korean.")
    print("Nepali lyrics may work but with varying pronunciation quality.")
    print("Consider using romanized Nepali or English-Nepali mix for best results.")
    print()
    
    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {gpu_mem:.1f} GB")
            if gpu_mem < 24:
                print("WARNING: Less than 24GB VRAM. May run into OOM issues.")
        else:
            print("WARNING: No GPU detected. YuE requires GPU for generation.")
            return
    except ImportError:
        print("PyTorch not installed. Installing...")
    
    # Setup
    # print("\n--- Setting up YuE ---")
    # setup_yue()
    
    # Create prompts
    print("\n--- Creating prompts ---")
    
    # Use English-Nepali mix for better pronunciation
    lyrics = NEPALI_LOFI_LYRICS_EXAMPLES["english_nepali_mix"]
    
    genre_file = create_genre_prompt("genre.txt")
    lyrics_file = create_lyrics_prompt(lyrics, "lyrics.txt")
    
    print(f"\nGenre: lofi hip hop, nepali folk fusion, female soft vocal")
    print(f"Lyrics:\n{lyrics}")
    
    # Generate
    print("\n--- Generating music (this will take several minutes) ---")
    generate_with_yue(
        genre_file=genre_file,
        lyrics_file=lyrics_file,
        output_dir="./output_lofi_nepali",
        num_segments=4,  # 4 sections = ~2 minutes
        stage2_batch_size=4,  # Reduce if OOM
        max_new_tokens=3000,  # ~30s per segment
    )
    
    print("\n" + "=" * 60)
    print("  Generation complete!")
    print("  Check ./output_lofi_nepali/ for your song")
    print("=" * 60)


if __name__ == "__main__":
    main()