"""
Basic usage example for lofi-gen package.

This example demonstrates how to use the package after installing it with pip.
"""

from lofi_gen.music.models import MusicGenModel
from lofi_gen.music.pipelines import LongMusicGenerator
from datetime import datetime


def example_basic_generation():
    """Generate a simple lofi beat."""
    print("Example 1: Basic Music Generation")
    print("-" * 50)

    # Initialize the model
    model = MusicGenModel(model_size="small")  # Use "large" for better quality

    # Generate music
    audio, sample_rate = model.generate_music(
        prompt="lofi hip hop beat with soft piano, chill vibes",
        duration_seconds=30
    )

    # Save the audio
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"lofi_basic_{timestamp}.wav"
    model.save_audio(audio, sample_rate, filename)

    print(f"✓ Generated and saved to: {filename}")
    print()


def example_nepali_lofi():
    """Generate Nepali-inspired lofi music."""
    print("Example 2: Nepali Lofi Generation")
    print("-" * 50)

    model = MusicGenModel(model_size="medium")

    prompts = [
        "chill nepali lofi, soft sarangi loops, madal rhythm, lo-fi piano, vinyl warmth",
        "nepali ambient sleep music, slow bansuri, soft pads, himalayan drone, peaceful",
        "nepali lofi study beats, soft tungna plucks, subtle madal rhythm, focus music",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"Generating track {i}/3: {prompt[:50]}...")

        audio, sample_rate = model.generate_music(
            prompt=prompt,
            duration_seconds=45,
            guidance_scale=3.0,
            temperature=1.0
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nepali_lofi_{i}_{timestamp}.wav"
        model.save_audio(audio, sample_rate, filename)

        print(f"✓ Saved: {filename}")

    print("\n✓ All tracks generated!")
    print()


def example_long_form_music():
    """Generate and extend music to create a long seamless loop."""
    print("Example 3: Long-Form Music Generation")
    print("-" * 50)

    # Step 1: Generate initial clip
    print("Step 1: Generating base clip (60 seconds)...")
    model = MusicGenModel(model_size="medium")

    audio, sample_rate = model.generate_music(
        prompt="relaxing nepali lofi with bansuri flute, gentle madal, ambient textures",
        duration_seconds=60
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"base_clip_{timestamp}.wav"
    model.save_audio(audio, sample_rate, base_filename)
    print(f"✓ Base clip saved: {base_filename}")

    # Step 2: Extend to long form
    print("\nStep 2: Extending to 10-minute seamless loop...")
    extender = LongMusicGenerator(
        crossfade_sec=3.0,  # 3-second crossfade
        variation=True      # Add subtle variations
    )

    extended_filename = f"extended_lofi_{timestamp}.wav"
    extender.generate(
        input_file=base_filename,
        output_file=extended_filename,
        target_duration_mins=10,  # 10 minutes
        verbose=True
    )

    print(f"\n✓ Extended music saved: {extended_filename}")
    print()


def example_custom_parameters():
    """Demonstrate custom generation parameters."""
    print("Example 4: Custom Generation Parameters")
    print("-" * 50)

    model = MusicGenModel(model_size="small")

    # High guidance - more adherent to prompt
    print("Generating with high guidance (more adherent to prompt)...")
    audio_high, sr = model.generate_music(
        prompt="aggressive drum and bass, fast tempo, heavy bass",
        duration_seconds=20,
        guidance_scale=5.0,  # Higher = more adherent
        temperature=0.8      # Lower = less random
    )
    model.save_audio(audio_high, sr, "high_guidance.wav")
    print("✓ Saved: high_guidance.wav")

    # Low guidance - more creative
    print("\nGenerating with low guidance (more creative)...")
    audio_low, sr = model.generate_music(
        prompt="experimental ambient soundscape",
        duration_seconds=20,
        guidance_scale=1.5,  # Lower = more creative
        temperature=1.5      # Higher = more random
    )
    model.save_audio(audio_low, sr, "low_guidance.wav")
    print("✓ Saved: low_guidance.wav")
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("lofi-gen: Usage Examples")
    print("=" * 50)
    print()

    # Run examples
    # Uncomment the ones you want to run:

    example_basic_generation()
    # example_nepali_lofi()
    # example_long_form_music()
    # example_custom_parameters()

    print("=" * 50)
    print("All examples completed!")
    print("=" * 50)
