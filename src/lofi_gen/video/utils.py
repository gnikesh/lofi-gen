from pathlib import Path

def _resolve_input(video_path):
    return Path(video_path).expanduser().resolve()

def _build_output_path(input_path: Path, suffix: str, output_path: str | None):
    if output_path:
        return Path(output_path).expanduser().resolve()
    return input_path.with_name(f"{input_path.stem}_{suffix}{input_path.suffix}")
