import os
import subprocess
import tempfile
from pathlib import Path


def get_video_duration(video_path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def merge_time_ranges(ranges):
    if not ranges:
        return []

    ranges = sorted(ranges, key=lambda x: x[0])
    merged = [ranges[0]]

    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def write_summary_video(video_path, ranges, output_path):
    video_path = str(video_path)
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if not ranges:
        return False

    # Platformdan bagimsiz gecici klasor (Windows'ta /tmp yok).
    temp_dir = Path(tempfile.mkdtemp(prefix="cliptsa_summary_"))

    temp_files = []

    try:
        for idx, (start, end) in enumerate(ranges):
            temp_file = temp_dir / f"segment_{idx:04d}.mp4"
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}",
                "-to", f"{end:.3f}",
                "-i", video_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-c:a", "aac",
                str(temp_file),
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            temp_files.append(temp_file)

        concat_file = temp_dir / "concat.txt"
        with open(concat_file, "w") as f:
            for file in temp_files:
                # ffmpeg concat listesi ileri-bolu ister; Windows yollarini cevir.
                safe = str(file).replace("\\", "/")
                f.write(f"file '{safe}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            output_path,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    finally:
        for file in temp_files:
            try:
                os.remove(file)
            except OSError:
                pass
        try:
            os.remove(temp_dir / "concat.txt")
        except OSError:
            pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass

    return True
