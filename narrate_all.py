"""Parallel narration of all chapter texts using XTTS-v2 voice clone.

Runs N worker processes (each loads the XTTS-v2 model once, then narrates
chapters one at a time). Already-complete MP3s are skipped, so the run is
resumable: if a worker crashes, rerun the same command and it picks up.

Usage:
    python narrate_all.py \\
        --text-dir chapters_text/ \\
        --voice voice.wav \\
        --audio-dir chapters_audio/ \\
        --workers 4 --threads 4

Worker sizing rule of thumb on CPU-only hardware:
    workers * threads ≈ physical core count
    and workers * ~5 GB ≤ available RAM (each loads its own model copy)
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

os.environ.setdefault("COQUI_TOS_AGREED", "1")

MIN_MP3_BYTES = 50_000  # below this we treat the output as incomplete and re-narrate

# Per-worker globals (populated in _init_worker)
_tts = None
_split_sentences = None
_voice_path: str | None = None
_audio_dir: Path | None = None
_pause_ms: int = 250


def _init_worker(voice_path: str, audio_dir: str, threads: int, pause_ms: int) -> None:
    global _tts, _split_sentences, _voice_path, _audio_dir, _pause_ms
    _voice_path = voice_path
    _audio_dir = Path(audio_dir)
    _pause_ms = pause_ms

    import torch

    torch.set_num_threads(threads)

    # Avoid anaconda-compiler env leaking into any subprocess this worker spawns
    os.environ["PATH"] = ":".join(
        p for p in os.environ.get("PATH", "").split(":") if "anaconda3" not in p
    )

    from TTS.api import TTS

    sys.path.insert(0, str(Path(__file__).parent))
    from narrate_chapter import split_sentences

    _split_sentences = split_sentences
    _tts = TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        progress_bar=False,
    ).to("cpu")
    print(f"[worker {os.getpid()}] model ready", flush=True)


def _narrate_one(txt_path_str: str) -> tuple[str, str, float]:
    from pydub import AudioSegment

    assert _audio_dir is not None and _voice_path is not None and _tts is not None
    txt_path = Path(txt_path_str)
    out_mp3 = _audio_dir / (txt_path.stem + ".mp3")
    if out_mp3.exists() and out_mp3.stat().st_size >= MIN_MP3_BYTES:
        return (txt_path.name, "skipped", 0.0)

    t0 = time.time()
    try:
        assert _split_sentences is not None
        text = txt_path.read_text(encoding="utf-8")
        chunks = [c for c in _split_sentences(text) if c.strip()]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            wav_paths: list[Path] = []
            for i, chunk in enumerate(chunks, 1):
                wav_path = tmp / f"chunk_{i:04d}.wav"
                _tts.tts_to_file(
                    text=chunk,
                    speaker_wav=_voice_path,
                    language="en",
                    file_path=str(wav_path),
                )
                wav_paths.append(wav_path)

            combined = AudioSegment.empty()
            pause = AudioSegment.silent(duration=_pause_ms)
            for wav_path in wav_paths:
                combined += AudioSegment.from_wav(str(wav_path)) + pause

            _audio_dir.mkdir(parents=True, exist_ok=True)
            combined.export(str(out_mp3), format="mp3", bitrate="128k")

        return (txt_path.name, "ok", time.time() - t0)
    except Exception as e:
        tb = traceback.format_exc()
        return (txt_path.name, f"FAIL {type(e).__name__}: {e}\n{tb}", time.time() - t0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-dir", type=Path, required=True)
    parser.add_argument("--voice", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4,
                        help="torch threads per worker (default 4)")
    parser.add_argument("--pause-ms", type=int, default=250)
    parser.add_argument("--log", type=Path, default=Path("narration.log"))
    args = parser.parse_args()

    if not args.text_dir.is_dir():
        sys.exit(f"Text directory not found: {args.text_dir}")
    if not args.voice.exists():
        sys.exit(f"Voice sample not found: {args.voice}")
    args.audio_dir.mkdir(parents=True, exist_ok=True)

    all_chapters = sorted(str(p) for p in args.text_dir.glob("*.txt"))
    if not all_chapters:
        sys.exit(f"No .txt files in {args.text_dir}")

    pending = [
        p for p in all_chapters
        if not (
            (args.audio_dir / (Path(p).stem + ".mp3")).exists()
            and (args.audio_dir / (Path(p).stem + ".mp3")).stat().st_size >= MIN_MP3_BYTES
        )
    ]
    done = len(all_chapters) - len(pending)
    print(f"{len(all_chapters)} total chapters, {done} already done, {len(pending)} pending")
    if not pending:
        print("All chapters already narrated.")
        return

    t_start = time.time()
    with open(args.log, "a") as log:
        header = (
            f"\n=== Run started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            f"Workers: {args.workers}, threads/worker: {args.threads}\n"
            f"Pending: {len(pending)}\n"
        )
        print(header.strip())
        log.write(header)
        log.flush()

        completed = 0
        init_args = (str(args.voice), str(args.audio_dir), args.threads, args.pause_ms)
        with mp.Pool(args.workers, initializer=_init_worker, initargs=init_args) as pool:
            for name, status, elapsed in pool.imap_unordered(_narrate_one, pending):
                completed += 1
                stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                first_line = status.splitlines()[0] if status else ""
                line = (
                    f"[{stamp}] ({completed}/{len(pending)}) "
                    f"{name}: {first_line} ({elapsed:.0f}s)"
                )
                print(line, flush=True)
                log.write(line + "\n")
                if status.startswith("FAIL"):
                    log.write(status + "\n")
                log.flush()

        total = time.time() - t_start
        footer = f"=== Complete in {total / 3600:.2f}h ==="
        print(footer)
        log.write(footer + "\n")


if __name__ == "__main__":
    mp.set_start_method("spawn")  # required for PyTorch on macOS
    main()
