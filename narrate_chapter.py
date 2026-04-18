"""Narrate a single chapter with XTTS-v2 using a cloned voice.

XTTS-v2 has an effective ~400-character limit per generate call, so the text
is split into sentence-sized chunks that each stay well under that. The
per-chunk WAVs are concatenated with a small pause, then encoded to MP3.

Usage:
    python narrate_chapter.py --text chapter.txt --voice voice.wav --out chapter.mp3

Set COQUI_TOS_AGREED=1 to skip the Coqui CPML license prompt (you must still
abide by the license — non-commercial use).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from pathlib import Path

# Must be set before TTS import picks it up
os.environ.setdefault("COQUI_TOS_AGREED", "1")

# XTTS-v2 is unstable past ~250 chars; keep well under the ~400 ceiling.
MAX_CHUNK_CHARS = 240


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-sized chunks for XTTS."""
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    buf = ""
    for part in parts:
        if not part:
            continue
        if len(part) > MAX_CHUNK_CHARS:
            if buf:
                chunks.append(buf.strip())
                buf = ""
            chunks.extend(_hard_split(part))
            continue
        if len(buf) + 1 + len(part) > MAX_CHUNK_CHARS:
            chunks.append(buf.strip())
            buf = part
        else:
            buf = f"{buf} {part}".strip()
    if buf:
        chunks.append(buf.strip())
    return [c for c in chunks if c.strip()]


def _hard_split(sentence: str) -> list[str]:
    """Break an overlong sentence at commas, falling back to word groups."""
    pieces = re.split(r"(?<=,)\s+", sentence)
    out: list[str] = []
    buf = ""
    for p in pieces:
        if len(p) > MAX_CHUNK_CHARS:
            words = p.split()
            sub = ""
            for w in words:
                if len(sub) + 1 + len(w) > MAX_CHUNK_CHARS:
                    out.append(sub.strip())
                    sub = w
                else:
                    sub = f"{sub} {w}".strip()
            if sub:
                out.append(sub.strip())
            continue
        if len(buf) + 1 + len(p) > MAX_CHUNK_CHARS:
            out.append(buf.strip())
            buf = p
        else:
            buf = f"{buf} {p}".strip()
    if buf:
        out.append(buf.strip())
    return out


def narrate(chapter_path: Path, voice_path: Path, out_mp3: Path, pause_ms: int = 250) -> None:
    from TTS.api import TTS
    from pydub import AudioSegment

    text = chapter_path.read_text(encoding="utf-8")
    chunks = split_sentences(text)
    if not chunks:
        sys.exit(f"No content extracted from {chapter_path}")
    print(f"[{chapter_path.name}] {len(chunks)} chunks, "
          f"{sum(len(c) for c in chunks)} chars total")

    print("Loading XTTS-v2 model (first run downloads ~2 GB)...")
    t0 = time.time()
    tts = TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        progress_bar=True,
    ).to("cpu")
    print(f"Model loaded in {time.time() - t0:.1f}s")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        wav_paths: list[Path] = []
        for i, chunk in enumerate(chunks, 1):
            wav_path = tmp / f"chunk_{i:04d}.wav"
            t_chunk = time.time()
            tts.tts_to_file(
                text=chunk,
                speaker_wav=str(voice_path),
                language="en",
                file_path=str(wav_path),
            )
            wav_paths.append(wav_path)
            print(f"  [{i}/{len(chunks)}] {len(chunk):3d} chars "
                  f"in {time.time() - t_chunk:5.1f}s")

        print("Stitching chunks into MP3...")
        combined = AudioSegment.empty()
        pause = AudioSegment.silent(duration=pause_ms)
        for wav_path in wav_paths:
            combined += AudioSegment.from_wav(str(wav_path)) + pause

        out_mp3.parent.mkdir(parents=True, exist_ok=True)
        combined.export(str(out_mp3), format="mp3", bitrate="128k")
        size_mb = out_mp3.stat().st_size / 1024 / 1024
        print(f"Wrote {out_mp3}  ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=Path, required=True, help="Chapter .txt file")
    parser.add_argument("--voice", type=Path, required=True,
                        help="Reference voice sample (WAV, mono, ≥22050 Hz recommended)")
    parser.add_argument("--out", type=Path, required=True, help="Output MP3 path")
    parser.add_argument("--pause-ms", type=int, default=250,
                        help="Silence between chunks in milliseconds (default 250)")
    args = parser.parse_args()

    if not args.text.exists():
        sys.exit(f"Text file not found: {args.text}")
    if not args.voice.exists():
        sys.exit(f"Voice sample not found: {args.voice}")

    t0 = time.time()
    narrate(args.text, args.voice, args.out, pause_ms=args.pause_ms)
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
