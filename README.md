# clone-and-narrate

Turn any PDF with embedded chapter bookmarks into an audiobook narrated in **your own voice**, using only open-source tools and your CPU. No cloud, no API keys, no subscription.

Built with [Coqui XTTS-v2](https://github.com/coqui-ai/TTS) for voice cloning, [PyMuPDF](https://pymupdf.readthedocs.io/) for chapter extraction, and [pydub](https://github.com/jiaaro/pydub) + [ffmpeg](https://ffmpeg.org/) for audio stitching.

## What it does

1. **Extract** — reads a PDF's embedded table of contents, writes one clean `.txt` per chapter (strips footnotes, fixes hyphenated line breaks, expands Roman numerals).
2. **Clone** — takes a ~60 second recording of your voice as reference.
3. **Narrate** — generates an MP3 per chapter in your cloned voice, with optional pronunciation respellings for domain terms the model mispronounces.
4. **Parallelize** — runs multiple workers so a full book finishes overnight on a multi-core CPU.

End result: a folder of numbered MP3s, one per chapter, that drops straight into any audiobook player.

## Requirements

- macOS, Linux, or Windows (tested on Intel macOS)
- Python **3.11** (TTS 0.22.0 doesn't support 3.12+ reliably)
- `ffmpeg` on `PATH`
- ~4 GB free disk (model + dependencies), ~5 GB RAM per worker
- A PDF with an embedded table of contents (no OCR step — text-based PDFs only)

No GPU required. A GPU makes it much faster, but it runs fine on a multi-core CPU.

## Install

This stack is finicky. The pinned versions in `requirements.txt` exist for specific reasons — don't casually upgrade them.

```bash
git clone https://github.com/braincomputingsantosh/clone-and-narrate.git
cd clone-and-narrate

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Gotchas you may hit** (and why the requirements are pinned the way they are):

- **`ImportError: cannot import name 'BeamSearchScorer' from 'transformers'`** — TTS 0.22.0 was written against the older transformers generation API. Modern transformers removed `BeamSearchScorer`. Pinning to `transformers==4.40.2` is the last version that still exports it.
- **`Failed to build llvmlite` / CMake can't find LLVM** — llvmlite 0.47.0 ships only a source tarball for Python 3.11 macOS x86_64 on PyPI; compiling it requires LLVM 14 and a non-anaconda toolchain. Using `llvmlite==0.43.0` and `numba==0.60.0` skips all of this because both have prebuilt wheels.
- **`fatal error: 'stdio.h' file not found`** — if anaconda's base environment is active in your shell, its cross-compiler toolchain (`/opt/anaconda3/bin/x86_64-apple-darwin13.4.0-*`) leaks into subprocess builds. Create your venv with a non-anaconda Python (Homebrew's `python@3.11` or pyenv), not anaconda's `python3`.

If you see compile errors, the fastest fix is usually: deactivate conda, recreate the venv from a clean Python, and reinstall.

## Usage

### 1. Record your voice

60 seconds of you reading any passage, in a quiet room. Save as WAV — mono, at least 22050 Hz. On macOS, QuickTime → New Audio Recording → set Quality to **High** (44.1 kHz) → record → export. Convert to WAV:

```bash
ffmpeg -i voice_sample.m4a -ac 1 -ar 22050 voice.wav
```

Tips:
- Consistent distance from the mic (6–12 inches).
- No background music, fan noise, or echo.
- Varied prosody helps — read a passage with statements and questions, not a monotone list.

### 2. Extract chapters from the PDF

```bash
python extract_chapters.py --pdf your-book.pdf --out-dir chapters_text/
```

If the PDF's TOC uses different labels (e.g. "Part I", "Section 1") adjust the filter:

```bash
python extract_chapters.py --pdf your-book.pdf --chapter-regex "^(chapter|part)\b"
```

To fix pronunciations of domain-specific names, copy `pronunciations.py`, fill in your own entries, and pass it in:

```bash
python extract_chapters.py --pdf your-book.pdf --pronunciations pronunciations.py
```

### 3. Test on one chapter first

Always do this before the full run. The first invocation downloads the ~2 GB XTTS-v2 model, and it lets you catch pronunciation or voice-quality issues early.

```bash
python narrate_chapter.py \
    --text chapters_text/01_Chapter_1.txt \
    --voice voice.wav \
    --out chapters_audio/01_Chapter_1.mp3
```

Listen to the result. If names sound wrong, add entries to `pronunciations.py`, re-run `extract_chapters.py`, and re-narrate the chapter.

### 4. Narrate the full book in parallel

```bash
python narrate_all.py \
    --text-dir chapters_text/ \
    --voice voice.wav \
    --audio-dir chapters_audio/ \
    --workers 4 --threads 4
```

**Worker sizing**: `workers × threads` should roughly equal your physical CPU core count. Each worker also holds its own copy of the model in RAM (~5 GB), so `workers × 5 GB ≤ available RAM`. On a 16-core / 64 GB box, `--workers 4 --threads 4` is a safe default. On a 4-core laptop, try `--workers 2 --threads 2`.

**Resumability**: any chapter with an MP3 ≥50 KB is skipped. If a worker dies mid-run, just re-run the same command and it picks up from where it stopped.

**Keeping your machine awake** (macOS):

```bash
caffeinate -is python narrate_all.py ...
```

## Performance notes

Numbers from an Intel Xeon W (16 cores, no GPU), narrating a ~100k-word book:
- **Single CPU, sequential**: ~32 hours
- **4 workers, 4 threads each, parallel**: ~14 hours (≈2.3× speedup)
- Parallel speedup is sub-linear because XTTS inference is memory-bandwidth-bound at scale, not compute-bound.

A modern GPU (even a consumer RTX card) would finish the same work in 1–2 hours. If you have one, set `.to("cuda")` in the TTS construction. Apple Silicon (`mps`) works but is unstable for long runs.

## Pronunciation dictionary

TTS models trained on English apply English phonetic rules to every word. Names and terms from other languages get mangled. The fix is to respell them using English letter patterns the model already interprets correctly.

Respelling heuristics:
- `aa` → long /aː/ as in "father"
- `oo` → long /uː/ as in "boot"
- `ee` → long /iː/ as in "see"
- `u` → short /ʌ/ as in "cup"

So `"Bhagavad"` respelled as `"Bhugavud"` often sounds closer to the intended pronunciation than letting the model guess from the original spelling.

See `pronunciations.py` for the template.

## What this does NOT do

- **OCR** — only PDFs with extractable text. Scanned PDFs need a tool like `ocrmypdf` first.
- **Automatic speaker adaptation** — the clone is only as good as your reference sample. A noisy or short sample gives a noisy or weak clone.
- **Multi-voice narration** — one voice for the whole book. Dialogue will still sound like you reading both sides.
- **ID3 tagging / M4B packaging** — the output is plain MP3s. Tag them with `mutagen` or bundle into an M4B with `ffmpeg` if you want audiobook-player metadata.

## Licenses and ethics

- **This code**: MIT (see `LICENSE`).
- **XTTS-v2 model**: [Coqui Public Model License (CPML)](https://coqui.ai/cpml.txt), **non-commercial use only**. You must set `COQUI_TOS_AGREED=1` or accept the prompt on first model load. Don't use this for commercial audiobook production without a commercial TTS engine.
- **Voice cloning ethics**: only clone voices you own or have explicit, informed permission to use. Cloning someone else's voice without consent is a serious harm, and in some jurisdictions a crime. Keep your reference samples private.
- **Source PDFs**: respect the copyright of whatever you're converting. Personal reading of content you legally own is typically fine; redistribution of the generated audio likely is not.

## Credits

- [Coqui TTS](https://github.com/coqui-ai/TTS) — the XTTS-v2 model and generation pipeline.
- [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) — the cleanest PDF text and TOC extraction in Python.
- [pydub](https://github.com/jiaaro/pydub) — audio stitching on top of ffmpeg.
