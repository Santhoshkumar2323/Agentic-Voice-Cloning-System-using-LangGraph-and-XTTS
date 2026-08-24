# 🎙️ Agentic Voice Cloning System using LangGraph and XTTS

A text-to-speech (TTS) synthesis pipeline that leverages a multi-agent orchestration loop to optimize raw user text inputs before executing localized voice cloning.


### 🔊 Audio Pipeline Documentation Walkthrough

Listen to the factual pipeline voice output overview (Qwen-2.5 & XTTS-v2):

👉 **[Click Here to Listen to the Audio Demo](demo-audio.mp3)**




Because this pipeline depends on dedicated CUDA hardware to execute deep learning models, a live working instance is hosted on shared cloud infrastructure:

👉 **[Click Here for Live Application] (https://huggingface.co/spaces/santhosh2323/voice-cloner)**

---

## 🛠️ The Architecture: How It Works

This project implements a **decoupled processing architecture** split into three distinct pipeline:

```text
[ Raw Audio Input ] ──> FFmpeg (Denoise & -20 LUFS Normalize) ──> [ Clean Voice Reference ]
                                                                             │
[ Raw Text Prompt ] ──> LangGraph State Loop (Director <─> Evaluator) ───────┴─> [ XTTS-v2 Engine ] ──> [ Final WAV Output ]
```

### 1: Neural Digital Signal Processing (DSP)
Before your reference voice clip ever touches the cloning model, the script safeguards audio fidelity through automated command-line workflows:
* **Background Noise Removal:** The system calls an internal `ffmpeg` process graph utilizing the `arnndn` filter—powered by a localized Recurrent Neural Network weights file (`std.rnnn`)—to actively isolate and strip room ambient hiss and microphone static.
* **Loudness Normalization:** Rather than applying un-calibrated digital gain, the pipeline uses the `pyloudnorm` library to read integrated loudness over time, standardizing all user inputs to a strict broadcasting target of **-20.0 LUFS** with a safe peak cap at -1.0 dB to prevent clipping distortions.
* **Topographic Silence Trimming:** It leverages `librosa.effects.trim` with a `top_db=30` gate to discard dead air at the beginning and end of the audio track, preserving VRAM timeline execution space.

### 2: The Agentic Orchestration Cycle (LangGraph)
Raw textual prompt feeds are highly prone to breaking open-source TTS models (e.g., characters like `%` or raw digits like `12.5` cause speech synthesizers to stumble or spell literally). To solve this autonomously, the system initiates a structured **StateGraph execution loop**:

1. **The Director Node (`Qwen2.5-3B-Instruct`):** Takes raw prompt text and rewrites it into a completely readable spoken script (e.g., converting `75%` to `"seventy-five percent"` and adding ellipses `...` for human pacing breathe-pauses).
2. **The Evaluator Node:** Runs dual-layered screening. First, a strict deterministic validation layer checks text strings against custom Regex rules. Second, a cognitive LLM gate evaluates textual flow.
3. **The Conditional Router:** If a validation rule fails, the graph state logs the explicit failure context, loops backward into the Director node, and appends the structural error warning as a fresh hint to force a self-corrected generation. It safely exits the loop to feed the downstream voice engine only upon achieving a clean `PASS` verdict or exhausting `max_retries`.

### 3: Hardware-Bounded Audio Ingestion (`XTTS-v2`)
To allow local inference generation inside constrained hardware container boundaries without generating severe Out-Of-Memory (OOM) memory faults:
* **Micro-Chunking Matrix:** The script automatically tokenizes and segments long finalized script texts into highly controlled character chunks (`max_chars=180`) before shipping arrays to the graphics card.
* **Deterministic Assembly:** Every mini-chunk is individually synthesized using localized `XTTS-v2` multi-dataset weight models, padded with explicit `pydub.AudioSegment` silence variables (250ms phrase-pauses, 600ms paragraph-pauses) to guarantee natural conversational cadences, and exported as a clean final `.wav` track.
* **VRAM Aggressive Collection:** To protect the runtime server lifecycle, every localized generation loop executes manual pointer teardowns, running Python's native `gc.collect()` and forcing `torch.cuda.empty_cache()` to maintain clean GPU cache pools.

---

## Local Configuration Guide

1. Clone the project file structure:
   ```bash
   git clone https://github.com
   cd YOUR_REPO_NAME
   ```

2. Pin down requirements to your virtual environment:
   ```bash
   pip install -r requirements.txt
   ```

3. Initialize the application server locally:
   ```bash
   python app.py
   ```
   *Note: Upon terminal initialization, the system will verify system CUDA availability, allocate model dependencies, and open an active local server link (`http://127.0.0.1:7860`) directly inside your web browser.*
