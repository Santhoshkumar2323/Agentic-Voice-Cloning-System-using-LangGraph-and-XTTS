import os
os.environ["COQUI_TOS_AGREED"] = "1"

import re
import gc
import subprocess
from datetime import datetime
from typing import TypedDict
import torch
import gradio as gr
import spaces
import transformers.pytorch_utils

def isin_mps_friendly(*args, **kwargs):
    if hasattr(torch, 'isin') and 'elements' in kwargs and 'test_elements' in kwargs:
        return torch.isin(kwargs['elements'], torch.tensor(kwargs['test_elements']).to(kwargs['elements'].device))
    return torch.tensor([False])

transformers.pytorch_utils.isin_mps_friendly = isin_mps_friendly

import transformers
if not hasattr(transformers, 'BeamSearchScorer'):
    try:
        from transformers.generation.beam_search import BeamSearchScorer
        transformers.BeamSearchScorer = BeamSearchScorer
    except ImportError:
        class BeamSearchScorer:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("BeamSearchScorer is not available in this transformers version")
        transformers.BeamSearchScorer = BeamSearchScorer

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

print("Loading Qwen2.5-3B-Instruct...")
from transformers import AutoModelForCausalLM, AutoTokenizer
QWEN_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
qwen_model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)
print("✅ Qwen loaded")

print("Loading XTTS-v2...")
from TTS.api import TTS
tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
print("✅ XTTS-v2 loaded")

CLONE_PARAMS = dict(
    temperature=0.7, top_p=0.85, repetition_penalty=2.0,
    length_penalty=1.0, speed=1.0, gpt_cond_len=30,
    gpt_cond_chunk_len=4, enable_text_splitting=False
)

DIRECTOR_SYSTEM_PROMPT = (
    "You are an expert audio script director. Rewrite the user's text so it is perfectly readable by a Text-to-Speech voice clone. "
    "This text can be about ANY topic — finance, science, tutorials, stories, business, etc. Apply these rules generally:\n"
    "1. Spell out numbers, decimals, and percentages into plain text words (e.g. '17.4' -> 'seventeen point four', '67%' -> 'sixty-seven percent').\n"
    "2. EXCEPTION: for pixel dimensions (e.g. '224x224'), coordinates, or mathematical formulas/equations, "
    "describe them in natural spoken language instead of spelling out every symbol.\n"
    "3. Unpack all technical abbreviations, file types, acronyms, or code short-hands into clean spoken words on first mention.\n"
    "4. Strip out ALL brackets [ ], markdown symbols (#, *, **), dashes used as bullets, and the tilde (~) symbol — rewrite what they contained as plain spoken words instead of deleting the meaning.\n"
    "5. Make the tone conversational and engaging. Inject natural human pacing punctuation like ellipses (...) or commas.\n"
    "6. Respond ONLY with the final cleaned script text itself. Never mention corrections, previous attempts, fixes, or the rewriting process."
)

EVALUATOR_SYSTEM_PROMPT = (
    "You are a strict quality checker for TTS scripts, covering any topic. Check the text against these rules:\n"
    "1. No raw digits, decimals, or percentages left unspelled in normal sentences.\n"
    "2. Pixel dimensions, coordinates, and math formulas ARE ALLOWED to be described in natural technical language.\n"
    "3. No literal brackets, markdown symbols, or dash-bullets.\n"
    "4. No meta-commentary about corrections, fixes, or previous attempts anywhere in the text.\n"
    "Respond in EXACTLY this format, nothing else:\n"
    "VERDICT: PASS or FAIL\n"
    "REASON: <short explanation, or 'none' if PASS>"
)

def deterministic_check(text):
    issues = []
    if re.search(r'\d+\s*%', text):
        issues.append("leftover '%' symbol found")
    if re.search(r'\[[^\]]*\]', text):
        issues.append("leftover square brackets found")
    if re.search(r'^\s*[-*•]\s+', text, re.MULTILINE):
        issues.append("leftover markdown bullet points found")
    if re.search(r'^\s*#+\s', text, re.MULTILINE):
        issues.append("leftover markdown headers found")
    if re.search(r'~', text):
        issues.append("leftover '~' symbol found")
    if re.search(r'\*\*[^*]+\*\*', text):
        issues.append("leftover markdown bold found")
    return len(issues) == 0, issues

def auto_fix_mechanical_issues(text):
    fixed = text
    fixed = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1 percent', fixed)
    fixed = re.sub(r'\[([^\]]*)\]', r'\1', fixed)
    fixed = re.sub(r'\*\*([^*]+)\*\*', r'\1', fixed)
    fixed = re.sub(r'(?<!\w)\*([^*\n]+)\*(?!\w)', r'\1', fixed)
    fixed = re.sub(r'^\s*#+\s*', '', fixed, flags=re.MULTILINE)
    fixed = re.sub(r'^\s*[-*•]\s+', '', fixed, flags=re.MULTILINE)
    fixed = re.sub(r'~\s*(\d)', r'approximately \1', fixed)
    fixed = fixed.replace('~', '')
    fixed = re.sub(r' {2,}', ' ', fixed)
    return fixed.strip()

def parse_evaluator_response(response):
    try:
        lines = response.strip().splitlines()
        verdict_line = next((l for l in lines if l.upper().startswith("VERDICT")), "")
        reason_line = next((l for l in lines if l.upper().startswith("REASON")), "")
        verdict = "FAIL" if "FAIL" in verdict_line.upper() else "PASS"
        reason = reason_line.split(":", 1)[-1].strip() if ":" in reason_line else "none"
        return verdict, reason
    except Exception:
        return "PASS", "parse failure, defaulted to pass"

@spaces.GPU
def qwen_generate(system_prompt, user_prompt, max_new_tokens=1500, temperature=0.3):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    text = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = qwen_tokenizer([text], return_tensors="pt").to(qwen_model.device)
    with torch.no_grad():
        generated_ids = qwen_model.generate(
            **model_inputs, max_new_tokens=max_new_tokens,
            temperature=temperature, do_sample=True
        )
    generated_ids = [out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)]
    result = qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    del model_inputs, generated_ids
    gc.collect()
    torch.cuda.empty_cache()
    return result

if LANGGRAPH_AVAILABLE:
    class DirectorState(TypedDict):
        raw_text: str
        draft_script: str
        verdict: str
        feedback: str
        retry_count: int
        max_retries: int

    def director_node(state: DirectorState) -> DirectorState:
        try:
            if state["retry_count"] == 0:
                user_prompt = state["raw_text"]
            else:
                user_prompt = (
                    f"{state['raw_text']}\n\n"
                    f"(Internal note, do not reference this in your output: ensure the script avoids this specific problem: {state['feedback']})"
                )
            draft = qwen_generate(DIRECTOR_SYSTEM_PROMPT, user_prompt)
            if not draft:
                raise ValueError("Director returned empty output")
            state["draft_script"] = draft
        except Exception:
            state["draft_script"] = state.get("draft_script") or state["raw_text"]
        return state

    def evaluator_node(state: DirectorState) -> DirectorState:
        draft = state["draft_script"]
        det_passed, det_issues = deterministic_check(draft)
        try:
            response = qwen_generate(EVALUATOR_SYSTEM_PROMPT, draft, max_new_tokens=100)
            llm_verdict, llm_reason = parse_evaluator_response(response)
        except Exception:
            llm_verdict, llm_reason = "PASS", "LLM evaluator error, deterministic check used instead"

        if not det_passed:
            state["verdict"] = "FAIL"
            state["feedback"] = "; ".join(det_issues) + (f"; also: {llm_reason}" if llm_verdict == "FAIL" else "")
        else:
            state["verdict"] = llm_verdict
            state["feedback"] = llm_reason

        state["retry_count"] += 1
        return state

    def route_after_evaluation(state: DirectorState) -> str:
        if state["verdict"] == "PASS":
            return "end"
        if state["retry_count"] >= state["max_retries"]:
            return "end"
        return "retry"

    graph = StateGraph(DirectorState)
    graph.add_node("director", director_node)
    graph.add_node("evaluator", evaluator_node)
    graph.set_entry_point("director")
    graph.add_edge("director", "evaluator")
    graph.add_conditional_edges("evaluator", route_after_evaluation, {"retry": "director", "end": END})
    compiled_graph = graph.compile()


def process_paragraph(paragraph, para_idx, max_retries=2):
    yield f"[Director] Rewriting paragraph {para_idx}..."

    if LANGGRAPH_AVAILABLE:
        initial_state = {
            "raw_text": paragraph, "draft_script": "", "verdict": "",
            "feedback": "", "retry_count": 0, "max_retries": max_retries
        }
        state = director_node(initial_state)
        while True:
            yield f"[Evaluator] Checking paragraph {para_idx}..."
            state = evaluator_node(state)
            if state["verdict"] == "PASS":
                yield f"[Evaluator] PASS"
                break
            if state["retry_count"] >= state["max_retries"]:
                yield f"[Evaluator] FAIL — {state['feedback']} (max retries reached, using best draft)"
                break
            yield f"[Evaluator] FAIL — {state['feedback']}. Retrying..."
            state = director_node(state)
            yield f"[Director] Rewriting paragraph {para_idx} (attempt {state['retry_count'] + 1})..."
        script = state["draft_script"]
    else:
        script = qwen_generate(DIRECTOR_SYSTEM_PROMPT, paragraph)
        yield f"[Evaluator] Skipped (LangGraph unavailable) — using deterministic check only"

    passed, issues = deterministic_check(script)
    if not passed:
        script = auto_fix_mechanical_issues(script)
        yield f"[Director] Auto-fixed remaining mechanical issues in paragraph {para_idx}"

    yield ("RESULT", script)


@spaces.GPU
def clone_speech(text, speaker_wav, out_path):
    tts_model.tts_to_file(text=text, speaker_wav=speaker_wav, language="en", file_path=out_path, **CLONE_PARAMS)
    return out_path

def clean_and_prepare(input_path, output_path, target_duration_sec=60,
                       denoise_mix=0.8, target_lufs=-20.0, highpass_hz=80):
    import soundfile as sf
    import pyloudnorm as pyln
    import librosa
    import numpy as np

    model_path = "std.rnnn"
    if not os.path.exists(model_path):
        subprocess.run(["wget", "-q",
                         "https://raw.githubusercontent.com/richardpl/arnndn-models/master/std.rnnn",
                         "-O", model_path], check=True)

    denoised_path = "_temp_denoised.wav"
    cmd = ["ffmpeg", "-y", "-i", input_path,
           "-af", f"highpass=f={highpass_hz},arnndn=m='{model_path}':mix={denoise_mix}",
           denoised_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg denoise failed: {result.stderr[-500:]}")

    y, sr = librosa.load(denoised_path, sr=None)
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    trimmed_path = "_temp_trimmed.wav"
    sf.write(trimmed_path, y_trimmed, sr)

    data, rate = sf.read(trimmed_path)
    meter = pyln.Meter(rate)
    try:
        current_loudness = meter.integrated_loudness(data)
        normalized = pyln.normalize.loudness(data, current_loudness, target_lufs)
        peak = np.max(np.abs(normalized))
        if peak > 0.98:
            normalized = normalized * (0.98 / peak)
    except Exception:
        normalized = pyln.normalize.peak(data, -1.0)

    max_samples = int(target_duration_sec * rate)
    if len(normalized) > max_samples:
        normalized = normalized[:max_samples]

    sf.write(output_path, normalized, rate)
    for p in [denoised_path, trimmed_path]:
        if os.path.exists(p):
            os.remove(p)
    return output_path

def split_into_chunks(text, max_chars=180):
    text = text.strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) <= max_chars:
            current += (" " if current else "") + s
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks


def generate(audio_file, raw_text):
    if audio_file is None:
        raise gr.Error("Please upload a reference voice recording first.")
    if not raw_text or not raw_text.strip():
        raise gr.Error("Please paste the text you want narrated.")

    log = "🧹 Cleaning reference audio...\n"
    yield None, "", log, 0.05
    clean_path = clean_and_prepare(audio_file, "clean_reference.wav")
    log += "✅ Audio cleaned (denoised, trimmed, normalized)\n\n"
    yield None, "", log, 0.1

    paragraphs = [p for p in raw_text.split("\n\n") if p.strip()] or [raw_text]

    from pydub import AudioSegment
    sentence_pause = AudioSegment.silent(duration=250)
    paragraph_pause = AudioSegment.silent(duration=600)
    combined = AudioSegment.empty()
    full_script_parts = []
    chunk_counter = 0

    progress_per_paragraph = 0.8 / len(paragraphs)
    base_progress = 0.1

    for p_idx, paragraph in enumerate(paragraphs, start=1):
        clean_paragraph = None
        for item in process_paragraph(paragraph, p_idx):
            if isinstance(item, tuple) and item[0] == "RESULT":
                clean_paragraph = item[1]
            else:
                log += item + "\n"
                yield None, "", log, base_progress
        full_script_parts.append(clean_paragraph)

        chunks = split_into_chunks(clean_paragraph)
        log += f"[TTS] Paragraph {p_idx} -> {len(chunks)} chunk(s) queued\n"
        yield None, "", log, base_progress

        paragraph_audio = AudioSegment.empty()
        for c_idx, chunk in enumerate(chunks, start=1):
            chunk_counter += 1
            chunk_path = f"_chunk_{chunk_counter}.wav"
            clone_speech(chunk, clean_path, chunk_path)
            paragraph_audio += AudioSegment.from_wav(chunk_path) + sentence_pause
            os.remove(chunk_path)
            log += f"[TTS] ✅ chunk {c_idx}/{len(chunks)} done\n"
            step_progress = base_progress + progress_per_paragraph * (c_idx / len(chunks))
            yield None, "", log, step_progress

        combined += paragraph_audio + paragraph_pause
        log += f"[TTS] ✅ Paragraph {p_idx} audio assembled\n\n"
        base_progress += progress_per_paragraph
        yield None, "", log, base_progress

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = f"output_{timestamp}.wav"
    combined.export(final_path, format="wav")
    final_script = "\n\n".join(full_script_parts)

    log += f"✅ All done — {len(paragraphs)} paragraph(s), {chunk_counter} chunk(s) total\n"
    yield final_path, final_script, log, 1.0


THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    block_radius="12px",
)

CUSTOM_CSS = """
#header { text-align: center; margin-bottom: 0.5rem; }
#header h1 { font-size: 2rem; margin-bottom: 0.2rem; }
#header p { color: var(--body-text-color-subdued); }
.status-box textarea, .log-box textarea { font-size: 0.85rem !important; font-family: monospace !important; }
footer { visibility: hidden; }
"""

with gr.Blocks(theme=THEME, css=CUSTOM_CSS, title="Voice Cloner") as demo:
    with gr.Column(elem_id="header"):
        gr.Markdown(
            "# 🎙️ Voice Cloner\n"
            "Clone a voice from a short reference clip and narrate any text with it — "
            "powered by **XTTS-v2** for voice cloning and a **Qwen2.5 director/evaluator loop** "
            "(built with LangGraph) that rewrites text for natural TTS delivery."
        )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Reference voice")
            audio_input = gr.Audio(
                type="filepath",
                label="Upload a clean voice recording (10–60s)",
                sources=["upload", "microphone"],
            )
            gr.Markdown("### 2. Script")
            text_input = gr.Textbox(
                lines=10,
                label="Text to narrate",
                placeholder="Paste the text you want spoken in the cloned voice...",
            )
            run_btn = gr.Button("🎬 Generate Narration", variant="primary", size="lg")
            progress_bar = gr.Slider(0, 1, value=0, label="Progress", interactive=False)

            with gr.Accordion("Agent Log (live director/evaluator/TTS trace)", open=True):
                log_output = gr.Textbox(
                    label="",
                    lines=14,
                    interactive=False,
                    elem_classes="log-box",
                    show_label=False,
                )

        with gr.Column(scale=1):
            gr.Markdown("### Output")
            audio_output = gr.Audio(label="Generated narration", type="filepath")
            script_output = gr.Textbox(
                label="Cleaned script (what was actually spoken)",
                lines=10,
                interactive=False,
            )

    gr.Markdown(
        "---\n"
        "**Pipeline:** clean & denoise reference audio → per-paragraph LangGraph "
        "director/evaluator loop rewrites the script for TTS → XTTS-v2 clones the voice "
        "per chunk → chunks stitched with natural pacing. The Agent Log above streams "
        "each real step live as it happens.\n\n"
        "*Runs on Hugging Face ZeroGPU — models load once at startup, so the Space itself "
        "may take a minute to show as \"Running\" after each restart.*"
    )

    run_btn.click(
        fn=generate,
        inputs=[audio_input, text_input],
        outputs=[audio_output, script_output, log_output, progress_bar],
    )

demo.launch()
