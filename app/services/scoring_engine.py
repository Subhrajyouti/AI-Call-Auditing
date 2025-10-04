# app/services/scoring_engine.py
import os
import re
import json
import pandas as pd
import google.generativeai as genai
from app.config import GEMINI_API_KEY
import librosa
import numpy as np
from typing import Dict, Any

# configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

# Path to the audit parameters Excel (your uploaded file)
AUDIT_SHEET_PATH = "Audit Parameters (1).xlsx"  # make sure this is in project root

# --- 1) Load parameters & weights from the sheet on import ---
def load_parameters_from_sheet(path: str = AUDIT_SHEET_PATH):
    """
    Reads sheet and returns a list of (param_name, weight)
    Only picks rows that match: Name (N)
    """
    if not os.path.exists(path):
        # fallback: handful defaults (defensive)
        return [
            ("Call Greeting & Introduction", 5),
            ("Call Control & Professionalism", 5),
            ("Customer Needs Analysis", 5),
            ("Product Knowledge & Solution Offering", 5),
            ("Booking Process", 5),
            ("Handling Customer Objections/Queries", 5),
            ("Call Closure", 5),
            ("Active Listening & Empathy", 5),
            ("Tone & Rapport Building", 5),
            ("Hold Etiquette", 5),
            ("Upselling Attempted", 5),
            ("Problem-Solving & Resolution", 10),
            ("Communication Skills", 10),
            ("Compliance & Data Security", 10),
        ]
    xl = pd.ExcelFile(path)
    df = xl.parse(xl.sheet_names[0]).fillna('')
    params = []
    # columns discovered earlier: parameter name appears in column index 3
    for _, row in df.iterrows():
        cell = str(row.iloc[3]).strip()
        if cell and '(' in cell and ')' in cell:
            m = re.search(r'(.+)\((\d+)\)', cell)
            if m:
                name = m.group(1).strip()
                weight = int(m.group(2))
                params.append((name, weight))
    return params

PARAMS_WITH_WEIGHTS = load_parameters_from_sheet()

# --- 2) Decide which parameters are audibly-detectable ---
# Only these will be scored by the AI transcript + audio. (Others remain unscored by AI)
AUDIBLE_ALLOWLIST = {
    "Call Greeting & Introduction",
    "Call Control & Professionalism",
    "Customer Needs Analysis",
    "Product Knowledge & Solution Offering",
    "Booking Process",
    "Handling Customer Objections/Queries",
    "Call Closure",
    "Active Listening & Empathy",
    "Tone & Rapport Building",
    "Hold Etiquette",
    "Upselling Attempted",
    "Problem-Solving & Resolution",
    "Communication Skills",
    "Compliance & Data Security",
    # background noise & transfer accuracy handled separately
}

# small helper to construct strict Gemini prompt
def build_strict_prompt(transcript: str, params_weights: Dict[str,int]) -> str:
    """
    Build a tightly constrained prompt instructing Gemini to return strict JSON.
    Each parameter must be 'pass' or 'fail' and there must be a one-line evidence string
    (an exact quote / short excerpt from transcript) OR 'No evidence'.
    The model must not output any commentary beyond the specified JSON.
    """
    param_list = []
    for p,w in params_weights.items():
        param_list.append(f"- {p} : {w} points")

    param_text = "\n".join(param_list)

    prompt = f"""
You are a strict auditor assistant. You will be given a call transcript. Evaluate only the listed parameters and ONLY using evidence present in the transcript. Do NOT assume things not in the transcript. For each parameter return either pass or fail. If pass, the mark awarded is the parameter's full points (as provided); if fail, award 0. Provide a single short evidence snippet from the transcript (no more than 15 words) that justifies the pass. If there is no clear evidence in the transcript for pass, mark fail and evidence 'No evidence'.

Return EXACTLY a JSON object (no commentary, no explanation). Format:

{{
  "per_parameter": {{
     "Parameter Name": {{"status": "pass" or "fail", "evidence": "short excerpt", "weight": number}},
     ...
  }},
  "notes": "optional short note if necessary"
}}

Now evaluate the transcript provided below.

Parameters and weights (only these can be evaluated; everything else is out of scope):
{param_text}

Transcript:
\"\"\"
{transcript}
\"\"\"

Remember: JSON only. Each evidence should be an exact short quote from the transcript or 'No evidence'. For ambiguous cases prefer 'fail'.
"""
    return prompt

# --- 3) Background noise detection (simple energy-based) ---
def detect_background_noise(file_path: str, sr=16000, rms_threshold=0.01) -> bool:
    """
    Simple background noise detector:
    - loads audio via librosa
    - computes mean RMS energy
    - returns True if energy indicates notable background noise compared to voice energy.
    This is a heuristic and should be tuned per your dataset.
    """
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
        # compute RMS
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        mean_rms = float(np.mean(rms))
        # Also compute percent of frames with low energy (silence ratio)
        silence_frames = np.sum(rms < (mean_rms * 0.3))
        silence_ratio = silence_frames / max(1, rms.shape[0])
        # Heuristic: high mean RMS or lots of non-silence frames -> denote noise present
        return mean_rms > rms_threshold
    except Exception:
        # fail-safe: return False (no detection)
        return False

# --- 4) Primary scoring function ---
def summarize_call(transcript: str) -> Dict[str, str]:
    """
    Uses Gemini to summarize the call:
    - what was the issue
    - what actions agent took
    - how it was resolved
    - overall summary
    Returns a structured dictionary.
    """
    summary_prompt = f"""
You are a call audit summarizer. Read the following customer service call transcript
and extract these details clearly:

1. Customer Issue or Request (brief)
2. Actions Taken by Agent
3. Resolution or Outcome
4. Overall Summary (neutral, concise 3–4 lines)

Be concise and factual. Return ONLY JSON like:
{{
  "customer_issue": "...",
  "agent_actions": "...",
  "resolution": "...",
  "summary": "..."
}}

Transcript:
\"\"\"{transcript}\"\"\"
    """

    try:
        response = model.generate_content(
            summary_prompt,
            generation_config={"temperature": 0.3, "top_p": 0.9}
        )
        raw_text = response.text.strip()
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        json_text = match.group(0) if match else "{}"
        summary = json.loads(json_text)
    except Exception as e:
        print("⚠️ Summary generation failed:", e)
        summary = {
            "customer_issue": "Not detected",
            "agent_actions": "Not detected",
            "resolution": "Not detected",
            "summary": "Could not summarize this call."
        }

    return summary

def score_transcript(transcript: str, file_path: str = None) -> Dict[str, Any]:
    """
    Improved version with stronger prompt, example-based instruction,
    and robust JSON parsing fallback.
    """

    param_weights = {name: w for (name, w) in PARAMS_WITH_WEIGHTS if name in AUDIBLE_ALLOWLIST}

    # --- STRONGER PROMPT ---
    param_text = "\n".join([f"- {p} ({w} points)" for p, w in param_weights.items()])
    prompt = f"""
You are an experienced QA auditor analyzing a customer service call transcript.
Your task is to audit the agent's performance strictly based on what is said in the transcript.
Follow these rules carefully:

- Only give full marks or zero marks (no partial).
- Return JSON only. Do not include explanations outside JSON.
- If transcript clearly shows compliance with a rule → mark "pass"
- If not explicitly clear → mark "fail"
- Each "evidence" must be an exact phrase from the transcript proving the pass.
- If no proof, evidence = "No evidence".
- Default to "fail" if uncertain.

### Example:
If the parameter is "Greeting & Introduction (5)", and the transcript starts with
"Good morning, welcome to Happy Foods", then output:
"status": "pass", "evidence": "Good morning, welcome to Happy Foods"

If the transcript never greets → "status": "fail", "evidence": "No evidence"

Now audit this transcript strictly using only the following parameters and weights:
{param_text}

Transcript:
\"\"\"{transcript}\"\"\"

Output format:
{{
  "per_parameter": {{
     "Parameter Name": {{"status": "pass"/"fail", "evidence": "...", "weight": number}},
     ...
  }}
}}
    """

    # --- CALL GEMINI ---
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.1, "top_p": 0.9}
    )

    raw_text = response.text.strip()

    # --- JSON RECOVERY ---
    # Try to extract JSON even if Gemini adds extra text
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    json_text = match.group(0) if match else "{}"

    try:
        parsed = json.loads(json_text)
    except Exception as e:
        print("⚠️ Gemini returned invalid JSON, using fallback:", e)
        parsed = {"per_parameter": {}}
        for p, w in param_weights.items():
            parsed["per_parameter"][p] = {"status": "fail", "evidence": "No evidence", "weight": w}

    # --- SCORE CALCULATION ---
    per_parameter = {}
    total_score = 0
    max_score = 0
    for p, w in param_weights.items():
        max_score += w
        entry = parsed.get("per_parameter", {}).get(p, {})
        status = entry.get("status", "fail").lower()
        evidence = entry.get("evidence", "No evidence")
        mark = w if status == "pass" else 0
        per_parameter[p] = {"weight": w, "mark": mark, "evidence": evidence}
        total_score += mark

    # --- BACKGROUND NOISE DETECTION ---
    if file_path:
        noise_flag = detect_background_noise(file_path)
        per_parameter["Background Noise"] = {
            "weight": 0,
            "mark": 0,
            "evidence": "Yes" if noise_flag else "No"
        }

    # --- TRANSFER EVIDENCE ---
    if re.search(r'\btransfer(?:red|ring)?\b', transcript, re.I):
        per_parameter["Transfer Accuracy"] = {
            "weight": 0,
            "mark": 0,
            "evidence": "Transfer word found in transcript"
        }

    call_summary = summarize_call(transcript)

    return {
        "per_parameter": per_parameter,
        "total_score": total_score,
        "max_score": max_score,
        "summary": call_summary,
        "raw_model_output": raw_text
    }


