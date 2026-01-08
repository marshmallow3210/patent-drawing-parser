# src/vision_lmm.py
import os
import re, json, base64, requests
from io import BytesIO
from typing import List, Dict, Tuple, Optional
from PIL import Image
from collections import OrderedDict
from .pdf_utils import (
    detect_and_correct_rotation,
    get_unified_content_bbox,
    apply_uniform_crop,
    get_ocr_hints
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # gemini-3-flash-preview
LMM_DEBUG      = os.getenv("LMM_DEBUG", "0") == "1"

# Regex to support: 140, 140', 140", 1a, G, W1, e1
_COMPONENT_RE = re.compile(r"^(?P<main>[a-zA-Z0-9\-]+)(?P<prime>['\"]{0,2})$")


def _component_sort_key(s: str):
    s = (s or "").strip()
    m = _COMPONENT_RE.fullmatch(s)
    if not m:
        return (9999, s, 9)  # Put non-matching labels at the end

    main_part = m.group("main")
    prime_part = m.group("prime")

    # Try extracting the leading number from main_part for numeric sorting
    num_match = re.match(r"(\d+)", main_part)
    num = int(num_match.group(1)) if num_match else 0

    # Prime/suffix weight
    prime_rank = {"": 0, "'": 1, '"': 2, "''": 3, '""': 4}.get(prime_part, 9)

    # Priority: 1) numeric value  2) remaining string  3) prime/suffix rank
    return (num, main_part.lower(), prime_rank)


def _require_key():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Please set the environment variable first.")


def _normalize_figure(fig: str) -> str:
    if not isinstance(fig, str):
        return ""
    s = fig.strip()
    m = re.search(r"(\d+)\s*([A-Za-z])?$", s)
    if not m:
        n = re.search(r"\d+", s)
        if not n:
            return ""
        num, tail = n.group(0), ""
    else:
        num, tail = m.group(1), (m.group(2) or "").upper()
    return f"FIG. {num}{tail}"


def _parse_json_array(txt: str) -> List[Dict]:
    if not txt:
        return []
    txt = txt.strip()

    # Remove Markdown fences like ```json ... ```
    if txt.startswith("```"):
        txt = re.sub(r"```(?:json)?\n?|\n?```", "", txt)

    try:
        return json.loads(txt)
    except Exception:
        # If parsing fails, try extracting from the first '[' to the last ']'
        match = re.search(r"(\[.*\])", txt, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                # If still failing, JSON might be truncated; attempt a simple tail fix
                fixed_txt = match.group(1)
                if not fixed_txt.endswith("]"):
                    fixed_txt += '"]}]'  # naive attempt to close brackets
                try:
                    return json.loads(fixed_txt)
                except:
                    return []
        return []


def extract_figures_with_lmm(
    pages: List[Image.Image],
    page_numbers: Optional[List[int]] = None,
    filename: str = "unknown"  # Used to name output log files
) -> List[Dict]:

    if page_numbers is None:
        page_numbers = list(range(1, len(pages) + 1))

    # Stage 1 & 2: rotation correction and unified crop bbox computation (keep behavior unchanged)
    for i in range(len(pages)):
        pages[i] = detect_and_correct_rotation(pages[i])
    unified_bbox = get_unified_content_bbox(pages)

    merged: Dict[Tuple[int, str], Dict] = {}

    # Prepare OCR log path
    ocr_log_path = f"ocr_log_{filename}.txt"

    with open(ocr_log_path, "w", encoding="utf-8") as f_log:
        for i, (page_no, page_img) in enumerate(zip(page_numbers, pages)):
            cropped_img = apply_uniform_crop(page_img, unified_bbox, padding=50)
            pages[i] = cropped_img

            # Get normalized OCR hints
            hints = get_ocr_hints(pages[i])

            # --- Logging: write OCR hints into the TXT log ---
            f_log.write(f"=== Page {page_no} OCR Hints (Normalized 0-1000) ===\n")
            f_log.write(json.dumps(hints, ensure_ascii=False, indent=2))
            f_log.write("\n\n")

            try:
                # Call Gemini with OCR hints
                res = _call_gemini_rest(pages[i], _prompt_for_page(page_no, hints))
                if not res:
                    print(f"DEBUG: Page {i+1} - Gemini returned empty or invalid JSON")
                else:
                    print(f"DEBUG: Page {i+1} - Gemini returned {len(res)} figures")
            except Exception as e:
                print(f"Error on page {page_no}: {e}")
                res = []

            for item in res or []:
                fig = _normalize_figure(item.get("figure", ""))
                if not fig:
                    continue

                key = (page_no, fig)
                if key not in merged:
                    merged[key] = {
                        "page": page_no,
                        "figure": fig,
                        "components": [],
                        "hierarchy": []
                    }

                new_comps = _clean_components(item.get("components", []))
                merged[key]["components"] = sorted(
                    list(set(merged[key]["components"] + new_comps)),
                    key=_component_sort_key
                )

                # Merge hierarchy relationships if present
                if "hierarchy" in item:
                    merged[key]["hierarchy"].extend(item["hierarchy"])

    # Format output: keep only requested fields, and sort by page + figure
    final_output = []
    sorted_keys = sorted(merged.keys(), key=lambda x: (x[0], _figure_sort_key(x[1])))

    for k in sorted_keys:
        item = merged[k]
        # Use OrderedDict to enforce output key order
        ordered_item = OrderedDict([
            ("components", item.get("components", [])),
            ("hierarchy", item.get("hierarchy", [])),
            ("figure", item.get("figure", "")),
            ("page", item.get("page", 0))
        ])
        final_output.append(ordered_item)

    return final_output


# Prompt builder: emphasizes the meaning of normalized coordinates
def _prompt_for_page(page_num: int, hints: List[Dict]) -> str:
    hints_json = json.dumps(hints[:50], ensure_ascii=False)  # limit to avoid excessive tokens
    return f"""
You are analyzing a patent drawings page (page {page_num}).

[Local OCR Hints]
The coordinates are normalized [ymin, xmin, ymax, xmax] from 0 to 1000:
{hints_json}

[Task]
Extract ALL component labels. Include:
1. Numbers (10, 140, 1601).
2. Labels with primes or suffixes (140', 150", 181a).
3. Dimension letters (G, W1, e1).
4. Labels with arrows (1a).

[Requirements]
- Return ONLY a JSON array: [{{
    "page": {page_num},
    "figure": "FIG. X",
    "components": ["10", "1a", "G"],
    "hierarchy": [
        {{
        "parent": "20",
        "children": ["201", "202", "203"]
        }}
    ]
    }}]
- Use the OCR hints to locate labels, but if you see more labels in the image, include them.
""".strip()


def _infer_once(img: Image.Image, page_idx: int) -> List[Dict]:
    from .pdf_utils import get_ocr_hints
    hints = get_ocr_hints(img)

    prompt = _prompt_for_page(page_idx, hints)
    res = _call_gemini_rest(img, prompt)

    if not res:
        print(f"DEBUG: Page {page_idx} - Gemini returned empty or invalid JSON")
    else:
        print(f"DEBUG: Page {page_idx} - Gemini returned {len(res)} figures")

    if isinstance(res, list):
        for item in res:
            item["raw_ocr_hints"] = hints
    return res


def _call_gemini_rest(img: Image.Image, prompt: str, return_raw: bool = False):
    if not GEMINI_API_KEY:
        return [] if not return_raw else ["[error] GEMINI_API_KEY not set"]

    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # model_name = "gemini-2.5-flash-lite"
    url = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    base_payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": b64}}
            ]
        }],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json"
        }
    }

    try:
        resp = requests.post(url, json=base_payload, timeout=120)

        # Debug: print the failure reason if non-200
        if resp.status_code != 200:
            print(f"Error Code: {resp.status_code}")
            print(f"Error Detail: {resp.text}")
            return [] if not return_raw else [f"[{resp.status_code}] {resp.text}"]

        data = resp.json()
        # Vertex AI responses are typically similar to AI Studio, but may be wrapped in candidates.
        txt = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

        if return_raw:
            return [txt]

        return _parse_json_array(txt)

    except Exception as e:
        return [] if not return_raw else [f"[exception] {repr(e)}"]


def _figure_sort_key(fig: str):
    """
    FIG. 10B -> (10, 'B'); FIG. 3 -> (3, '')
    """
    m_num = re.search(r"\d+", fig or "")
    num = int(m_num.group()) if m_num else 0
    m_chr = re.search(r"([A-Z])$", fig or "")
    suf = m_chr.group(1) if m_chr else ""
    return (num, suf)


def _sort_components_inplace(d: dict):
    """
    Deduplicate components, then sort by: numeric -> alpha -> prime/suffix.
    Example order: 10, 10p, 140', 150", 1501 ...
    """
    comps = list(dict.fromkeys(d.get("components", []) or []))  # dedupe while preserving order
    comps.sort(key=_component_sort_key)
    d["components"] = comps


def _clean_components(comps):
    # 1) Basic cleaning and regex matching
    comps = [c.strip() for c in comps if isinstance(c, str)]
    comps = [c for c in comps if _COMPONENT_RE.fullmatch(c)]
    if not comps:
        return []

    # 2) Decide whether to filter out single-digit pure numbers
    # Only filter labels that are:
    # - all digits
    # - length == 1 (e.g., "1", "2")
    # Labels containing letters (e.g., "1a", "G", "e1") should be kept.
    pure_numbers = [c for c in comps if c.isdigit()]
    has_multi_digit = any(len(c) >= 2 for c in pure_numbers)

    if has_multi_digit:
        # If there exists any 2+ digit pure number, drop single-digit pure numbers
        # while keeping all non-pure-number labels (e.g., G, 1a, W1).
        comps = [c for c in comps if not (c.isdigit() and len(c) < 2)]

    # 3) Deduplicate and sort
    return sorted(list(set(comps)), key=_component_sort_key)


# For /api/debug
def debug_gemini_raw(img: Image.Image, page_idx: int) -> str:
    from .pdf_utils import get_ocr_hints
    hints = get_ocr_hints(img)
    arr = _call_gemini_rest(img, _prompt_for_page(page_idx, hints), return_raw=True)
    return arr[0] if arr else ""