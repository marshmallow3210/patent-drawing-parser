# Patent Drawing Image Parser (PDF Input)

This project parses **patent drawing pages** from a **PDF** (even without a text layer).  
It converts each PDF page to an image, automatically detects and corrects page rotation (OCR-based), performs a unified content crop, extracts OCR-based location hints, and then calls a pluggable LLM backend (provider-agnostic; this repo currently uses Google Gemini) to return structured JSON results (figure/component labels and optional hierarchy).

# Directory Structure
```
patent-drawing-parser/
├─ app.py
├─ requirements.txt
└─ src/
   ├─ pdf_utils.py       # PDF rendering + rotation correction + content cropping + OCR hint extraction (normalized 0–1000)
   └─ vision_lmm.py      # LLM backend (Gemini implementation): REST call + prompt + JSON parsing/cleanup
```

# Environment Requirements
- Python 3.10 ~ 3.12
- The commands below use Windows PowerShell as an example.

# Prerequisites
- Install **Tesseract OCR** (Windows) and ensure `tesseract.exe` is available.
- If needed, update the path in `src/pdf_utils.py`:
  - `pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"`


# Installation & Run (Windows PowerShell)
1. Place the entire patent-drawing-parser folder anywhere you like.
2. Open PowerShell and go to the project folder
```powershell
cd <YOUR_PATH>\patent-drawing-parser
```

3. Create and activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

4. Upgrade tools and install dependencies
```powershell
python -m pip install -U pip wheel setuptools
pip install -r requirements.txt
```

5. Set your API key (only effective in the current PowerShell session)
```powershell
$env:GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

- Verify it was set correctly
```powershell
$env:GEMINI_API_KEY
```

6. Start the backend server
```powershell
python .\app.py
```

- If you see the following message, the server is running
```powershell
 * Running on http://127.0.0.1:8000
```

7. Health check
```powershell
curl http://127.0.0.1:8000/api/health
```

- Expected example response
```powershell
{"dpi":400,"gemini_key_loaded":true,"ok":true,"provider":"gemini"}
```

8. Parse a PDF
- Parse the full document
```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/parse" ^
  -H "Accept: application/json" ^
  -F "file=@YOUR_FILE_NAME;type=application/pdf"
```

- Parse a single page (optional), e.g. page 1
```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/parse?page=1" ^
  -H "Accept: application/json" ^
  -F "file=@YOUR_FILE_NAME;type=application/pdf"
```

- Parse a page range (optional), e.g. pages 2~5 (inclusive)
```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/parse?from=2&to=5" ^
  -H "Accept: application/json" ^
  -F "file=@YOUR_FILE_NAME;type=application/pdf"
```

Include auto-rotation info per page (optional)
```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/parse?page=1&show_rotation=1" ^
  -H "Accept: application/json" ^
  -F "file=@YOUR_FILE_NAME;type=application/pdf"
```

# Output format
Each item corresponds to a detected figure on a page:
- page: page number (1-based)
- figure: normalized figure label (e.g., FIG. 1A)
- components: component labels found on the figure (sorted and deduplicated)
- hierarchy: optional parent-children relationships when the model can infer them

# Notes
- Replace `YOUR_GEMINI_API_KEY` and `YOUR_FILE_NAME` with your own API key and PDF file.

- The server will also write a corrected multi-page image file named:
```
corrected_<YOUR_FILE_NAME> (saved to the working directory).
```

- OCR hints are logged to:
```
ocr_log_<YOUR_FILE_NAME>.txt (saved to the working directory).
```

- Debug (optional): view the model raw response for a specific page
```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/debug?page=1" ^
  -H "Accept: application/json" ^
  -F "file=@YOUR_FILE_NAME;type=application/pdf"
```
- Response format:
```json
{"raw":"...","page_rotation":0}
```