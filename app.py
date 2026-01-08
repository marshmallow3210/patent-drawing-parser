# app.py
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from src.pdf_utils import pdf_to_images
from src.vision_lmm import extract_figures_with_lmm, debug_gemini_raw

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

DPI = 400


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "provider": "gemini",
        "dpi": DPI,
        "gemini_key_loaded": bool(os.getenv("GEMINI_API_KEY")),
    })


@app.post("/api/parse")
def parse():
    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400
    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "empty file"}), 400

    filename = secure_filename(f.filename)
    tmp_path = os.path.join(".", "_tmp_" + filename)
    f.save(tmp_path)

    # Optional page selection parameters:
    #   ?page=3            -> only page 3
    #   ?from=2&to=5       -> pages 2~5 (inclusive)
    #   no params          -> the whole document (default)
    page_param = request.args.get("page")
    frm_param = request.args.get("from")
    to_param = request.args.get("to")

    try:
        pages = pdf_to_images(tmp_path, dpi=DPI)
        total = len(pages)

        if page_param:
            try:
                p = int(page_param)
            except ValueError:
                return jsonify({"error": "page must be int"}), 400
            if p < 1 or p > total:
                return jsonify({"error": f"page out of range (1..{total})"}), 400
            selected = [pages[p - 1]]
            base_index = p  # 1-based
        elif frm_param or to_param:
            try:
                frm = int(frm_param) if frm_param else 1
                to = int(to_param) if to_param else total
            except ValueError:
                return jsonify({"error": "from/to must be int"}), 400
            if frm < 1 or to < 1 or frm > to or to > total:
                return jsonify({"error": f"invalid range. valid: 1..{total}"}), 400
            selected = pages[frm - 1:to]
            base_index = frm
        else:
            selected = pages
            base_index = 1

        if page_param:
            page_numbers = [p]
        elif frm_param or to_param:
            page_numbers = list(range(frm, to + 1))
        else:
            page_numbers = None  # default: 1..N

        raw_results = extract_figures_with_lmm(
            selected,
            page_numbers=page_numbers,
            filename=filename
        )

        if len(selected) > 0:
            output_filename = "corrected_" + filename
            # Save as a multi-page file; pass images after the first one as append_images.
            selected[0].save(
                output_filename,
                save_all=True,
                append_images=selected[1:] if len(selected) > 1 else [],
                resolution=DPI
            )
            print(f"Saved corrected file: {output_filename}")

        show_rotation = request.args.get("show_rotation", "0") == "1"
        if show_rotation:
            # Build a page -> rotation lookup table
            if page_numbers is None:
                # Whole document: pages 1..N
                rot_map = {
                    i: (pages[i - 1].info.get("auto_rotate_deg", 0))
                    for i in range(1, len(pages) + 1)
                }
            else:
                # Specific page / range
                rot_map = {
                    pn: img.info.get("auto_rotate_deg", 0)
                    for pn, img in zip(page_numbers, selected)
                }

            # Attach rotation to each result item (same page => same rotation)
            for item in raw_results:
                try:
                    pg = int(item.get("page", 0))
                except Exception:
                    pg = 0
                item["page_rotation"] = int(rot_map.get(pg, 0))

        return jsonify(raw_results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass


@app.post("/api/debug")
def debug():
    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400
    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "empty file"}), 400

    filename = secure_filename(f.filename)
    tmp_path = os.path.join(".", "_tmp_" + filename)
    f.save(tmp_path)
    try:
        pages = pdf_to_images(tmp_path, dpi=DPI)
        total = len(pages)
        page_num = int(request.args.get("page", "1"))
        if page_num < 1 or page_num > total:
            return jsonify({"error": f"page out of range (1..{total})"}), 400

        raw = debug_gemini_raw(pages[page_num - 1], page_num)
        rot = pages[page_num - 1].info.get("auto_rotate_deg", 0)
        return jsonify({"raw": raw, "page_rotation": int(rot)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)