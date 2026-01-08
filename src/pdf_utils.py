# src/pdf_utils.py

import fitz
from PIL import Image, ImageOps
import io
import re
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def pdf_to_images(pdf_path: str, dpi: int = 300):
    images = []
    doc = fitz.open(pdf_path)
    try:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            images.append(img)
    finally:
        doc.close()
    return images


def rotate_image(img: Image.Image, angle: int) -> Image.Image:
    """Rotate counterclockwise by the given angle (degrees)."""
    return img.rotate(angle, expand=True)


def detect_and_correct_rotation(img: Image.Image) -> Image.Image:
    """
    Optimized: detect rotation using a low-resolution thumbnail to improve speed.
    """
    best_angle = 0
    max_figs = -1

    # Optimization: shrink the image to ~1000px on the long side for detection
    test_img = img.copy()
    test_img.thumbnail((1000, 1000))

    angles = [0, 90, 180, 270]
    fig_pattern = re.compile(r'FIG[\.\s]*\d+', re.IGNORECASE)

    for angle in angles:
        rotated_test = test_img.rotate(angle, expand=True)
        # Even on a thumbnail, set DPI to 300 to help Tesseract behave consistently
        text = pytesseract.image_to_string(rotated_test, config='--psm 11 --dpi 300')
        fig_count = len(fig_pattern.findall(text))

        if fig_count > max_figs:
            max_figs = fig_count
            best_angle = angle
        if fig_count >= 2:
            break

    # Apply the best rotation to the original (full-resolution) image
    if best_angle != 0:
        img = img.rotate(best_angle, expand=True)

    img.info["auto_rotate_deg"] = best_angle
    return img


def get_unified_content_bbox(images: list):
    """
    Optimized: compute content bounding boxes on downscaled images, then scale back up.
    """
    g_left, g_top, g_right, g_bottom = float('inf'), float('inf'), 0, 0
    found_any = False

    for img in images:
        # Create a small detection copy with width fixed at 800px
        detect_w = 800
        ratio = img.width / detect_w
        detect_h = int(img.height / ratio)

        small_img = img.resize((detect_w, detect_h), Image.Resampling.NEAREST)
        gray = small_img.convert("L")
        inverted = ImageOps.invert(gray)
        bbox = inverted.getbbox()

        if bbox:
            found_any = True
            # Scale detected bbox coordinates back to the original image size
            g_left = min(g_left, bbox[0] * ratio)
            g_top = min(g_top, bbox[1] * ratio)
            g_right = max(g_right, bbox[2] * ratio)
            g_bottom = max(g_bottom, bbox[3] * ratio)

    return (g_left, g_top, g_right, g_bottom) if found_any else None


def apply_uniform_crop(img: Image.Image, bbox: tuple, padding: int = 40):
    """
    Crop using the given unified bounding box plus padding.
    """
    if not bbox:
        return img

    l, t, r, b = bbox
    w, h = img.size

    # Add padding and clamp to image bounds
    l = max(0, l - padding)
    t = max(0, t - padding)
    r = min(w, r + padding)
    b = min(h, b + padding)

    return img.crop((l, t, r, b))


def get_ocr_hints(img: Image.Image):
    """
    Get OCR hints, filter them, and normalize coordinates to 0-1000.
    """
    w_img, h_img = img.size
    # Use psm 11 for sparse text detection
    data = pytesseract.image_to_data(
        img,
        config='--psm 11 --dpi 300',
        output_type=pytesseract.Output.DICT
    )

    hints = []
    fig_pattern = re.compile(r"FIG[\.\s]*\d+", re.I)
    # Component regex: supports digits + optional letter + optional primes
    comp_pattern = re.compile(r"^\d+[A-Za-z]?['\"]{0,2}$")

    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])

        # 1) Basic confidence filtering and empty text removal
        if conf < 20 or not text:
            continue

        # 2) Filter invalid text
        # Drop overly long pure numbers (patent labels usually <= 6 digits) or label "0"
        if (text.isdigit() and len(text) > 6) or text == "0":
            continue

        left, top, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

        # 3) Coordinate normalization to 0-1000
        # This helps the LLM reason about locations across different image resolutions.
        ymin = int(top * 1000 / h_img)
        xmin = int(left * 1000 / w_img)
        ymax = int((top + h) * 1000 / h_img)
        xmax = int((left + w) * 1000 / w_img)

        norm_box = [ymin, xmin, ymax, xmax]

        if fig_pattern.match(text):
            hints.append({"type": "figure_label", "text": text, "box_2d": norm_box})
        elif comp_pattern.match(text) or len(text) == 1:
            # Include single letters (e.g., G) here; downstream logic can decide how to use them.
            hints.append({"type": "component", "text": text, "box_2d": norm_box})

    return hints