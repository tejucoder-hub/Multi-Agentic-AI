# # Step1: Install tectonic & Import deps
# from langchain_core.tools import tool
# from datetime import datetime
# from pathlib import Path
# import subprocess
# import shutil

# @tool
# def render_latex_pdf(latex_content: str) -> str:
#     """Render a LaTeX document to PDF.

#     Args:
#         latex_content: The LaTeX document content as a string

#     Returns:
#         Path to the generated PDF document
#     """
#     if shutil.which("tectonic") is None:
#         raise RuntimeError(
#             "tectonic is not installed. Install it first on your system."
#         )

#     try:
#         # Step2: Create directory
#         output_dir = Path("output").absolute()
#         output_dir.mkdir(exist_ok=True)
#         # Step3: Setup filenames
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         tex_filename = f"paper_{timestamp}.tex"
#         pdf_filename = f"paper_{timestamp}.pdf"
#         # Step4: Export as tex & pdf
#         tex_file = output_dir / tex_filename
#         tex_file.write_text(latex_content)

#         result = subprocess.run(
#                     ["tectonic", tex_filename, "--outdir", str(output_dir)],
#                     cwd=output_dir,
#                     capture_output=True,
#                     text=True,
#                 )

#         final_pdf = output_dir / pdf_filename
#         if not final_pdf.exists():
#             raise FileNotFoundError("PDF file was not generated")

#         print(f"Successfully generated PDF at {final_pdf}")
#         return str(final_pdf)

#     except Exception as e:
#         print(f"Error rendering LaTeX: {str(e)}")
#         raise


# # check working properly or not 
# # sample_latex_data = r"""
# # \documentclass{article}
# # \begin{document}
# # Hello dosto, Jai Hind.
# # \end{document}
# # """

# # render_latex_pdf(sample_latex_data)


# ----------------------------------
# ---------------------------------------------
import re
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas


OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def clean_filename(text: str, max_len: int = 80) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s_-]", "", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip("_")
    if not text:
        return "research_paper"
    return text[:max_len]


def extract_title_and_body(content: str) -> tuple[str, str]:
    text = (content or "").strip()
    if not text:
        return "Research Paper Export", "No content provided."

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    banned_title_phrases = [
        "your research paper draft",
        "you can download the pdf",
        "download pdf",
        "sandbox:/",
        "pdf has been generated successfully",
        "use the attached file below to download it",
    ]

    for line in lines[:15]:
        low = line.lower()

        if any(bad in low for bad in banned_title_phrases):
            continue

        if low.startswith("title:"):
            title = line.split(":", 1)[1].strip()
            if title:
                return title, text

    for line in lines[:12]:
        low = line.lower()

        if any(bad in low for bad in banned_title_phrases):
            continue

        if low in {
            "abstract",
            "introduction",
            "conclusion",
            "references",
            "methodology",
            "discussion",
            "results",
            "background",
            "literature review",
            "future work",
        }:
            continue

        if 12 < len(line) < 140:
            return line, text

    return "Research Paper Export", text


def init_page(c: canvas.Canvas, title: str):
    c.setTitle(title)
    c.setAuthor("Research Workspace")
    c.setSubject(title)
    c.setFont("Helvetica", 11)


def draw_wrapped_block(
    c: canvas.Canvas,
    text: str,
    x: int,
    y: int,
    width: int,
    font_name: str,
    font_size: int,
    line_gap: int,
    page_height: int,
) -> int:
    lines = simpleSplit(text, font_name, font_size, width)
    c.setFont(font_name, font_size)

    for line in lines:
        if y < 80:
            c.showPage()
            init_page(c, "Research Paper Export")
            c.setFont(font_name, font_size)
            y = page_height - 60
        c.drawString(x, y, line)
        y -= line_gap

    return y


def write_paragraph(c: canvas.Canvas, paragraph: str, x: int, y: int, width: int, page_height: int) -> int:
    paragraph = paragraph.strip()
    if not paragraph:
        return y

    lowered = paragraph.lower()
    heading_like = (
        len(paragraph) < 80
        and lowered in {
            "abstract",
            "introduction",
            "background",
            "literature review",
            "methodology",
            "results",
            "discussion",
            "conclusion",
            "future work",
            "references",
        }
    )

    if heading_like:
        if y < 100:
            c.showPage()
            init_page(c, "Research Paper Export")
            y = page_height - 60
        c.setFont("Helvetica-Bold", 14)
        c.drawString(x, y, paragraph.title())
        y -= 22
        return y

    c.setFont("Helvetica", 11)
    lines = simpleSplit(paragraph, "Helvetica", 11, width)

    for line in lines:
        if y < 80:
            c.showPage()
            init_page(c, "Research Paper Export")
            c.setFont("Helvetica", 11)
            y = page_height - 60
        c.drawString(x, y, line)
        y -= 16

    return y - 8


@tool
def render_latex_pdf(content: str) -> str:
    """Create a clean PDF from plain research text content.

    Args:
        content: Research draft or report in plain text.

    Returns:
        Absolute path to the generated PDF file.
    """
    text = (content or "").strip()
    if not text:
        raise ValueError("No content provided for PDF generation.")

    title, body = extract_title_and_body(text)
    safe_name = clean_filename(title)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = OUTPUT_DIR / f"{safe_name}_{timestamp}.pdf"

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    page_width, page_height = A4
    left_margin = 50
    right_margin = 50
    usable_width = page_width - left_margin - right_margin

    init_page(c, title)

    y = page_height - 60

    y = draw_wrapped_block(
        c=c,
        text=title,
        x=left_margin,
        y=y,
        width=usable_width,
        font_name="Helvetica-Bold",
        font_size=22,
        line_gap=28,
        page_height=page_height,
    )

    y -= 8
    c.setFont("Helvetica", 11)
    c.drawString(left_margin, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 26

    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [body]

    for paragraph in paragraphs:
        y = write_paragraph(
            c=c,
            paragraph=paragraph,
            x=left_margin,
            y=y,
            width=usable_width,
            page_height=page_height,
        )

    c.save()
    return str(pdf_path.resolve())