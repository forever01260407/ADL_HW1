"""Preserve the original eight-page report and append one clarification page.

This optional document-authoring script requires reportlab and pypdf. They are
not inference/training dependencies and are never imported by run.sh.
The appendix contains verified validation evidence only. It does not read any
test labels, training examples, or model weights. All numbers below were checked
against the corresponding saved run_summary.json, validation_metrics.json, and
analysis/diagnostics.json files on 2026-10-04.

Example (from the submission folder):
  python report_sources/build_report.py \
    --source report_sources/original_report.pdf --output report.pdf
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import RectangleObject
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


VERIFIED = {
    "selected_run": "v02_ctx128_weighted",
    "legacy_run": "v02_legacy_weighted",
    "long_context_run": "v02_ctx512_weighted_s42",
    "mbert_run": "mbert_ctx128_weighted",
    "selected_macro": 0.8350549539976889,
    "selected_micro": 0.8430257979886314,
    "legacy_macro": 0.8213197794785131,
    "legacy_micro": 0.8292465448060633,
    "mbert_macro": 0.8038236501716309,
    "mbert_micro": 0.8130008904719501,
    "class_f1_delta_128_to_512_pp": {
        "Too_Expensive": -5.67603748326639,
        "Compare_Competitor": -3.28214393255043,
        "Doubt": -2.62067723342939,
        "State_Need": 3.34415584415584,
    },
}


def appendix_page(width: float, height: float) -> PdfReader:
    buffer = io.BytesIO()
    doc = canvas.Canvas(buffer, pagesize=(width, height))
    doc.setTitle("ADL HW1 - Clarification appendix")
    doc.setAuthor("r15922a14")
    doc.setFillColor(colors.HexColor("#F6F2E9"))
    doc.rect(0, 0, width, height, fill=1, stroke=0)
    doc.setFillColor(colors.HexColor("#243D38"))
    doc.rect(56, height - 65, 8, 38, fill=1, stroke=0)
    ink = colors.HexColor("#1F2523")
    muted = colors.HexColor("#48564F")
    available_width = width - 112

    def text(value: str, x: float, top: float, w: float, size: float = 18,
             leading: float | None = None, bold: bool = False,
             color: colors.Color = ink) -> float:
        style = ParagraphStyle(
            "appendix",
            fontName="Helvetica-Bold" if bold else "Helvetica",
            fontSize=size,
            leading=leading or size * 1.25,
            textColor=color,
            spaceAfter=0,
        )
        paragraph = Paragraph(value, style)
        _, paragraph_height = paragraph.wrap(w, height)
        paragraph.drawOn(doc, x, top - paragraph_height)
        return top - paragraph_height

    text("Reproducibility and clarification appendix", 82, height - 27,
         width - 138, size=34, leading=40, bold=True)
    text("r15922a14 | Supplement to the original report (pages 1-8 preserved). "
         "All scores below come from the saved 600-example validation split.",
         56, height - 88, available_width, size=17, color=muted)

    text("Selected submission and numerical correction", 56, height - 143,
         available_width, size=22, bold=True)
    text("The submitted checkpoint is <b>XLM-R-base, ctx128, split seed 42 / model seed 42</b>, "
         "with weighted BCE and calibrated per-label thresholds. "
         "This identifies the selected uploaded model; it does not claim global optimality.",
         56, height - 177, available_width, size=18)
    macro_delta = 100 * (VERIFIED["selected_macro"] - VERIFIED["mbert_macro"])
    micro_delta = 100 * (VERIFIED["selected_micro"] - VERIFIED["mbert_micro"])
    text(f"<b>Correction to page 4:</b> XLM-R exceeds mBERT by <b>{macro_delta:.2f} percentage points "
         f"Macro-F1</b> and <b>{micro_delta:.2f} percentage points Micro-F1</b> "
         "(the original page-3 comparison table is correct).",
         56, height - 229, available_width, size=18)

    text("Long-input preprocessing: two bundled strategies", 56, height - 286,
         available_width, size=22, bold=True)
    cell_style = ParagraphStyle("cell", fontName="Helvetica", fontSize=17,
                                leading=21, textColor=ink)
    data = [
        ["Preprocessing strategy", "Macro-F1", "Micro-F1"],
        [Paragraph("<b>Legacy:</b> context first, recent token-suffix cropping; "
                   "no whole-turn packing or explicit current/history markers.", cell_style),
         f"{VERIFIED['legacy_macro']:.5f}", f"{VERIFIED['legacy_micro']:.5f}"],
        [Paragraph("<b>Selected ctx128:</b> current utterance first, recent whole-turn packing, "
                   "128-token context budget, explicit current/history role markers.", cell_style),
         f"{VERIFIED['selected_macro']:.5f}", f"{VERIFIED['selected_micro']:.5f}"],
    ]
    table = Table(data, colWidths=[available_width - 284, 142, 142],
                  rowHeights=[33, 57, 57])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243D38")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (1, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 17),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#FFFFFF")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("LINEBELOW", (0, 1), (-1, 1), 0.7, colors.HexColor("#D8DED8")),
    ]))
    _, table_height = table.wrap(available_width, height)
    table.drawOn(doc, 56, height - 320 - table_height)
    text("Both runs use the same 2,400/600 split, split/model seed 42, weighted loss, "
         "512-token total cap, and calibration procedure. Input order, turn packing, "
         "and markers change together; the difference cannot isolate one component's causal effect.",
         56, height - 481, available_width, size=16, leading=20)

    text("Affected classes: ctx128 to ctx512", 56, height - 540,
         665, size=21, bold=True)
    y = height - 575
    for label, delta in VERIFIED["class_f1_delta_128_to_512_pp"].items():
        text(label, 56, y, 475, size=18)
        text(f"{delta:+.2f} pp F1", 544, y, 166, size=18, bold=True)
        y -= 26

    text("A 512-token context budget is not unlimited", 774, height - 540,
         width - 830, size=21, bold=True)
    text("All experiments cap the <b>entire encoded input at 512 tokens</b>, including "
         "utterance and special tokens. Context receives only the remaining space. "
         "Recent turns are packed; an oversized current utterance keeps its head and tail. "
         "Thus ctx512 cannot retain every history in a long conversation.",
         774, height - 577, width - 830, size=17, leading=22)

    text("Interpretation limit", 56, height - 701, available_width,
         size=21, bold=True)
    text("Checkpoint selection and per-label threshold calibration reuse the same validation split. "
         "These scores describe this validation split and may be optimistic; they do not establish "
         "performance on a separate held-out test set or support a generalization claim.",
         56, height - 735, available_width, size=17, leading=21)
    doc.setStrokeColor(colors.HexColor("#CCD4CD"))
    doc.line(56, 29, width - 56, 29)
    text("Evidence: saved validation metrics and diagnostics for the named runs; "
         "preprocessing behavior verified in common.py. PDF authoring only; no inference dependency added.",
         56, 23, width - 135, size=11, leading=13, color=muted)
    doc.setFillColor(ink)
    doc.setFont("Helvetica", 16)
    doc.drawRightString(width - 56, 12, "9")
    doc.save()
    buffer.seek(0)
    return PdfReader(buffer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Original eight-page PDF to preserve")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output nine-page report.pdf")
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError("Source and output must differ; the original is preserved")
    original = PdfReader(args.source)
    if len(original.pages) != 8:
        raise ValueError(f"Expected the original eight-page report, got {len(original.pages)}")
    first_box = original.pages[0].mediabox
    appendix = appendix_page(float(first_box.width), float(first_box.height)).pages[0]
    appendix.add_transformation(Transformation().translate(
        tx=float(first_box.left), ty=float(first_box.bottom)))
    for box_name in ("mediabox", "cropbox", "trimbox", "bleedbox", "artbox"):
        setattr(appendix, box_name, RectangleObject(list(getattr(original.pages[0], box_name))))
    output = PdfWriter()
    for page in original.pages:
        output.add_page(page)
    output.add_page(appendix)
    output.add_metadata({"/Title": "ADL HW1 - r15922a14",
                         "/Author": "r15922a14",
                         "/Subject": "Original report with verified clarification appendix"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        output.write(handle)
    checked = PdfReader(args.output)
    assert len(checked.pages) == 9
    for index, original_page in enumerate(original.pages):
        assert checked.pages[index].get_contents().get_data() == original_page.get_contents().get_data()
        assert list(checked.pages[index].mediabox) == list(original_page.mediabox)
    print(f"Created {args.output}: eight original pages preserved plus one appendix")


if __name__ == "__main__":
    main()
