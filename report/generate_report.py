import argparse
from pathlib import Path

import pandas as pd
import yaml
from fpdf import FPDF

RHAT_EXCELLENT = 1.01
RHAT_GOOD      = 1.10
ESS_EXCELLENT  = 400
ESS_ACCEPTABLE = 100

COLOR_EXCELLENT = (220, 237, 200)
COLOR_GOOD      = (255, 243, 176)
COLOR_POOR      = (255, 204, 188)
COLOR_HEADER    = (220, 220, 220)

PAGE_MARGIN_MM = 15
FONT = "Helvetica"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a PDF report for an MCMC calibration run.")
    parser.add_argument("--bundle-dir", type=Path, required=True,
                        help="Path to the bundle output directory produced by BUNDLE_OUTPUTS.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PDF path. Defaults to report.pdf in CWD.")
    return parser.parse_args()


def parse_params(params_path: Path) -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def rhat_label(value: float) -> tuple:
    if value < RHAT_EXCELLENT:
        return "Excellent", COLOR_EXCELLENT
    elif value < RHAT_GOOD:
        return "Good", COLOR_GOOD
    return "Poor", COLOR_POOR


def ess_label(value: float) -> tuple:
    if value > ESS_EXCELLENT:
        return "Excellent", COLOR_EXCELLENT
    elif value > ESS_ACCEPTABLE:
        return "Acceptable", COLOR_GOOD
    return "Poor", COLOR_POOR


class MCMCReport(FPDF):
    def __init__(self, title: str, session_id: str):
        super().__init__()
        self._title = title
        self._session_id = session_id
        self.set_margins(PAGE_MARGIN_MM, PAGE_MARGIN_MM, PAGE_MARGIN_MM)
        self.set_auto_page_break(auto=True, margin=PAGE_MARGIN_MM)

    def header(self):
        self.set_font(FONT, "B", 10)
        self.cell(0, 6, self._title, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font(FONT, "I", 8)
        self.cell(0, 5, f"{self._session_id}", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font(FONT, "I", 8)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def section_title(self, text: str):
        self.set_font(FONT, "B", 13)
        self.set_fill_color(*COLOR_HEADER)
        self.cell(0, 9, text, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def subsection_title(self, text: str):
        self.set_font(FONT, "B", 11)
        self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text: str):
        self.set_font(FONT, "", 10)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def embed_image(self, img_path: Path, caption: str, w_mm: float = 160):
        x = (self.w - w_mm) / 2
        self.image(str(img_path), x=x, w=w_mm)
        self.set_font(FONT, "I", 9)
        self.cell(0, 6, caption, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)


def add_general_section(pdf: MCMCReport, cfg: dict):
    pdf.add_page()
    pdf.section_title("1. Run Information")

    cal   = cfg["calibration"]
    model = cfg["model"]

    # Two-column layout: run metadata left, parameter priors right
    left_x      = PAGE_MARGIN_MM
    right_x     = PAGE_MARGIN_MM + 95
    label_w     = 45   # fixed width so all values align at the same x
    val_w       = 45
    prior_key_w = 28
    row_h       = 6
    start_y     = pdf.get_y()

    # Left column: run metadata
    calibrate_noise  = cal.get("calibrate_noise", False)
    noise_parameters = cal.get("noise_parameters", [])

    run_lines = [
        ("Model name",        model["name"]),
        ("Calibrated params", ", ".join(cal["parameters"])),
        ("Likelihood",        cal["likelihood"]),
        ("MCMC walkers",      str(cal["nwalkers"])),
        ("Burn-in steps",     str(cal["nburn"])),
        ("Production steps",  str(cal["nsteps"])),
    ]
    run_lines.append(("Calibrate noise", str(calibrate_noise)))
    if calibrate_noise:
        run_lines.append(("Noise parameters", ", ".join(noise_parameters)))
    else:
        run_lines.append(("Noise sigma", str(cal.get("noise_sigma", ""))))
    pdf.set_font(FONT, "", 10)
    for label, value in run_lines:
        pdf.set_x(left_x)
        pdf.cell(label_w, row_h, f"{label}:", align="L")
        pdf.cell(val_w, row_h, value, align="L", new_x="LMARGIN", new_y="NEXT")

    left_end_y = pdf.get_y()

    # Right column: parameter priors
    pdf.set_xy(right_x, start_y)
    pdf.set_font(FONT, "B", 10)
    pdf.cell(0, row_h, "Parameter Priors", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    displayed_priors = [
        p for p in cal["priors"]
        if calibrate_noise or p["name"] not in noise_parameters
    ]
    for prior in displayed_priors:
        dist = prior["distribution"]
        attr = dist["attribute"]

        pdf.set_xy(right_x, pdf.get_y())
        pdf.set_font(FONT, "B", 10)
        pdf.cell(0, row_h, prior["name"], new_x="LMARGIN", new_y="NEXT")

        if dist["type"] == "uniform":
            prior_lines = [
                ("Lower bound", str(attr["lower_bound"])),
                ("Upper bound", str(attr["upper_bound"])),
            ]
        else:
            prior_lines = [
                ("Location", str(attr.get("location", ""))),
                ("Scale",    str(attr.get("scale", ""))),
            ]

        pdf.set_font(FONT, "", 9)
        for lbl, val in prior_lines:
            pdf.set_xy(right_x + 4, pdf.get_y())
            pdf.cell(prior_key_w, row_h - 1, f"{lbl}:", align="L")
            pdf.cell(0, row_h - 1, val, align="L", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    right_end_y = pdf.get_y()
    pdf.set_y(max(left_end_y, right_end_y) + 4)


def add_diagnostics_section(pdf: MCMCReport, bundle_dir: Path):
    diag_dir = bundle_dir / "diagnostics"

    pdf.add_page()
    pdf.section_title("2. MCMC Diagnostics")
    pdf.embed_image(img_path = diag_dir / "trace.png", caption="Figure 2 - Trace plots. Left: marginal posteriors; right: sample traces per walker.", w_mm=160)
    add_diagnostics_table(pdf, diag_dir / "convergence_diagnostics.csv")
    pdf.body_text(
        "R-hat thresholds: < 1.01 = Excellent, 1.01-1.10 = Good, >= 1.10 = Poor.\n"
        "ESS thresholds: > 400 = Excellent, 101-400 = Acceptable, <= 100 = Poor."
    )    
    pdf.embed_image(img_path = diag_dir / "autocorr.png", caption="Figure 3 - Autocorrelation by lag for each parameter.", w_mm=160)



def add_diagnostics_table(pdf: MCMCReport, csv_path: Path):
    df = pd.read_csv(csv_path)

    col_widths = [25, 20, 20, 22, 22, 20, 27, 27]
    headers    = ["Parameter", "Mean", "SD", "ESS bulk", "ESS tail",
                  "R-hat", "ESS quality", "R-hat quality"]
    row_h = 8

    pdf.set_font(FONT, "B", 9)
    pdf.set_fill_color(*COLOR_HEADER)
    for header, w in zip(headers, col_widths):
        pdf.cell(w, row_h, header, border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font(FONT, "", 9)
    for _, row in df.iterrows():
        ess_bulk = float(row["ess_bulk"])
        ess_tail = float(row["ess_tail"])
        rhat     = float(row["r_hat"])

        ess_bulk_text, ess_bulk_color = ess_label(ess_bulk)
        ess_tail_text, ess_tail_color = ess_label(ess_tail)
        rhat_text,     rhat_color     = rhat_label(rhat)

        # ESS quality uses the worse of bulk/tail
        ess_quality_text  = ess_bulk_text if ess_bulk <= ess_tail else ess_tail_text
        ess_quality_color = ess_bulk_color if ess_bulk <= ess_tail else ess_tail_color

        cells = [
            (str(row["parameter"]),    None,              "L"),
            (f"{float(row['mean']):.4f}", None,           "R"),
            (f"{float(row['sd']):.4f}",   None,           "R"),
            (f"{ess_bulk:.0f}",         ess_bulk_color,   "R"),
            (f"{ess_tail:.0f}",         ess_tail_color,   "R"),
            (f"{rhat:.4f}",             rhat_color,       "R"),
            (ess_quality_text,          ess_quality_color,"C"),
            (rhat_text,                 rhat_color,       "C"),
        ]

        for (text, color, align), w in zip(cells, col_widths):
            if color:
                pdf.set_fill_color(*color)
                pdf.cell(w, row_h, text, border=1, fill=True, align=align)
            else:
                pdf.cell(w, row_h, text, border=1, fill=False, align=align)
        pdf.ln()


def main():
    args = parse_args()
    bundle_dir = args.bundle_dir.resolve()
    output_path = args.output if args.output else Path("report.pdf")

    cfg = parse_params(bundle_dir / "_params.yml")
    title = f"Calibration Report / {cfg['model']['name']}"
    session_id = next(p for p in bundle_dir.name.split("_") if "-" in p)

    pdf = MCMCReport(title=title, session_id=session_id)
    add_general_section(pdf, cfg)
    pdf.embed_image(
        bundle_dir / "corner_plot.png",
        caption="Figure 1 - Pairwise posterior corner plot.",
        w_mm=150,
    )
    add_diagnostics_section(pdf, bundle_dir)

    pdf.output(str(output_path))
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
