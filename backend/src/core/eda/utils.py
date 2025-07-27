from fpdf import FPDF
import tempfile

def outlier_drift_report_pdf(stats, drift_stats):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, "Outlier & Drift Detection Report", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, "Outlier Statistics:", ln=True)
    for row in stats:
        pdf.cell(0, 8, f"{row['column']}: {row['outlier_count']} outliers ({row['outlier_percent']:.1f}%)", ln=True)
    pdf.cell(0, 10, "Drift Statistics:", ln=True)
    for row in drift_stats:
        drifted = "YES" if row.get("drifted") else "NO"
        pdf.cell(0, 8, f"{row['column']}: KS={row.get('ks_stat', 0):.2f}, W={row.get('wasserstein_distance', 0):.2f}, Drifted: {drifted}", ln=True)
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tf.name)
    tf.close()
    return tf.name
