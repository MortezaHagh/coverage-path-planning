import cairosvg
import os

figures_dir = 'docs/figures'
files = ['system_overview_v2.svg', 'plugin_architecture_v2.svg']

for file in files:
    svg_path = os.path.join(figures_dir, file)
    pdf_path = os.path.join(figures_dir, file.replace('.svg', '.pdf'))

    if os.path.exists(svg_path):
        print(f"Converting {svg_path} to {pdf_path}")
        cairosvg.svg2pdf(url=svg_path, write_to=pdf_path)
    else:
        print(f"File not found: {svg_path}")
