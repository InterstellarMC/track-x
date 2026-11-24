"""Typeset the measured Markdown manuscript to offline HTML and LaTeX."""

import html
import re
from pathlib import Path

OUT = Path("reports/omega")
source = (OUT / "FINAL_RESEARCH_REPORT.md").read_text().strip().split("\n\n")
tex = [
    r"\documentclass[10pt]{article}",
    r"\usepackage[margin=23mm]{geometry}",
    r"\usepackage{graphicx,hyperref}",
    r"\begin{document}",
]
css = """@page{size:A4;margin:17mm 17mm 16mm} body{font:10.5pt/1.4 Georgia,serif;color:#152d38}h1{font: bold 25pt/1.12 Georgia;color:#10445a}h2{font: bold 14pt/1.2 Georgia;color:#10445a;break-after:avoid;margin:14pt 0 4pt}p{margin:5pt 0 8pt}a{color:#075273;text-decoration:none}table{border-collapse:collapse;width:100%;font:8pt/1.38 Arial,sans-serif;margin:8pt 0 12pt}td,th{text-align:right;padding:3pt;border-bottom:1px solid #cfdbdf}td:first-child,th:first-child{text-align:left}th{background:#e9f1f4}img{width:100%;display:block}figure{margin:8pt 0;break-inside:avoid}figcaption{font:9pt Arial,sans-serif}"""
page = [
    f'<!doctype html><html lang="en"><meta charset="utf-8"><title>TRACK-X Omega</title><style>{css}</style><main>'
]


def link(m):
    return f'<a href="{html.escape(m[2], quote=True)}">{html.escape(m[1])}</a>'


def render(s):
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, html.escape(s))


def escape(s):
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s)
    return "".join(
        {
            "%": r"\%",
            "_": r"\_",
            "&": r"\&",
            "#": r"\#",
            "≥": r"$\geq$",
            "≤": r"$\leq$",
            "Δ": r"$\Delta$",
            "φ": r"$\phi$",
            "χ": r"$\chi$",
        }.get(ch, ch)
        for ch in s
    )


for block in source:
    if block.startswith("# "):
        title = block[2:]
        page.append("<h1>" + render(title) + "</h1>")
        tex.extend(
            [
                r"\title{" + escape(title) + "}",
                r"\author{TRACK-X contributors}",
                r"\date{}\maketitle",
            ]
        )
    elif block.startswith("## "):
        title = block[3:]
        if title in ("Independent confirmation", "Seed contamination and ablations"):
            page.append('<div style="break-before:page"></div>')
            tex.append(r"\clearpage")
        page.append("<h2>" + render(title) + "</h2>")
        tex.append(r"\section*{" + escape(title) + "}")
    elif block.startswith("|"):
        rows = [
            [x.strip() for x in r.strip("|").split("|")]
            for r in block.splitlines()
            if "---" not in r
        ]
        page.append("<table>")
        tex.append(
            r"\noindent\resizebox{\linewidth}{!}{\begin{tabular}{l"
            + "r" * (len(rows[0]) - 1)
            + "}"
        )
        for i, row in enumerate(rows):
            tag = "td" if i else "th"
            page.append(
                "<tr>" + "".join(f"<{tag}>{render(c)}</{tag}>" for c in row) + "</tr>"
            )
            tex.append(" & ".join(map(escape, row)) + r"\\")
        page.append("</table>")
        tex.append(r"\end{tabular}}\par")
        if rows[0][0] == "event":
            page.append(
                '<figure><img src="confirmation-pairs.svg"><figcaption>Paired event differences. Each row represents a complete simulated event.</figcaption></figure>'
            )
            tex.append(
                r"\noindent\includegraphics[width=\linewidth]{confirmation-pairs.pdf}"
            )
    else:
        page.append("<p>" + render(block) + "</p>")
        tex.append(escape(block) + r"\par")
tex.extend(
    [
        r"\section*{References}",
        r"S. Amrouche et al., The Tracking Machine Learning challenge: Accuracy phase, arXiv:1904.06778.",
        r"\par X. Ai et al., A Common Tracking Software Project, arXiv:2106.13593.",
        r"\end{document}",
    ]
)
(OUT / "manuscript.tex").write_text("\n".join(tex) + "\n")
(OUT / "manuscript.html").write_text("\n".join(page) + "</main></html>\n")
