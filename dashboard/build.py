"""Assemble the dashboard: page.html + engine.js + app.js + dashboard_data.json (parity samples dropped).

Writes seller_vs_holder.html (artifact body, no document skeleton) and seller_vs_holder_standalone.html
(a complete HTML document to open locally in any browser).
"""
import json
from pathlib import Path

D = Path(__file__).resolve().parent
data = json.loads((D / "dashboard_data.json").read_text())
data.pop("parity", None)
page = (D / "page.html").read_text()
body = (page.replace("/*__ENGINE__*/", (D / "engine.js").read_text())
            .replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
            .replace("/*__APP__*/", (D / "app.js").read_text())
            .replace("/*__MATH__*/", (D / "math.html").read_text()))
(D / "seller_vs_holder.html").write_text(body)
head, rest = body.split("<style>", 1)
standalone = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
              "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
              + head + "<style>\nbody{margin:0}\n" + rest.split("</style>", 1)[0] + "</style>\n</head>\n<body>\n"
              + rest.split("</style>", 1)[1] + "\n</body>\n</html>\n")
(D / "seller_vs_holder_standalone.html").write_text(standalone)
for f in ("seller_vs_holder.html", "seller_vs_holder_standalone.html"):
    print(f, f"{(D / f).stat().st_size / 1e6:.2f} MB")
