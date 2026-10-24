"""Build public/music/era/index.html: the Era album page.

The record's measurements (floor.json), its cuts (tracks.json) and the liner notes
(liner.json) come from the record's own folder in folk-variations; page.html beside
this script is the page, with @@NAME@@ slots for what this script fills in.

    python3 scripts/era/build.py [path/to/score/era/cover]
"""
import base64
import html
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = HERE.parent.parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "dev/folk-variations/score/era/cover"
OUT = SITE / "public/music/era/index.html"


def esc(s):
    return html.escape(s, quote=False)


def clock(seconds):
    s = int(seconds + 0.5)
    return f"{s // 60}:{s % 60:02d}"


def floor_bytes(rows):
    """Three bytes per quarter second: loudness, brightness, and pitch class with its share."""
    out = bytearray()
    for db, centroid, _flux, pc, share in rows:
        level = max(0, min(255, round((db + 72) / 66 * 255)))
        bright = 0 if centroid <= 0 else max(1, min(255, round(math.log(centroid / 150) / math.log(4000 / 150) * 255)))
        out += bytes([level, bright, (pc << 4) | max(0, min(15, round(share * 15)))])
    return base64.b64encode(bytes(out)).decode()


def words_html(blocks):
    parts = []
    for block in blocks:
        if len(block) == 1 and ": " in block[0]:
            parts.append(f'<p class="direction">{esc(block[0])}</p>')
        elif len(block) == 1:
            parts.append(f'<h4>{esc(block[0])}</h4>')
        else:
            head, lines = (block[0], block[1:]) if block[0].split()[0].isupper() else (None, block)
            if head:
                parts.append(f'<h4>{esc(head)}</h4>')
            parts.append('<p class="stanza">' + "<br>".join(esc(l) for l in lines) + "</p>")
    return '<div class="words">' + "\n".join(parts) + "</div>"


def build():
    liner = json.loads((SRC / "liner.json").read_text())
    floor = json.loads((SRC / "floor.json").read_text())
    cuts = json.loads((SRC / "tracks.json").read_text())
    rate = cuts["rate"]
    notes = liner["tracks"]
    durations = [t["frames"] / rate for t in cuts["tracks"]]

    tracks_js = [{"url": c["url"], "frames": c["frames"], "skip": c["skip"], "title": n["title"]}
                 for c, n in zip(cuts["tracks"], notes)]

    tracklist = "\n".join(
        f'<li data-i="{i}"><span class="n">{n["n"]}</span>'
        f'<button type="button" class="go">{esc(n["title"])}</button>'
        f'<span class="dur">{clock(d)}</span>'
        f'<a class="to-note" href="#t{n["n"]}" aria-label="Notes on {esc(n["title"])}">¶</a></li>'
        for i, (n, d) in enumerate(zip(notes, durations)))

    intro = "\n".join(f"<p>{esc(p)}</p>" for p in liner["intro"])

    articles = []
    for i, (n, d) in enumerate(zip(notes, durations)):
        sub = f' <span class="sub">{esc(n["subtitle"])}</span>' if n.get("subtitle") else ""
        words = words_html(liner["words"][n["words"]]) if n.get("words") else ""
        articles.append(
            f'<article class="note" id="t{n["n"]}" data-i="{i}">'
            f'<header><span class="n">{n["n"]}</span><h3>{esc(n["title"])}{sub}</h3>'
            f'<span class="meta"><span class="kind">{esc(n["kind"])}</span> · {clock(d)}</span>'
            f'<button type="button" class="go" aria-label="Play {esc(n["title"])}">play</button></header>'
            f'<p>{esc(n["note"])}</p>{words}</article>')

    lineage = "\n".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in liner["lineage"])

    page = (HERE / "page.html").read_text()
    slots = {
        "LENGTH": liner["length"],
        "YEAR": str(liner["year"]),
        "TRACKLIST": tracklist,
        "INTRO": intro,
        "NOTES": "\n".join(articles),
        "LINEAGE": lineage,
        "COLOPHON": esc(liner["cover"]),
        "TRACKS": json.dumps({"rate": rate, "tracks": tracks_js}, ensure_ascii=False),
        "FLOOR": floor_bytes(floor["rows"]),
        "TOTAL": str(int(sum(durations))),
    }
    for k, v in slots.items():
        page = page.replace(f"@@{k}@@", v)
    assert "@@" not in page, page[page.index("@@") - 40: page.index("@@") + 40]
    OUT.write_text(page)
    print(f"wrote {OUT.relative_to(SITE)} ({len(page) // 1024} kB)")


if __name__ == "__main__":
    build()
