#!/usr/bin/env python3
"""Build the Almanac's banjo plate into public/banjo-plate/ from the composer's studio.

  uv run scripts/banjo/build.py [--studio ~/dev/folk-variations]

The plate (`scripts/models/banjo_fit/film.html` in the studio) is the drawn banjo that
films its own fit to recordings. This copies a public cut of it: the model's audio at
steps 0, 25, 100 and 600, with the reference recordings left out (the Philharmonia
samples may not be made available as they are; the plate still draws their spectra).
The page is pinned to its light palette and a transparent table so it sits on the
Almanac's paper inside an iframe, and its "about this fit" card gets public wording.
The étude is the fitted banjo, dry, playing a short articulation study.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

SITE = Path(__file__).resolve().parents[2]
OUT = SITE / "public/banjo-plate"
STEPS = ["0", "25", "100", "600"]

SOURCE = {
    "notes_source": "Philharmonia Orchestra sound samples (banjo)",
    "caveats": [
        "The room, microphone, background hiss and MP3 stages are whatever helps the fit match; they aren't a measurement of the real room or microphone.",
        "These are the nine notes the fit learned from. Notes it never heard score about as well, but aren't drawn here.",
        "The dry switch plays the banjo alone, without the recording chain.",
        "Each note is played at the same loudness, so the plate compares sound, not level.",
        "The recordings themselves aren't reproduced here: their licence doesn't allow sharing them as they are. The blue curve is their spectrum.",
    ],
}
STAGE = "The banjo and the recording chain it is heard through are fitted together, starting from a chain that does nothing."

EMBED_STYLE = """<style>
/* inside the Almanac page: the plate lies on the page's own paper */
body { background: transparent; }
.sheet { min-height: 0; justify-content: flex-start; }
</style>
</head>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--studio", type=Path, default=Path.home() / "dev/folk-variations")
    a = ap.parse_args()
    fit = a.studio / "scripts/models/banjo_fit"
    if OUT.exists():
        shutil.rmtree(OUT)
    subprocess.run(["python3", fit / "plate.py", "--public", a.studio / "build/banjo-allway/plate", OUT, *STEPS], check=True)

    traj = OUT / "trajectory.json"
    d = json.loads(traj.read_text())
    d["source"] = SOURCE
    d["stages"][0]["what"] = STAGE
    d["plate"] = {"steps": [int(s) for s in STEPS]}
    traj.write_text(json.dumps(d, separators=(",", ":"), allow_nan=False) + "\n")

    page = (fit / "film.html").read_text()
    for old, new in (('<html lang="en">', '<html lang="en" data-theme="light">'),
                     ("<!--pages-->", ""),
                     ("</head>", EMBED_STYLE)):
        assert page.count(old) == 1, old
        page = page.replace(old, new)
    (OUT / "index.html").write_text(page)

    shutil.copy2(a.studio / "build/banjo-allway/etude-reviewed-final/audio/allway-final-dry.mp3", OUT / "etude.mp3")


if __name__ == "__main__":
    main()
