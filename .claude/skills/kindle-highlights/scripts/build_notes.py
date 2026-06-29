#!/usr/bin/env python3
"""Build one combined, location-cited Markdown file from a Kindle notebook scrape.

Usage:
    python3 build_notes.py <highlights.json> <out.md> [completions.json]

- <highlights.json> is produced by extract_highlights.js (has .book and .highlights).
- <completions.json> (optional) maps "<loc>" -> the text AFTER the "…" for each
  export-truncated highlight (recovered from the Cloud Reader). The full text is
  reconstructed as  exact-prefix(minus "…") + " " + completion , so the scraped
  prefix stays byte-exact and you only transcribe short completions.

Fill EDITION / PUBLISHER / YEAR below for a complete citation (not on the notebook page).
Re-run any time; it's deterministic.
"""
import json
import os
import sys

EDITION = ""    # e.g. "2nd ed."
PUBLISHER = ""  # e.g. "McGraw-Hill"
YEAR = ""       # e.g. "2007"
RECOVERED_TAG = True  # set False to make recovered entries look identical to the rest


def reconstruct(text, completion):
    base = text[:-1].rstrip() if text.endswith("…") else text.rstrip()
    return base + " " + completion


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    comp_path = sys.argv[3] if len(sys.argv) > 3 else None

    payload = json.load(open(src))
    book = payload.get("book", {})
    data = list(payload["highlights"])
    completions = {}
    if comp_path and os.path.exists(comp_path):
        completions = {int(k): v for k, v in json.load(open(comp_path)).items()}

    data.sort(key=lambda x: (x["loc"] is None, x["loc"]))
    n = len(data)
    n_trunc = sum(1 for x in data if x["truncated"])
    n_rec = sum(1 for x in data if x["truncated"] and x["loc"] in completions)
    n_pending = n_trunc - n_rec

    colors = sorted({x["color"] for x in data if x["color"]})
    n_notes = sum(1 for x in data if x.get("note"))

    title = book.get("title") or "UNKNOWN TITLE"
    author = book.get("author") or "UNKNOWN AUTHOR"
    asin = book.get("asin") or ""
    cite_bits = ", ".join(b for b in [EDITION, PUBLISHER, YEAR] if b)
    cite = f"{author}, *{title}*" + (f", {cite_bits}" if cite_bits else "") + ", Kindle."

    color_note = ("all highlights " + colors[0]) if len(colors) == 1 else f"colors: {', '.join(colors)}"
    pend_note = f" ({n_pending} still pending, flagged ⚠ truncated)" if n_pending else ""
    trunc_note = (
        f" {n_trunc} of these were cut off by Amazon's export limit on the notebook page; "
        f"their full text was recovered from the Kindle Cloud Reader"
        + (" and is marked with a `↻` tag" if RECOVERED_TAG else "")
        + f"{pend_note}."
    ) if n_trunc else ""

    L = [f"# {title} — Kindle Highlights", ""]
    L.append(f"**{author}**" + (f" · {cite_bits}" if cite_bits else "") + (f" · Kindle (ASIN {asin})" if asin else " · Kindle"))
    L.append("")
    L.append(f"{n} highlights · {n_notes} notes · {color_note}. Quoted verbatim and located by Kindle location number.{trunc_note}")
    L.append("")
    L.append(f"> **Full citation:** {cite} Each highlight below is cited by its Kindle location.")
    L.append("")
    L.append("---")
    L.append("")

    for x in data:
        loc, color = x["loc"], x["color"]
        is_rec = x["truncated"] and loc in completions
        still_trunc = x["truncated"] and loc not in completions
        head = f"### Location {loc} · {color}"
        if is_rec and RECOVERED_TAG:
            head += " · ↻ recovered"
        elif still_trunc:
            head += " · ⚠ truncated"
        L.append(head)
        L.append("")
        if is_rec:
            L.append("<!-- full text recovered from Kindle Cloud Reader (export-limited on notebook page) -->")
            text = reconstruct(x["text"], completions[loc])
        elif still_trunc:
            L.append(f"<!-- TRUNCATED loc={loc} full_text_pending -->")
            text = x["text"]
        else:
            text = x["text"]
        quote = "> " + text
        if x.get("note"):
            quote += f"\n>\n> **Note:** {x['note']}"
        if still_trunc:
            quote += " `[truncated by Amazon export limit — full text pending]`"
        L.append(quote)
        L.append("")

    content = "\n".join(L).rstrip() + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    open(out, "w").write(content)
    print(f"wrote {out}")
    print(f"total={n} truncated={n_trunc} recovered={n_rec} pending={n_pending}")


if __name__ == "__main__":
    main()
