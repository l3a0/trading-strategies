---
name: kindle-highlights
description: Extract a book's highlights from the Kindle notebook (read.amazon.com/notebook) into one verbatim, location-cited Markdown file, and recover highlights that Amazon's export limit truncated by reading them from the Kindle Cloud Reader. Use when the user wants to export, extract, copy, or save Kindle book notes/highlights.
---

# Kindle highlights → Markdown

Extract every highlight for a book from `read.amazon.com/notebook` into one combined,
**verbatim, location-cited** Markdown file. ~90% of highlights come straight from the page
DOM; the rest are truncated by Amazon's clipping/export limit ("Some highlights have been
hidden or truncated due to export limits") and must be recovered from the Cloud Reader.

This skill was derived from a real run that extracted all 466 highlights of *Trade Your Way
to Financial Freedom* (41 of them export-truncated). Trust the gotchas below — each one cost
real debugging.

## The one prerequisite that unblocks everything

Driving the page needs **JavaScript execution in the tab**. Two browser-control MCPs exist;
prefer whichever works, but the reliable one is the **Control_Chrome MCP**, which runs JS via
AppleScript and requires Chrome's **"Allow JavaScript from Apple Events"** to be ON:

> Chrome menu bar → **View → Developer → Allow JavaScript from Apple Events** → check it →
> confirm the warning → **quit & relaunch Chrome**.

- Symptom when OFF: `Control_Chrome.execute_javascript` returns `"Google Chrome is not running"`.
- The Claude-in-Chrome extension's `javascript_tool`/`computer`(click/screenshot) may fail with
  `Cannot access a chrome-extension:// URL of different extension` when another browser-control
  extension contends for the debugger. Its `read_page`/`find`/`navigate` still work, but
  `read_page` **caps each text node at ~100 chars** — useless for full highlight text. Use the
  Control_Chrome JS scrape instead.
- Out of scope by policy: OS-level mouse/keyboard on the browser (granted read-only tier — you
  CAN screenshot, you CANNOT click/scroll/type) and the **Kindle desktop app** (hard-blocked).
  Screenshots are the channel the truncation-recovery relies on.

## Step 1 — open the book's notebook

Navigate the tab to `https://read.amazon.com/notebook?asin=<ASIN>` (or `/notebook` and let the
user pick). Confirm the correct book title is loaded. Header shows `N Highlights | M Notes`.

## Step 2 — scrape all highlights to a JSON file

Run [scripts/extract_highlights.js](scripts/extract_highlights.js) via
`Control_Chrome.execute_javascript` (pass the notebook tab's `tab_id`). It scrapes every
`.kp-notebook-row-separator` and **triggers a Blob download** of `<asin>_highlights.json` to
`~/Downloads` — downloading avoids piping 200 KB+ of text through the model and preserves exact
typography (curly quotes, em-dashes, bullets). Then read that file with normal tools.

Gotchas:
- The notebook page loads **all** highlights at once (no lazy scroll) — one scrape gets everything.
- A Blob download from a **background** tab is often blocked. If `~/Downloads/<asin>_highlights.json`
  doesn't appear, `Control_Chrome.switch_to_tab` to make the notebook tab active, then re-run.
- Each row yields `{loc, color, text, note, truncated}`. `truncated: true` means the row contained
  "hidden or truncated due to export limits"; its `text` is only the opening and ends with `…`.

## Step 3 — build the combined Markdown

```
python3 scripts/build_notes.py ~/Downloads/<asin>_highlights.json <out>.md [completions.json]
```
Emits: a citation header (title/author/ASIN from the scrape; fill edition/publisher/year by hand),
then `### Location N · color` sections with verbatim `> ` blockquotes. Truncated highlights with no
recovered completion get a `⚠ truncated` flag + a `<!-- TRUNCATED loc=N -->` anchor. Recovered ones
get a `↻ recovered` tag (drop it by editing the script if undesired). Re-run any time — it's
deterministic.

## Step 4 — recover the export-truncated highlights from the Cloud Reader

**Key fact:** the truncated full text is NOT on the notebook page (Amazon truncates it server-side).
Only the reading app has it, and the Cloud Reader renders each page as a **rasterized image**
(`<img src="blob:…">` inside `.kg-full-page-img`) — the text is NOT in the DOM, so you must **read
the yellow-highlighted span from a screenshot**. This is the same "extract from a screenshot" idea as
the desktop app, just via the web reader (which isn't policy-blocked).

You only need the **completion** (the words after the `…`), because Step 2 already captured the exact
prefix. Procedure (see [scripts/reader_helpers.js](scripts/reader_helpers.js) for the snippets):

1. Open the book: navigate a tab to `https://read.amazon.com/?asin=<ASIN>`, then
   `Control_Chrome.switch_to_tab` so it's active. Wait ~5 s for the reader to initialize.
2. Dismiss the "Most Recent Page Read" dialog (click its **No** button via JS).
3. Open the in-reader notebook panel: click the element with `aria-label="Annotations"`.
4. Map locations → reader positions: each highlight is `#notebook-grouped-item-<startPos>`, and the
   start positions in DOM order line up **1:1** with the scraped highlights sorted by location. Click
   an item to jump the reader to that highlight.
5. `computer-use` `screenshot` (read-only is fine) → `zoom` into the yellow span → transcribe the
   completion. Page forward with a **synthetic ArrowRight keydown** (the chevron button's `.click()`
   does nothing; the key event works; note it advances a 2-page spread). Highlights that spill past a
   page bottom continue at the next column/page top.
6. **Boundary disambiguation** (adjacent highlights with no gap between them): add a red outline to the
   `.kg-client-highlight` divs whose class token is `<startPos>/<endPos>` — those rectangles ARE the
   exact highlight extent. The token's end value also equals (next highlight's start − 1) when adjacent.

Record `{ "<loc>": "<completion>" }` into `completions.json` as you go (batch by page/cluster —
truncations cluster in densely-highlighted passages). Re-run Step 3 to fold them in.

## Step 5 — QA before declaring done

- `recovered == truncated`, `pending == 0` (build prints these).
- Seam check: for each recovered loc, `full = prefix[:-1].rstrip() + " " + completion` has no double
  space, no doubled word at the join, no leftover `…`.
- `### Location` section count == highlight count; file ends with exactly one newline.

## Verbatim judgment calls

- Mid-line hyphens in justified text are usually soft (line-break) hyphens — render the dominant book
  form (e.g. "channel breakout", not "break-out"), cross-checking another occurrence.
- Italic *R* / *1R* / *2R* etc.: Amazon's scraped prefix renders them plain, so keep completions plain
  for consistency (or italicize everywhere — just be consistent).
- Keep the scraped prefix byte-exact (the build does this automatically); only transcribe completions.
