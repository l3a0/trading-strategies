/* Cloud Reader recovery snippets — run individually via
 * Control_Chrome.execute_javascript on the read.amazon.com/?asin=<ASIN> tab.
 * Each is a self-contained IIFE you can paste as the `code` argument.
 * Pages render as images, so these only NAVIGATE/MARK — you READ the text from
 * a computer-use screenshot + zoom.
 */

// 1) Dismiss the "Most Recent Page Read" dialog (click its "No").
(() => { const b = [...document.querySelectorAll('button,a,[role=button]')].find(x => (x.innerText || '').trim() === 'No'); if (b) b.click(); return JSON.stringify({ dismissed: !!b }); })()

// 2) Open the in-reader notebook (Annotations) panel.
(() => { const el = [...document.querySelectorAll('[aria-label],[title]')].find(e => (e.getAttribute('aria-label') || e.title) === 'Annotations'); if (el) el.click(); return JSON.stringify({ opened: !!el }); })()

// 3) List the notebook items' start positions, in order (line up 1:1 with the
//    scraped highlights sorted by location). Run once to build loc -> startPos.
(() => { const items = [...document.querySelectorAll('.notebook-content ion-item.notebook-editable-item')]; return JSON.stringify({ count: items.length, positions: items.map(it => Number((it.id || '').replace('notebook-grouped-item-', ''))).filter(Number.isFinite) }); })()

// 4) Navigate to a highlight (replace POS with its startPos). Also clears any prior outline.
(() => { document.querySelectorAll('[data-hloutline]').forEach(e => { e.style.outline = ''; e.removeAttribute('data-hloutline'); }); const it = document.querySelector('#notebook-grouped-item-' + POS); if (!it) return JSON.stringify({ err: 'no item' }); (it.querySelector('[data-testid=notebook-item-label]') || it).click(); return JSON.stringify({ navigated: POS }); })()

// 5) Page forward / back (chevron .click() is a no-op; synthetic ArrowRight/Left works).
//    NOTE: one key press advances a 2-page spread.
(() => { ['keydown', 'keyup'].forEach(t => document.dispatchEvent(new KeyboardEvent(t, { key: 'ArrowRight', code: 'ArrowRight', keyCode: 39, which: 39, bubbles: true }))); return JSON.stringify({ adv: 1 }); })()
(() => { ['keydown', 'keyup'].forEach(t => document.dispatchEvent(new KeyboardEvent(t, { key: 'ArrowLeft', code: 'ArrowLeft', keyCode: 37, which: 37, bubbles: true }))); return JSON.stringify({ back: 1 }); })()

// 6) Outline a highlight's EXACT extent in red (replace POS). Use when two highlights
//    are adjacent with no gap, to see precisely where one ends. Returns the start/end
//    position token; end == (next highlight's start - 1) when truly adjacent.
(() => { document.querySelectorAll('[data-hloutline]').forEach(e => { e.style.outline = ''; e.removeAttribute('data-hloutline'); }); const start = POS + '/'; const divs = [...document.querySelectorAll('.kg-client-highlight')].filter(d => [...d.classList].some(c => c.startsWith(start))); divs.forEach(d => { d.style.outline = '3px solid red'; d.setAttribute('data-hloutline', '1'); }); return JSON.stringify({ matched: divs.length, token: [...new Set(divs.flatMap(d => [...d.classList].filter(c => /^\d+\/\d+$/.test(c))))] }); })()
