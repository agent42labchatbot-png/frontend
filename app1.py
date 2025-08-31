import io
import json
import logging
import re
import secrets
import time
import hashlib
from flask import Flask, request, jsonify, Response, send_file, abort
from flask_cors import CORS
import requests
import cohere
import bleach
from rank_bm25 import BM25Okapi
from slugify import slugify
from werkzeug.utils import secure_filename

# ============ Configuration ============
COHERE_API_KEY = "h9BaM1K7lspKN7FAFe1uLr2pJb1I2s2ba1T5mSZW"
WP_BASE = "https://agent42labs.com"
BRAND_NAME = "Agent42 Labs"
ALLOWED_ORIGINS = ["*"]  # Set specific domains in production
BACKEND_API_KEY = ""     # Set if backend authentication is required

RERANK_MODEL = "rerank-english-v3.0"
FINAL_DOCS = 6
RERANK_THRESHOLD = 0.20

CONTACT_EMAIL = "hello@agent42labs.com"
CONTACT_PHONE = "+91 7027119799"
CONTACT_URL = "https://agent42labs.com/contact-us"

PAGES_TTL_SEC = 86400  # 24h cache for stored pages
ANSWER_TTL_SEC = 43200 # 12h cache for answers
PORT = 8080

ALLOWED_TAGS = [
    "article", "section", "header", "footer", "nav",
    "h1", "h2", "h3", "h4", "p", "ul", "ol", "li",
    "a", "img", "figure", "figcaption", "blockquote",
    "code", "pre", "em", "strong", "hr", "br", "div", "span",
    "table", "thead", "tbody", "tr", "th", "td", "sup"
]

ALLOWED_ATTRS = {
    "a": ["href", "title", "target", "rel"],
    "img": ["src", "alt", "width", "height", "loading", "decoding", "sizes", "srcset"],
    "*": ["class", "id", "data-*"]
}

# ============ Caches ============
TTL = 300  # Cache TTL (5 minutes)
POSTS_CACHE = {"ts": 0, "docs": []}
PAGES_CACHE = {"ts": 0, "docs": []}
MEDIA_CACHE = {"ts": 0, "imgs": []}

PAGES = {}        # page_id -> {title, html, ts}
ANSWER_CACHE = {} # cache_key -> {page_id, ts}

# ============ Flask App ============
app = Flask(__name__)
CORS(app, resources={r"/compose": {"origins": ALLOWED_ORIGINS or "*"}})
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("answer-backend")
co = cohere.Client(COHERE_API_KEY)
http = requests.Session()

# ============ Utility Functions ============
def trace_id_for(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def sanitize_html(html_str: str) -> str:
    return bleach.clean(html_str, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)

def _strip_html(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html or "").strip()

def chunk_text(text: str, max_chars=1200, overlap=120):
    text = re.sub(r"\s+\n", "\n", text or "")
    chunks, i, n = [], 0, len(text)
    while i < n:
        end = min(i + max_chars, n)
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        i = max(end - overlap, end)
    return chunks

def _tok(text: str):
    return re.findall(r"[a-z0-9]+", (text or "").lower())

def linkify_citations(html_str: str, citations: list) -> str:
    if not citations:
        return html_str
    out = html_str
    for i in range(1, len(citations) + 1):
        out = out.replace(f'[{i}]', f'<sup class="cite"><a href="#src-{i}">{i}</a></sup>')
    return out

def dedupe_citations(items: list) -> list:
    seen = set()
    out = []
    for c in items or []:
        key = (c.get("url") or c.get("title") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out

def _new_id() -> str:
    return secrets.token_urlsafe(10)

def _cleanup_pages():
    now = time.time()
    to_del = [pid for pid, p in PAGES.items() if now - p["ts"] > PAGES_TTL_SEC]
    for pid in to_del:
        del PAGES[pid]

def _cleanup_answers():
    now = time.time()
    to_del = [k for k, v in ANSWER_CACHE.items() if now - v["ts"] > ANSWER_TTL_SEC]
    for k in to_del:
        del ANSWER_CACHE[k]

def _save_page(title: str, html: str) -> str:
    _cleanup_pages()
    pid = _new_id()
    PAGES[pid] = {"title": title, "html": html, "ts": time.time()}
    return pid

def _cache_key(question: str, layout: str, include_citations: bool, brand_class: str, primary: str) -> str:
    norm = re.sub(r"\s+", " ", (question or "").strip().lower())
    key = f"{norm}|{layout}|{include_citations}|{brand_class}|{primary}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]

# ============ WordPress Data Fetchers ============
def fetch_wp_posts(per_page=50, max_pages=20):
    now = time.time()
    if POSTS_CACHE["docs"] and now - POSTS_CACHE["ts"] < TTL:
        return POSTS_CACHE["docs"]
    out = []
    for page in range(1, max_pages + 1):
        try:
            r = http.get(f"{WP_BASE}/wp-json/wp/v2/posts",
                         params={"per_page": per_page, "page": page, "status": "publish", "_fields": "id,link,title,content"},
                         timeout=20)
            if r.status_code == 400 and "rest_post_invalid_page_number" in r.text:
                break
            r.raise_for_status()
            items = r.json() or []
            if not items:
                break
            for p in items:
                pid = p.get("id")
                title = _strip_html((p.get("title") or {}).get("rendered", "") or "Untitled")
                link = p.get("link") or ""
                content = _strip_html((p.get("content") or {}).get("rendered", ""))
                for ch in chunk_text(content):
                    out.append({"post_id": pid, "title": title, "url": link, "text_chunk": ch})
            if len(items) < per_page:
                break
        except Exception as e:
            logger.error(f"Error fetching WP posts: {e}")
            break
    POSTS_CACHE["ts"] = now
    POSTS_CACHE["docs"] = out
    return out

def fetch_wp_pages(per_page=50, max_pages=10):
    now = time.time()
    if PAGES_CACHE["docs"] and now - PAGES_CACHE["ts"] < TTL:
        return PAGES_CACHE["docs"]
    out = []
    for page in range(1, max_pages + 1):
        try:
            r = http.get(f"{WP_BASE}/wp-json/wp/v2/pages",
                         params={"per_page": per_page, "page": page, "_fields": "id,link,title,content"},
                         timeout=20)
            if r.status_code == 400 and "rest_post_invalid_page_number" in r.text:
                break
            r.raise_for_status()
            items = r.json() or []
            if not items:
                break
            for p in items:
                pid = p.get("id")
                title = _strip_html((p.get("title") or {}).get("rendered", "") or "Untitled")
                link = p.get("link") or ""
                content = _strip_html((p.get("content") or {}).get("rendered", ""))
                for ch in chunk_text(content):
                    out.append({"post_id": pid, "title": title, "url": link, "text_chunk": ch})
            if len(items) < per_page:
                break
        except Exception as e:
            logger.error(f"Error fetching WP pages: {e}")
            break
    PAGES_CACHE["ts"] = now
    PAGES_CACHE["docs"] = out
    return out

def get_all_docs():
    return fetch_wp_posts() + fetch_wp_pages()

def fetch_wp_media(per_page=80, max_pages=10):
    now = time.time()
    if MEDIA_CACHE["imgs"] and now - MEDIA_CACHE["ts"] < TTL:
        return MEDIA_CACHE["imgs"]
    imgs = []
    for page in range(1, max_pages + 1):
        try:
            r = http.get(f"{WP_BASE}/wp-json/wp/v2/media",
                         params={"per_page": per_page, "page": page,
                                 "_fields": "id,alt_text,caption,media_type,media_details,source_url,title"},
                         timeout=20)
            if r.status_code == 400 and "rest_post_invalid_page_number" in r.text:
                break
            r.raise_for_status()
            items = r.json() or []
            if not items:
                break
            for m in items:
                if m.get("media_type") != "image":
                    continue
                details = m.get("media_details") or {}
                sizes = details.get("sizes") or {}
                parts = []
                for s in sizes.values():
                    u, w = s.get("source_url"), s.get("width")
                    if u and w:
                        parts.append((int(w), f"{u} {w}w"))
                parts.sort(key=lambda x: x[0])
                srcset = ", ".join(p[1] for p in parts)
                imgs.append({
                    "id": f"wp_{m['id']}",
                    "url": m.get("source_url"),
                    "title": (m.get("title") or {}).get("rendered", ""),
                    "alt": m.get("alt_text") or "",
                    "caption": _strip_html((m.get("caption") or {}).get("rendered", "")),
                    "width": details.get("width"),
                    "height": details.get("height"),
                    "srcset": srcset,
                    "sizes": "100vw",
                })
            if len(items) < per_page:
                break
        except Exception as e:
            logger.warning(f"attachments fetch failed for media page {page}: {e}")
            break
    MEDIA_CACHE["ts"] = now
    MEDIA_CACHE["imgs"] = imgs
    return imgs

def get_post_attachments(post_ids, k=3):
    if not post_ids:
        return []
    imgs = []
    for pid in post_ids:
        try:
            r = http.get(f"{WP_BASE}/wp-json/wp/v2/media",
                         params={"parent": pid, "per_page": 20,
                                 "_fields": "id,alt_text,caption,media_type,media_details,source_url,title"},
                         timeout=15)
            r.raise_for_status()
            media_items = r.json() or []
            for m in media_items:
                if m.get("media_type") != "image":
                    continue
                details = m.get("media_details") or {}
                sizes = details.get("sizes") or {}
                parts = []
                for s in sizes.values():
                    u, w = s.get("source_url"), s.get("width")
                    if u and w:
                        parts.append((int(w), f"{u} {w}w"))
                parts.sort(key=lambda x: x[0])
                srcset = ", ".join(p[1] for p in parts)
                imgs.append({
                    "id": f"wp_{m['id']}",
                    "url": m.get("source_url"),
                    "title": (m.get("title") or {}).get("rendered", ""),
                    "alt": m.get("alt_text") or "",
                    "caption": _strip_html((m.get("caption") or {}).get("rendered", "")),
                    "width": details.get("width"),
                    "height": details.get("height"),
                    "srcset": srcset,
                    "sizes": "100vw",
                })
        except Exception as e:
            logger.warning(f"attachments fetch failed for post {pid}: {e}")
    return imgs[:k] if len(imgs) > k else imgs

# ============ Search and Ranking ============

def search_docs(query: str, k: int=6):
    docs = get_all_docs()
    if not docs:
        return []
    corpus = [d["text_chunk"] for d in docs]
    bm25 = BM25Okapi([_tok(t) for t in corpus])
    scores = bm25.get_scores(_tok(query))
    M = max(k*5, k)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:min(M,len(scores))]
    candidates = [docs[i] for i in top_idx] or docs[:M]
    try:
        rr = co.rerank(model=RERANK_MODEL, query=query,
                       documents=[c["text_chunk"] for c in candidates],
                       top_n=min(k, len(candidates)))
        if rr.results:
            return [candidates[r.index] for r in rr.results]
        return candidates[:k]
    except Exception as e:
        logger.warning(f"rerank failed: {e}")
        return candidates[:k]

# ============ Planning and Rendering ============

def build_planner_prompt(question: str, sources: list, layout: str, include_citations: bool) -> str:
    ctx_lines = []
    for i, s in enumerate(sources, 1):
        t = s.get("text_chunk","")
        if len(t) > 1800:
            t = t[:1800]
        ctx_lines.append(f"[{i}] {s.get('title','Untitled')} — {s.get('url','')}\n{t}")
    return (
        "You are a content planner. Output ONE JSON object ONLY (no markdown, no extra text).\n"
        "Use only facts from the sources. Every factual sentence must end with a [n] citation that matches a numbered source.\n"
        "If a claim is not supported by the sources, omit it.\n"
        f"Layout: {layout}\n"
        "Schema: { \"title\": str, \"summary\": str, \"show_toc\": bool, \"sections\": [ {\"id\": str, \"heading\": str, \"paragraphs\": [str], \"bullets\": [str]} ] }\n\n"
        f"User question:\n{question}\n\n"
        "Relevant sources (cite as [n]):\n" + "\n\n".join(ctx_lines)
    )

def cohere_plan(prompt: str) -> dict:
    resp = co.generate(model="command-r-plus", prompt=prompt, max_tokens=900, temperature=0.2)
    txt = (resp.generations[0].text or "").strip()
    try:
        return json.loads(txt)
    except Exception:
        start, end = txt.find("{"), txt.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(txt[start:end+1])
        raise ValueError("Planner did not return valid JSON")

def render_article(plan: dict, citations: list, brand_class: str, include_citations: bool, images: list | None = None) -> str:
    images = images or []
    hero = images[0] if images else None
   # gallery = images[1:] if images and len(images) > 1 else []

    parts = []

    parts.append(f'''
    <article class="{brand_class}">
      <header class="{brand_class}__header" role="banner">
        <h1 class="{brand_class}__title">{plan.get("title", "")}</h1>
        <p class="{brand_class}__summary" aria-live="polite">{plan.get("summary", "")}</p>
      </header>
    ''')

    if hero:
        parts.append(f'''
        <figure class="{brand_class}__hero" role="img" aria-label="{hero.get('alt', '')}">
          <img src="{hero.get('url', '')}" alt="{hero.get('alt', '')}" loading="lazy" decoding="async" class="{brand_class}__hero-image"/>
          <figcaption class="{brand_class}__hero-caption">{hero.get('caption', '')}</figcaption>
        </figure>
        ''')

    if plan.get("show_toc"):
        parts.append(f'''
        <nav class="{brand_class}__toc" aria-label="Table of contents">
          <p class="{brand_class}__toc-title">Contents</p>
          <ul class="{brand_class}__toc-list">
        ''')
        for section in plan.get("sections") or []:
            parts.append(f'<li><a href="#{section.get("id","")}" class="{brand_class}__toc-link">{section.get("heading","")}</a></li>')
        parts.append('</ul></nav>')

    for section in plan.get("sections") or []:
        parts.append(f'<section id="{section.get("id","")}" class="{brand_class}__section" tabindex="-1">')
        parts.append(f'<h2 class="{brand_class}__section-title">{section.get("heading","")}</h2>')

        for para in section.get("paragraphs") or []:
            if para.strip().startswith(">"):
                parts.append(f'<blockquote class="{brand_class}__blockquote">{para.strip()[1:].strip()}</blockquote>')
            else:
                parts.append(f'<p class="{brand_class}__paragraph">{para}</p>')

        bullets = section.get("bullets") or []
        if bullets:
            parts.append(f'<ul class="{brand_class}__list">')
            for b in bullets:
                parts.append(f'<li>{b}</li>')
            parts.append('</ul>')
        parts.append('</section>')

    # if gallery:
    #     parts.append(f'''
    #     <section class="{brand_class}__gallery" aria-label="Image gallery">
    #       <h2 class="{brand_class}__gallery-title">Gallery</h2>
    #       <div class="{brand_class}__gallery-grid">
    #     ''')
    #     for img in gallery:
    #         parts.append(f'''
    #         <figure class="{brand_class}__gallery-item" role="group" aria-label="{img.get('alt', '')}">
    #           <img src="{img.get("url")}" alt="{img.get("alt")}" loading="lazy" decoding="async" class="{brand_class}__gallery-image"/>
    #           <figcaption class="{brand_class}__gallery-caption">{img.get("caption", "")}</figcaption>
    #         </figure>
    #         ''')
    #     parts.append('</div></section>')

    if include_citations and citations:
        parts.append(f'''
        <footer class="{brand_class}__sources" aria-label="Content sources">
          <h2 class="{brand_class}__sources-title">Sources</h2>
          <ol class="{brand_class}__sources-list">
        ''')
        for i, c in enumerate(citations, 1):
            url = c.get("url") or "#"
            title = c.get("title") or url
            parts.append(f'<li id="src-{i}"><a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a></li>')
        parts.append('</ol></footer>')

    parts.append('</article>')

    return ''.join(parts)

def build_fullpage_html(title: str, article_html: str, brand_class: str, primary: str, trace_id: str, page_id: str, base_url: str,
                        contact_email=CONTACT_EMAIL, contact_phone=CONTACT_PHONE, contact_url=CONTACT_URL) -> str:
    canonical = f"{base_url}/v/{page_id}"
    download_url = f"{canonical}/download"
    phone_part = f" · Phone: {contact_phone}" if contact_phone else ""
    safe_title = slugify(title or "article")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <meta name="description" content="Detailed article page for {title}" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="x-trace-id" content="{trace_id}" />
  <link rel="canonical" href="{canonical}" />
  <meta property="og:title" content="{title}" />
  <meta property="og:type" content="article" />
  <meta property="og:url" content="{canonical}" />
  <meta name="theme-color" content="{primary}" />
  <style>
    :root {{
      --primary: {primary};
      --text: #0f172a;
      --muted: #64748b;
      --bg: #f3f6fb;
      --card-bg: rgba(255, 255, 255, 0.3);
      --border: rgba(15, 23, 42, 0.08);
      --shadow-default: 0 14px 40px rgba(15, 23, 42, 0.08);
      --shadow-glass: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
      --pill-radius: 9999px;
      --card-radius: 16px;
      --max-width: 980px;
      --font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Inter, Arial, sans-serif;
    }}
    html, body {{
      margin: 0; padding: 0;
      background: var(--bg);
      color: var(--text);
      font-family: var(--font-family);
      scroll-behavior: smooth;
      min-height: 100vh;
    }}

    /* Removed top header for buttons */

    .wrap {{
      margin: 18px auto 40px;
      max-width: var(--max-width);
      padding: 0 16px;
      min-height: calc(100vh - 150px);
      display: flex;
      justify-content: center;
      align-items: flex-start;
    }}

    .card {{
      background: var(--card-bg);
      border-radius: var(--card-radius);
      box-shadow: var(--shadow-glass);
      border: 1px solid rgba(255, 255, 255, 0.18);
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      padding: clamp(24px, 3vw, 32px);
      width: 100%;
      max-width: var(--max-width);
      display: flex;
      flex-direction: column;
      box-sizing: border-box;
    }}

    /* Moved action buttons inside card, style them */
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      justify-content: flex-end;
      margin-bottom: 1.5rem;
      min-width: 200px;
    }}
    @media (max-width: 480px) {{
      .actions {{
        flex-direction: column;
        align-items: stretch;
        gap: 6px;
        min-width: 100%;
      }}
    }}

    .btn {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-weight: 700;
      padding: 9px 12px;
      border-radius: var(--pill-radius);
      border: 1px solid var(--border);
      background: #eef2f7;
      color: var(--text);
      cursor: pointer;
      text-decoration: none;
      transition: transform 0.08s ease, filter 0.2s ease;
      font-size: 0.9rem;
      user-select: none;
    }}
    .btn:hover {{
      filter: brightness(1.02);
    }}
    .btn:active {{
      transform: translateY(1px);
    }}
    .btn:focus {{
      outline: 2px solid var(--primary);
      outline-offset: 2px;
      outline-style: solid;
    }}
    .btn.primary {{
      background: var(--primary);
      color: #fff;
      border-color: transparent;
      box-shadow: 0 12px 26px rgba(34,197,94,0.25);
    }}

    /* Brand chip and contact styling */
    .brand {{
      display: flex;
      align-items: center;
      gap: 12px;
      margin-top: 2rem;
      user-select: none;
    }}
    .chip {{
      width: 36px;
      height: 36px;
      display: grid;
      place-items: center;
      border-radius: 50%;
      background: var(--primary);
      color: #fff;
      font-weight: 900;
      font-size: 18px;
      box-shadow: 0 8px 20px rgba(34, 197, 94, 0.35);
      flex-shrink: 0;
    }}
    .brand-name {{
      font-weight: 600;
      font-size: 1.125rem;
      color: #3b3c4a;
      user-select: text;
    }}
    .contact {{
      margin-top: 1.25rem;
      font-size: 1rem;
      color: var(--muted, #64748b);
      line-height: 1.5;
    }}
    .contact strong {{
      display: block;
      margin-bottom: 0.4rem;
      color: var(--text);
      font-weight: 600;
    }}
    .contact a {{
      color: var(--primary);
      text-decoration: none;
      transition: color 0.2s ease;
    }}
    .contact a:hover {{
      text-decoration: underline;
    }}

    /* Spacing between paragraphs and lists */
    .{brand_class} p,
    .{brand_class} li,
    .{brand_class} ul,
    .{brand_class} ol {{
      margin-bottom: 0.85em;
    }}

    .{brand_class} ul, .{brand_class} ol {{
      margin-top: 0.5em;
      padding-left: 2em;
    }}

    .{brand_class} li {{
      line-height: 1.8;
    }}

    @media print {{
      .actions {{
        display: none !important;
      }}
      body {{
        background: #fff;
      }}
      .card {{
        box-shadow: none;
        border: none;
        padding: 0;
        backdrop-filter: none;
      }}
      a {{
        color: #000;
        text-decoration: underline;
      }}
    }}
  </style>
</head>
<body>

  <main class="wrap" role="main">
    <div class="card {brand_class}">

      <!-- Actions moved inside card -->
      <nav class="actions" aria-label="Page actions">
        <a class="btn primary" href="{download_url}" download="{safe_title}.html" aria-label="Download article as HTML">
          <svg width="16" height="16" viewBox="0 0 24 24" stroke="currentColor" fill="none" stroke-width="2" aria-hidden="true">
            <path d="M12 3v12"/>
            <path d="M7 10l5 5 5-5"/>
            <path d="M5 21h14"/>
          </svg>
          Download .html
        </a>
        <button class="btn" id="print" aria-label="Print or save as PDF" title="Print or Save as PDF">
          <svg width="16" height="16" viewBox="0 0 24 24" stroke="currentColor" fill="none" stroke-width="2" aria-hidden="true">
            <path d="M6 9V2h12v7"/>
            <path d="M6 18h12v4H6z"/>
            <path d="M6 14h12"/>
          </svg>
          Print / Save as PDF
        </button>
        <button class="btn" id="copy" aria-label="Copy current page link" title="Copy Link">
          <svg width="16" height="16" viewBox="0 0 24 24" stroke="currentColor" fill="none" stroke-width="2" aria-hidden="true">
            <rect x="9" y="9" width="13" height="13" rx="2"/>
            <rect x="2" y="2" width="13" height="13" rx="2"/>
          </svg>
          Copy link
        </button>
      </nav>

      {article_html}

      <div class="brand" aria-label="Brand identity">
        <div class="chip" aria-label="Brand initial">{brand_class[:1].upper()}</div>
        <div class="brand-name">{brand_class}</div>
      </div>
      <div class="contact" aria-label="Contact information">
        <strong>Contact us</strong>
        Email: <a href="mailto:{contact_email}">{contact_email}</a> · Phone: {contact_phone} ·
        <a href="{contact_url}" target="_blank" rel="noopener noreferrer">Contact page</a>
      </div>
    </div>
  </main>
  <script>
    document.getElementById('print').onclick = () => window.print();
    document.getElementById('copy').onclick = async () => {{
      try {{
        await navigator.clipboard.writeText(window.location.href);
        const btn = document.getElementById('copy');
        const originalText = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => btn.textContent = originalText, 1000);
      }} catch (e) {{
        console.warn('Clipboard write failed', e);
      }}
    }};
  </script>
</body>
</html>
"""


# ============ Compose Pipeline ============

def compose_answer_page(question: str, layout: str, include_citations: bool, brand_class: str, primary: str, base_url: str):
    trace_id = trace_id_for(question)

    _cleanup_answers()
    ck = _cache_key(question, layout, include_citations, brand_class, primary)
    cached = ANSWER_CACHE.get(ck)
    if cached and cached.get("page_id") in PAGES:
        pid = cached["page_id"]
        return {"id": pid, "title": PAGES[pid]["title"], "trace_id": trace_id}

    # Retrieve and rank docs
    doc_hits = search_docs(question, FINAL_DOCS)
    if not doc_hits:
        plan = {
            "title": "No sources available",
            "summary": "",
            "show_toc": False,
            "sections": [{
                "id": "s1",
                "heading": "Please try again",
                "paragraphs": ["We couldn't find any relevant sources from the site. Try rephrasing your question."],
                "bullets": []
            }]
        }
        article_html = render_article(plan, [], brand_class, False, images=[])
        safe_article = sanitize_html(article_html)
        safe_article = linkify_citations(safe_article, [])
        title = plan["title"]
        pid = _save_page(title, build_fullpage_html(title, safe_article, brand_class, primary, trace_id, page_id="tmp", base_url=base_url))
        full_html = build_fullpage_html(title, safe_article, brand_class, primary, trace_id, pid, base_url)
        PAGES[pid]["html"] = full_html
        ANSWER_CACHE[ck] = {"page_id": pid, "ts": time.time()}
        return {"id": pid, "title": title, "trace_id": trace_id}

    top_score = 1.0
    try:
        rr_top = co.rerank(model=RERANK_MODEL, query=question, documents=[d["text_chunk"] for d in doc_hits], top_n=1)
        if rr_top.results:
            top_score = rr_top.results[0].relevance_score or 0.0
    except Exception as e:
        logger.warning(f"top-score check failed: {e}")

    if RERANK_THRESHOLD and top_score < RERANK_THRESHOLD:
        plan = {
            "title": "We couldn’t find a confident answer",
            "summary": "",
            "show_toc": False,
            "sections": [{
                "id": "s1",
                "heading": "Try rephrasing your question",
                "paragraphs": ["We couldn’t confidently answer from the site’s content. Please rephrase or narrow the topic."],
                "bullets": []
            }],
        }
        article_html = render_article(plan, [], brand_class, False, images=[])
        safe_article = sanitize_html(article_html)
        safe_article = linkify_citations(safe_article, [])
        title = plan["title"]
        pid = _save_page(title, build_fullpage_html(title, safe_article, brand_class, primary, trace_id, page_id="tmp", base_url=base_url))
        full_html = build_fullpage_html(title, safe_article, brand_class, primary, trace_id, pid, base_url)
        PAGES[pid]["html"] = full_html
        ANSWER_CACHE[ck] = {"page_id": pid, "ts": time.time()}
        return {"id": pid, "title": title, "trace_id": trace_id}

    sources = [{"title": h.get("title") or "Untitled", "url": h.get("url") or "", "text_chunk": h.get("text_chunk")[:1800]} for h in doc_hits[:FINAL_DOCS]]
    citations = dedupe_citations([{"title": s["title"], "url": s.get("url","")} for s in sources])

    post_ids = [h.get("post_id") for h in doc_hits if h.get("post_id")]
    images = get_post_attachments(post_ids, k=3)
    if not images:
        all_imgs = fetch_wp_media()
        terms = _tok(question)
        for i in all_imgs:
            text = " ".join([i.get("title",""), i.get("alt",""), i.get("caption","")]).lower()
            i["score"] = sum(text.count(t) for t in terms)
        all_imgs.sort(key=lambda x: x.get("score",0), reverse=True)
        images = all_imgs[:3]

    prompt = build_planner_prompt(question, sources, layout, include_citations)
    try:
        plan = cohere_plan(prompt)
    except Exception as e:
        logger.error(f"planner error: {e}")
        top = sources[0]
        plan = {
            "title": top["title"],
            "summary": "",
            "show_toc": False,
            "sections": [{"id": "s1", "heading": top["title"], "paragraphs": [top["text_chunk"]], "bullets": []}]
        }

    article_html = render_article(plan, citations, brand_class, include_citations, images=images)
    safe_article = sanitize_html(article_html)
    safe_article = linkify_citations(safe_article, citations)
    title = plan.get("title") or "Article"

    page_id = _save_page(title, build_fullpage_html(title, safe_article, brand_class, primary, trace_id, page_id="tmp", base_url=base_url))
    full_html = build_fullpage_html(title, safe_article, brand_class, primary, trace_id, page_id, base_url)
    PAGES[page_id]["html"] = full_html
    ANSWER_CACHE[ck] = {"page_id": page_id, "ts": time.time()}
    return {"id": page_id, "title": title, "trace_id": trace_id}

# ============ Flask Routes ============
@app.get("/healthz")
def healthz():
    return jsonify({
        "ok": True,
        "wp_base": WP_BASE,
        "posts_cached": len(POSTS_CACHE["docs"]),
        "pages_cached": len(PAGES_CACHE["docs"]),
        "media_cached": len(MEDIA_CACHE["imgs"]),
        "pages_stored": len(PAGES),
        "answer_cache": len(ANSWER_CACHE),
    })

@app.post("/compose")
def compose():
    if BACKEND_API_KEY:
        provided = request.headers.get("X-Backend-Api-Key", "")
        if provided != BACKEND_API_KEY:
            return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    layout = data.get("layout", "guide")
    include_citations = bool(data.get("include_citations", True))
    brand_class = data.get("brand_class", "agent42labs")
    primary = data.get("primary", "#21808d")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    base_url = request.host_url.rstrip("/")
    result = compose_answer_page(question, layout, include_citations, brand_class, primary, base_url)
    page_url = f"{base_url}/v/{result['id']}"
    return jsonify({
        "id": result["id"],
        "title": result["title"],
        "trace_id": result["trace_id"],
        "page_url": page_url
    })

@app.get("/v/<page_id>")
def view_page(page_id):
    _cleanup_pages()
    page = PAGES.get(page_id)
    if not page:
        abort(404, description="Page not found")
    return Response(page["html"], mimetype="text/html")

@app.get("/v/<page_id>/download")
def download_page(page_id):
    _cleanup_pages()
    page = PAGES.get(page_id)
    if not page:
        abort(404, description="Page not found")
    filename = secure_filename(f"{page.get('title', 'article')}-{page_id}.html")
    buf = io.BytesIO(page["html"].encode("utf-8"))
    return send_file(buf, mimetype="text/html", as_attachment=True, download_name=filename)

@app.get("/favicon.ico")
def favicon():
    return "", 204

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
