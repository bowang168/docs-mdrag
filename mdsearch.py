#!/usr/bin/env python3
"""
mdsearch — local markdown knowledge base with hybrid vector + BM25 search

Subcommands:
    ingest   Index markdown files into Qdrant
    search   Search the index
    fetch    Download URLs from urls.md, convert to markdown
    stats    Show collection statistics
    filters  List available filter keys and values

Usage:
    python3 mdsearch.py ingest [--rebuild] [--config config.yaml]
    python3 mdsearch.py search "query" [--mode hybrid|semantic|keyword]
                                       [--limit N] [--filter k=v] [--json]
    python3 mdsearch.py fetch [--force] [--dry-run] [--limit N]
    python3 mdsearch.py stats
    python3 mdsearch.py filters
"""

# ── 1. IMPORTS & OPTIONAL DEPS ──────────────────────────────────────────────

import json, math, re, sys, os, time, hashlib, uuid, argparse
from pathlib import Path
from collections import Counter
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import urlparse

from qdrant_client import QdrantClient, models

# Optional: jieba for CJK tokenization
try:
    import jieba
    jieba.setLogLevel(20)
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False

# Optional: fetch deps
try:
    from bs4 import BeautifulSoup
    from markdownify import markdownify as md_convert
    HAS_FETCH_DEPS = True
except ImportError:
    HAS_FETCH_DEPS = False


# ── 2. CONFIG ────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = "config.yaml"


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    """Load config.yaml. Resolve relative paths to absolute based on config file location."""
    try:
        import yaml
    except ImportError:
        print("PyYAML not found. Run: pip3 install pyyaml", file=sys.stderr)
        sys.exit(1)
    cfg_path = Path(path).resolve()
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_base"] = cfg_path.parent
    return cfg


def resolve_path(cfg: dict, rel: str) -> Path:
    """Resolve a config-relative or absolute path."""
    p = Path(rel).expanduser()
    if p.is_absolute():
        return p
    return (cfg["_base"] / p).resolve()


# ── 3. BM25 ──────────────────────────────────────────────────────────────────

_CJK_RE = re.compile(r'[一-鿿㐀-䶿豈-﫿]')
_EN_SPLIT_RE = re.compile(r'[a-zA-Z0-9][-a-zA-Z0-9_.]*[a-zA-Z0-9]|[a-zA-Z0-9]+')

STOPWORDS = frozenset({
    # Chinese
    "的","了","在","是","我","有","和","就","不","人","都","一","一个","上","也",
    "很","到","说","要","去","你","会","着","没有","看","好","自己","这","他",
    "她","它","们","那","被","从","把","其","与","但","而","对","以","可以",
    # English
    "the","a","an","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","can","to","of","in","for","on","with","at","by","from","as",
    "into","through","during","before","after","above","below","between",
    "out","off","over","under","again","further","then","once","here","there",
    "when","where","why","how","all","each","every","both","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","and","but","or","if","while","because","this","that","these",
    "those","it","its","i","me","my","we","our","you","your","he","him","his",
    "she","her","they","them","their","what","which","who","whom",
})


def tokenize(text: str) -> list[str]:
    """Tokenize mixed CJK/English text. Uses jieba if available, else regex."""
    tokens = []
    text_lower = text.lower()
    if HAS_JIEBA:
        for word in jieba.cut(text_lower):
            word = word.strip()
            if not word or word in STOPWORDS:
                continue
            if _CJK_RE.search(word):
                tokens.append(word)
            else:
                for tok in _EN_SPLIT_RE.findall(word):
                    if tok not in STOPWORDS and len(tok) >= 2:
                        tokens.append(tok)
    else:
        for tok in _EN_SPLIT_RE.findall(text_lower):
            if tok not in STOPWORDS and len(tok) >= 2:
                tokens.append(tok)
    return tokens


class BM25Encoder:
    """BM25 sparse vector encoder. k1=1.5, b=0.75 (Okapi defaults)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.idf: dict[int, float] = {}
        self.avg_dl: float = 0.0
        self.n_docs: int = 0

    def fit(self, documents: list[str]) -> "BM25Encoder":
        self.n_docs = len(documents)
        if not self.n_docs:
            return self
        doc_freq: Counter = Counter()
        total_len = 0
        all_tokens: set[str] = set()
        for doc in documents:
            toks = tokenize(doc)
            total_len += len(toks)
            unique = set(toks)
            for t in unique:
                doc_freq[t] += 1
            all_tokens |= unique
        self.avg_dl = total_len / self.n_docs
        self.vocab = {t: i for i, t in enumerate(sorted(all_tokens))}
        for t, i in self.vocab.items():
            df = doc_freq[t]
            self.idf[i] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
        return self

    def encode(self, text: str) -> models.SparseVector:
        toks = tokenize(text)
        if not toks:
            return models.SparseVector(indices=[], values=[])
        tf = Counter(toks)
        dl = len(toks)
        avg_dl = self.avg_dl or 1.0
        indices, values = [], []
        for tok, count in tf.items():
            idx = self.vocab.get(tok)
            if idx is None:
                idx = len(self.vocab)
                self.vocab[tok] = idx
                self.idf[idx] = math.log((self.n_docs + 0.5) / 0.5 + 1.0) if self.n_docs else 1.0
            idf = self.idf.get(idx, 1.0)
            score = idf * (count * (self.k1 + 1)) / (count + self.k1 * (1 - self.b + self.b * dl / avg_dl))
            if score > 0:
                indices.append(idx)
                values.append(round(score, 6))
        return models.SparseVector(indices=indices, values=values)

    def save(self, path: str):
        state = {"k1": self.k1, "b": self.b, "vocab": self.vocab,
                 "idf": {str(k): v for k, v in self.idf.items()},
                 "avg_dl": self.avg_dl, "n_docs": self.n_docs}
        Path(path).write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "BM25Encoder":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        enc = cls(k1=state["k1"], b=state["b"])
        enc.vocab = state["vocab"]
        enc.idf = {int(k): v for k, v in state["idf"].items()}
        enc.avg_dl = state["avg_dl"]
        enc.n_docs = state["n_docs"]
        return enc

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


# ── 4. EMBED ─────────────────────────────────────────────────────────────────

def embed(text: str, cfg: dict) -> list[float]:
    """Call Ollama embed API via urllib (no requests dep).
    Retries once on failure. Raises RuntimeError if Ollama is unreachable.
    """
    url = cfg["embedding"]["url"]
    model = cfg["embedding"]["model"]
    payload = json.dumps({"model": model, "input": text}).encode()
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})
    for attempt in range(2):
        try:
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())["embeddings"][0]
        except URLError as e:
            if attempt == 0:
                time.sleep(1)
                continue
            raise RuntimeError(f"Ollama unreachable at {url}: {e}") from e


# ── 5. INGEST ────────────────────────────────────────────────────────────────

def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter from markdown. Returns (meta, body)."""
    import yaml
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    fm_text = text[3:end].strip()
    body = text[end + 4:].lstrip("\n")
    try:
        meta = yaml.safe_load(fm_text) or {}
    except Exception:
        meta = {}
    return meta, body


def extract_path_meta(path: str, patterns: list[dict]) -> dict:
    """Apply all matching path_meta patterns from config. Returns merged dict."""
    meta = {}
    for rule in patterns:
        m = re.search(rule["regex"], path)
        if m:
            key = rule["key"]
            fmt = rule.get("format")
            val = fmt.replace("{1}", m.group(1)) if fmt else m.group(1)
            meta[key] = val
    return meta


def heading_aware_chunk(text: str, max_chars: int = 1500) -> list[str]:
    """Split markdown by headings, keeping heading breadcrumb context.
    If a section exceeds max_chars, split line-by-line.
    """
    heading_re = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    headings = [(m.start(), len(m.group(1)), m.group(2)) for m in heading_re.finditer(text)]
    if not headings:
        return _split_by_size(text, max_chars)

    chunks = []
    breadcrumb: list[str] = []

    positions = [h[0] for h in headings] + [len(text)]
    for i, (pos, level, title) in enumerate(headings):
        section_text = text[pos:positions[i + 1]].strip()
        breadcrumb = breadcrumb[:level - 1] + [title]
        prefix = " > ".join(breadcrumb[:-1])
        header = f"[{prefix}] " if prefix else ""
        full = header + section_text
        if len(full) <= max_chars:
            chunks.append(full)
        else:
            for sub in _split_by_size(full, max_chars):
                chunks.append(sub)
    return chunks


def _split_by_size(text: str, max_chars: int) -> list[str]:
    """Fallback: split text into chunks of max_chars on line boundaries."""
    lines = text.splitlines(keepends=True)
    chunks, current = [], []
    size = 0
    for line in lines:
        if size + len(line) > max_chars and current:
            chunks.append("".join(current).strip())
            current, size = [], 0
        current.append(line)
        size += len(line)
    if current:
        chunks.append("".join(current).strip())
    return [c for c in chunks if c]


PAYLOAD_INDEXES = ["source_file", "title", "ol_version", "product", "topic", "tags", "source_url"]


def ensure_collection(client: QdrantClient, cfg: dict):
    """Create collection with dense + sparse vectors if it doesn't exist."""
    name = cfg["collection"]
    dim = cfg["embedding"]["dim"]
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={"bm25": models.SparseVectorParams()},
        )
        for field in PAYLOAD_INDEXES:
            client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)
        print(f"Created collection: {name}")


HASH_CACHE_FILE = ".hash_cache.json"
BM25_MODEL_FILE = ".bm25.json"
SKIP_PATTERNS = [".db", ".git", "__pycache__", ".DS_Store"]


def cmd_ingest(cfg: dict, rebuild: bool = False):
    """
    Scan configured dirs, chunk markdown, embed, upsert to Qdrant.

    Incremental by default: skip files whose SHA-256 hash hasn't changed.
    --rebuild: drops the collection and re-creates from scratch.
    """
    db_path = resolve_path(cfg, cfg["db_path"])
    db_path.mkdir(parents=True, exist_ok=True)
    hash_cache_path = db_path / HASH_CACHE_FILE
    bm25_path = db_path / BM25_MODEL_FILE

    client = QdrantClient(path=str(db_path))

    if rebuild:
        name = cfg["collection"]
        if name in [c.name for c in client.get_collections().collections]:
            client.delete_collection(name)
            print(f"Dropped collection: {name}")
        hash_cache = {}
        bm25 = None
    else:
        hash_cache = json.loads(hash_cache_path.read_text()) if hash_cache_path.exists() else {}
        bm25 = BM25Encoder.load(str(bm25_path)) if bm25_path.exists() else None

    ensure_collection(client, cfg)

    md_files = []
    for d in cfg.get("dirs", []):
        dir_path = resolve_path(cfg, d)
        if not dir_path.exists():
            print(f"[WARN] Dir not found: {dir_path}", file=sys.stderr)
            continue
        for f in dir_path.rglob("*.md"):
            if any(skip in str(f) for skip in SKIP_PATTERNS):
                continue
            md_files.append(f)

    print(f"Found {len(md_files)} markdown files")

    path_meta_rules = cfg.get("path_meta", [])
    changed_files = []
    all_chunks_for_bm25 = []

    for f in md_files:
        text = f.read_text(encoding="utf-8", errors="replace")
        file_hash = hashlib.sha256(text.encode()).hexdigest()
        rel = str(f.relative_to(cfg["_base"]))
        if hash_cache.get(rel) == file_hash and not rebuild:
            continue
        changed_files.append((f, rel, text, file_hash))
        _, body = parse_frontmatter(text)
        for chunk in heading_aware_chunk(body, cfg.get("chunk_size", 1500)):
            all_chunks_for_bm25.append(chunk)

    if not changed_files:
        print("Nothing changed. Index is up to date.")
        client.close()
        return

    print(f"Processing {len(changed_files)} changed files...")

    if rebuild or bm25 is None:
        print(f"Fitting BM25 on {len(all_chunks_for_bm25)} chunks...")
        bm25 = BM25Encoder()
        bm25.fit(all_chunks_for_bm25)
        bm25.save(str(bm25_path))

    collection = cfg["collection"]
    batch: list[models.PointStruct] = []
    BATCH_SIZE = 50

    def flush():
        if batch:
            client.upsert(collection_name=collection, points=batch)
            batch.clear()

    for f, rel, text, file_hash in changed_files:
        meta, body = parse_frontmatter(text)
        path_meta = extract_path_meta(rel, path_meta_rules)
        chunks = heading_aware_chunk(body, cfg.get("chunk_size", 1500))

        for i, chunk in enumerate(chunks):
            try:
                dense_vec = embed(chunk, cfg)
            except RuntimeError as e:
                print(f"[ERROR] {e}", file=sys.stderr)
                sys.exit(1)
            sparse_vec = bm25.encode(chunk)

            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{rel}#{i}"))
            payload = {
                "source_file": rel,
                "chunk_index": i,
                "chunk_text": chunk[:2000],
                "title": meta.get("title", f.stem),
                "source_url": meta.get("source_url", ""),
                "tags": meta.get("tags", []),
                "topic": meta.get("topic", ""),
                **path_meta,
                **{k: v for k, v in meta.items()
                   if k not in ("title", "source_url", "tags", "topic")},
            }

            vectors = {"dense": dense_vec}
            if sparse_vec.indices:
                vectors["bm25"] = sparse_vec

            batch.append(models.PointStruct(id=point_id, vector=vectors, payload=payload))
            if len(batch) >= BATCH_SIZE:
                flush()

        hash_cache[rel] = file_hash
        print(f"  ingested: {rel} ({len(chunks)} chunks)")

    flush()

    hash_cache_path.write_text(json.dumps(hash_cache, ensure_ascii=False, indent=2))
    print("Done. Hash cache updated.")
    client.close()


# ── 6. SEARCH ────────────────────────────────────────────────────────────────

def build_filter(filters: dict) -> models.Filter | None:
    if not filters:
        return None
    must = [models.FieldCondition(key=k, match=models.MatchValue(value=v))
            for k, v in filters.items()]
    return models.Filter(must=must)


def search(query: str, cfg: dict, mode: str = "hybrid",
           limit: int = 5, filters: dict = None) -> list[dict]:
    """
    Search modes:
      hybrid  — dense prefetch + BM25 prefetch → RRF fusion (default)
      semantic — dense only
      keyword  — BM25 only

    Falls back to semantic if BM25 model missing or sparse vector is empty.
    """
    db_path = resolve_path(cfg, cfg["db_path"])
    bm25_path = db_path / BM25_MODEL_FILE
    client = QdrantClient(path=str(db_path))
    collection = cfg["collection"]
    qdrant_filter = build_filter(filters)

    bm25 = None
    if mode in ("hybrid", "keyword"):
        if bm25_path.exists():
            bm25 = BM25Encoder.load(str(bm25_path))
        else:
            print("[WARN] BM25 model not found; falling back to semantic", file=sys.stderr)
            mode = "semantic"

    results_raw = None

    if mode == "keyword":
        sparse_vec = bm25.encode(query)
        if not sparse_vec.indices:
            print("[WARN] Query produced empty BM25 vector; falling back to semantic", file=sys.stderr)
            mode = "semantic"
        else:
            results_raw = client.query_points(
                collection_name=collection, query=sparse_vec,
                using="bm25", query_filter=qdrant_filter,
                limit=limit, with_payload=True,
            )

    if mode == "semantic":
        vec = embed(query, cfg)
        results_raw = client.query_points(
            collection_name=collection, query=vec,
            using="dense", query_filter=qdrant_filter,
            limit=limit, with_payload=True,
        )

    if mode == "hybrid":
        vec = embed(query, cfg)
        sparse_vec = bm25.encode(query)
        prefetch = [models.Prefetch(query=vec, using="dense",
                                    limit=limit * 3, filter=qdrant_filter)]
        if sparse_vec.indices:
            prefetch.append(models.Prefetch(query=sparse_vec, using="bm25",
                                             limit=limit * 3, filter=qdrant_filter))
            fusion = models.FusionQuery(fusion=models.Fusion.RRF)
            results_raw = client.query_points(
                collection_name=collection, prefetch=prefetch,
                query=fusion, limit=limit, with_payload=True,
            )
        else:
            results_raw = client.query_points(
                collection_name=collection, query=vec,
                using="dense", query_filter=qdrant_filter,
                limit=limit, with_payload=True,
            )

    client.close()
    if results_raw is None:
        return []
    return [{"score": p.score, "payload": p.payload} for p in results_raw.points]


# ── 7. FETCH ─────────────────────────────────────────────────────────────────

def cmd_fetch(cfg: dict, force: bool = False, dry_run: bool = False,
              limit: int | None = None, filter_str: str | None = None):
    """
    Read urls.md, fetch each URL, convert HTML → Markdown, save to output_dir.

    Saved filename: sanitize URL path (/ → _, strip leading slash + .html).
    Frontmatter added: title, source_url, fetched timestamp.
    Skip already-saved files unless --force.
    """
    if not HAS_FETCH_DEPS:
        print("Missing deps: pip3 install markdownify beautifulsoup4", file=sys.stderr)
        sys.exit(1)

    fetch_cfg = cfg.get("fetch", {})
    urls_file = resolve_path(cfg, fetch_cfg.get("urls_file", "urls.md"))
    output_dir = resolve_path(cfg, fetch_cfg.get("output_dir", "docs/fetched/"))
    delay = fetch_cfg.get("delay", 1.5)
    timeout = fetch_cfg.get("timeout", 30)
    user_agent = fetch_cfg.get("user_agent", "mdsearch/1.0")
    strip_selectors = fetch_cfg.get("strip_selectors", ["nav", "header", "footer"])

    if not urls_file.exists():
        print(f"urls.md not found at {urls_file}. Create it with [title](url) links.")
        return

    link_re = re.compile(r'\[([^\]]+)\]\((https?://[^\)]+)\)')
    text = urls_file.read_text(encoding="utf-8")
    entries = []
    seen = set()
    for m in link_re.finditer(text):
        url = m.group(2).strip().rstrip("/")
        if url in seen:
            continue
        seen.add(url)
        if filter_str and filter_str not in url:
            continue
        entries.append({"title": m.group(1).strip(), "url": url})

    if limit:
        entries = entries[:limit]

    print(f"Found {len(entries)} URLs to fetch")
    if dry_run:
        for e in entries:
            print(f"  {e['url']}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone

    for i, entry in enumerate(entries):
        url = entry["url"]
        parsed = urlparse(url)
        clean = parsed.path.strip("/").replace("/", "_").replace(".html", "") or "index"
        out_file = output_dir / f"{clean}.md"

        if out_file.exists() and not force:
            print(f"  skip (exists): {out_file.name}")
            continue

        print(f"  [{i+1}/{len(entries)}] {url}")
        try:
            req = Request(url, headers={"User-Agent": user_agent,
                                         "Accept": "text/html"})
            with urlopen(req, timeout=timeout) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"    [ERROR] {e}", file=sys.stderr)
            continue

        soup = BeautifulSoup(html, "html.parser")
        page_title = soup.title.string.strip() if soup.title else entry["title"]
        for sel in strip_selectors:
            for tag in soup.select(sel):
                tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.find("body") or soup
        markdown = md_convert(str(main), heading_style="ATX", bullets="-")

        fetched_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        frontmatter = (f"---\ntitle: {json.dumps(page_title)}\n"
                       f"source_url: {url}\nfetched: {fetched_ts}\n---\n\n")
        out_file.write_text(frontmatter + markdown, encoding="utf-8")
        print(f"    saved: {out_file.relative_to(cfg['_base'])} ({len(markdown)} chars)")

        if i < len(entries) - 1:
            time.sleep(delay)

    print("Fetch complete.")


# ── 8. CLI ───────────────────────────────────────────────────────────────────

def cmd_stats(cfg: dict):
    """Print collection stats and BM25 vocab size."""
    db_path = resolve_path(cfg, cfg["db_path"])
    client = QdrantClient(path=str(db_path))
    name = cfg["collection"]
    try:
        info = client.get_collection(name)
        print(f"Collection : {name}")
        print(f"Points     : {info.points_count}")
        print(f"Status     : {info.status}")
        print(f"DB path    : {db_path}")
        bm25_path = db_path / BM25_MODEL_FILE
        if bm25_path.exists():
            bm25 = BM25Encoder.load(str(bm25_path))
            print(f"BM25 vocab : {bm25.vocab_size} tokens")
    except Exception as e:
        print(f"Error: {e}")
    client.close()


def cmd_filters(cfg: dict):
    """List distinct values for all payload fields."""
    db_path = resolve_path(cfg, cfg["db_path"])
    client = QdrantClient(path=str(db_path))
    name = cfg["collection"]
    fields = PAYLOAD_INDEXES + list({r["key"] for r in cfg.get("path_meta", [])})
    payloads = []
    offset = None
    while True:
        pts, next_off = client.scroll(collection_name=name, limit=200,
                                       offset=offset, with_payload=fields,
                                       with_vectors=False)
        payloads.extend(p.payload for p in pts)
        if next_off is None:
            break
        offset = next_off
    for field in sorted(set(fields)):
        vals = set()
        for p in payloads:
            v = p.get(field)
            if v is None:
                continue
            if isinstance(v, list):
                vals.update(str(x) for x in v)
            else:
                vals.add(str(v))
        if vals:
            print(f"{field}: {', '.join(sorted(vals))}")
    client.close()


def main():
    parser = argparse.ArgumentParser(prog="mdsearch", description="Local markdown search")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config.yaml")
    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest", help="Index markdown files")
    p_ingest.add_argument("--rebuild", action="store_true", help="Drop and rebuild from scratch")

    p_search = sub.add_parser("search", help="Search the index")
    p_search.add_argument("query")
    p_search.add_argument("--mode", "-m", choices=["hybrid", "semantic", "keyword"],
                          default="hybrid")
    p_search.add_argument("--limit", "-n", type=int, default=5)
    p_search.add_argument("--filter", "-f", action="append", metavar="KEY=VALUE")
    p_search.add_argument("--json", action="store_true", dest="as_json")

    p_fetch = sub.add_parser("fetch", help="Download URLs from urls.md")
    p_fetch.add_argument("--force", action="store_true")
    p_fetch.add_argument("--dry-run", action="store_true")
    p_fetch.add_argument("--limit", type=int)
    p_fetch.add_argument("--filter", metavar="SUBSTRING")

    sub.add_parser("stats", help="Collection statistics")
    sub.add_parser("filters", help="List filter values")

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return

    cfg_path = args.config
    if not Path(cfg_path).exists():
        fallback = Path(__file__).parent / cfg_path
        if fallback.exists():
            cfg_path = str(fallback)
    try:
        cfg = load_config(cfg_path)
    except FileNotFoundError:
        print(f"Config not found: {cfg_path}. Run from the repo root or pass --config.")
        sys.exit(1)

    if args.cmd == "ingest":
        cmd_ingest(cfg, rebuild=args.rebuild)

    elif args.cmd == "search":
        filters = {}
        if args.filter:
            for f in args.filter:
                k, v = f.split("=", 1)
                filters[k] = v
        results = search(args.query, cfg, mode=args.mode,
                         limit=args.limit, filters=filters or None)
        if args.as_json:
            out = [{"score": round(r["score"], 4),
                    "source_file": r["payload"].get("source_file", ""),
                    "title": r["payload"].get("title", ""),
                    "tags": r["payload"].get("tags", []),
                    "source_url": r["payload"].get("source_url", ""),
                    "text": (r["payload"].get("chunk_text") or "")[:300],
                    **{k: v for k, v in r["payload"].items()
                       if k not in ("chunk_text", "source_file", "title", "tags", "source_url",
                                    "chunk_index", "file_hash")}}
                   for r in results]
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            print(f"Query: {args.query} (mode: {args.mode})\nResults: {len(results)}\n")
            for i, r in enumerate(results):
                p = r["payload"]
                print(f"[{i+1}] score={r['score']:.4f}  {p.get('source_file','')}")
                print(f"     title: {p.get('title','')}")
                text = (p.get("chunk_text") or "").replace("\n", " ")[:200]
                print(f"     {text}...\n")

    elif args.cmd == "fetch":
        cmd_fetch_filter = getattr(args, "filter", None)
        cmd_fetch(cfg, force=args.force, dry_run=args.dry_run,
                  limit=args.limit, filter_str=cmd_fetch_filter)

    elif args.cmd == "stats":
        cmd_stats(cfg)

    elif args.cmd == "filters":
        cmd_filters(cfg)


if __name__ == "__main__":
    main()
