---
name: mdsearch
description: >
  Search a local markdown knowledge base using hybrid vector + BM25 search.
  Wraps mdsearch.py in ~/g/local-mdsearch (or any configured path).
triggers:
  - /mdsearch
  - phrases: "search my docs", "search local docs", "check local kb", "local knowledge base"
---

# mdsearch

Local hybrid search over your own markdown files.
Engine: Qdrant local mode + Ollama embeddings + BM25 sparse vectors.

## SETUP (first time)

```bash
# 1. Install deps
pip3 install qdrant-client markdownify beautifulsoup4

# 2. Start Ollama + pull model
ollama serve &
ollama pull qwen3-embedding:0.6b

# 3. Add your docs (or fetch from web)
python3 mdsearch.py fetch          # download URLs from urls.md
python3 mdsearch.py ingest         # index docs/ into Qdrant
```

## SEARCH STRATEGY

| Query type | Recommended mode | Example |
|------------|-----------------|---------|
| Most queries | hybrid (default) | `"kernel panic after upgrade"` |
| Exact string: error codes, CVE, package names, commands | keyword | `"CVE-2024-26704"` |
| Conceptual / fuzzy | semantic | `"how does kdump work"` |

## TOOL INVOCATION (for AI agents)

Always use `--json` for machine-readable output.

```bash
# Default search
python3 ~/g/local-mdsearch/mdsearch.py search "QUERY" --json

# With filters
python3 ~/g/local-mdsearch/mdsearch.py search "QUERY" --filter ol_version=ol9 --json

# Keyword mode (CVE, package names, error codes)
python3 ~/g/local-mdsearch/mdsearch.py search "QUERY" --mode keyword --json

# More results
python3 ~/g/local-mdsearch/mdsearch.py search "QUERY" --limit 10 --json
```

## JSON OUTPUT SCHEMA

```json
[
  {
    "score": 0.8312,
    "source_file": "docs/fetched/oracle-linux_9_network.md",
    "title": "Oracle Linux 9: Configuring the Network",
    "tags": ["networking", "ol9"],
    "source_url": "https://docs.oracle.com/en/operating-systems/oracle-linux/9/network/",
    "text": "First 300 chars of the matching chunk...",
    "ol_version": "ol9",
    "product": "oracle-linux"
  }
]
```

## FETCHING DOCS

```bash
# Edit urls.md to add [title](url) links, then:
python3 mdsearch.py fetch              # fetch new URLs only
python3 mdsearch.py fetch --force      # re-fetch all
python3 mdsearch.py fetch --dry-run    # preview what would be fetched
python3 mdsearch.py fetch --limit 5    # test with 5 URLs
```

## INGESTING DOCS

```bash
python3 mdsearch.py ingest             # incremental (unchanged files skipped)
python3 mdsearch.py ingest --rebuild   # full rebuild (drop + re-index)
```

## DIAGNOSTICS

```bash
python3 mdsearch.py stats              # points count, BM25 vocab, DB path
python3 mdsearch.py filters            # available filter keys and values
```

## EXTENDING

**Add a new docs directory:**
Edit `config.yaml` → `dirs:` → add path → run `mdsearch.py ingest`

**Add metadata from path:**
Edit `config.yaml` → `path_meta:` → add `{regex, key, format}` entry

**Change embedding model:**
Edit `config.yaml` → `embedding.model` and `embedding.dim` → run `mdsearch.py ingest --rebuild`

**Add URLs to fetch:**
Edit `urls.md` → add `[title](url)` lines → run `mdsearch.py fetch`
