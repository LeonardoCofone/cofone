from pathlib import Path


def load_documents(source):
    source = str(source)

    if "youtube.com" in source or "youtu.be" in source:
        return [_load_youtube(source)]

    if source.startswith("http://") or source.startswith("https://"):
        return [_load_url(source)]

    path = Path(source).resolve()

    if not path.exists():
        raise FileNotFoundError(f"[cofone] path not found: {path}")

    docs = []
    if path.is_file():
        docs.append(_read_file(path))
    elif path.is_dir():
        for f in path.rglob("*"):
            if f.suffix in (".txt", ".md", ".pdf"):
                docs.append(_read_file(f))

    return [d for d in docs if d]


def _read_file(path):
    try:
        if path.suffix == ".pdf":
            return _read_pdf(path)
        return path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[cofone] read error {path}: {e}")
        return None


def _read_pdf(path):
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except ImportError:
        raise ImportError("pip install pypdf")


def _load_url(url):
    try:
        import httpx
        from bs4 import BeautifulSoup

        if "wikipedia.org" in url:
            return _load_wikipedia(url)

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = httpx.get(url, timeout=15, follow_redirects=True, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        if len(text) < 100:
            raise ValueError(f"[cofone] page too short or blocked: {url}")
        return text
    except ImportError:
        raise ImportError("pip install httpx beautifulsoup4")


def _load_wikipedia(url):
    try:
        import wikipedia
        import re
        wikipedia.set_lang("it" if "it.wikipedia" in url else "en")
        slug = url.rstrip("/").split("/")[-1].replace("_", " ")
        slug = re.sub(r"%[0-9A-Fa-f]{2}", " ", slug)
        page = wikipedia.page(slug, auto_suggest=False)
        return page.content
    except ImportError:
        raise ImportError("pip install wikipedia")
    except Exception as e:
        raise ValueError(f"[cofone] Wikipedia error: {e}")


def _load_youtube(url):
    try:
        import re
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._transcripts import TranscriptList

        video_id = re.search(r"(?:v=|youtu\.be/)([^&\n?#]+)", url)
        if not video_id:
            raise ValueError(f"[cofone] video ID not found: {url}")

        vid = video_id.group(1)

        try:
            # new API (>=0.7.0)
            ytt = YouTubeTranscriptApi()
            transcript_list = ytt.list(vid)
            transcript = transcript_list.find_transcript(["en", "it"]).fetch()
            return " ".join(t.get("text", t) if isinstance(t, dict) else str(t) for t in transcript)
        except Exception:
            # fallback old API (<0.7.0)
            transcript = YouTubeTranscriptApi.list_transcripts(vid).find_transcript(["en", "it"]).fetch()
            return " ".join(t["text"] for t in transcript)

    except ImportError:
        raise ImportError("pip install youtube-transcript-api")