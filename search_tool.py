
from ddgs import DDGS
from trafilatura import fetch_url, extract


# Step 1: Extract webpage evidence
def get_webpage_evidence(url, snippet):

    try:
        downloaded = fetch_url(url)

        if downloaded:
            article_text = extract(
                downloaded,
                include_comments=False
            )

            if article_text and len(article_text.strip()) >= 500:

                # Limit text to control LLM token usage
                article_excerpt = article_text.strip()[:3500]

                return {
                    "status": "ARTICLE_EXCERPT",
                    "text": article_excerpt
                }

    except Exception:
        pass

    # Fallback if webpage cannot be extracted
    return {
        "status": "SNIPPET_ONLY",
        "text": snippet or "No description available."
    }


# Step 2: Search the internet
def search_web(query: str, max_results: int = 5) -> str:

    if not query or not query.strip():
        return "Error: Please enter a research topic."

    max_results = max(1, min(max_results, 10))

    try:
        results = DDGS(timeout=15).text(
            query=query.strip(),
            region="us-en",
            safesearch="moderate",
            max_results=max_results
        )

    except Exception as error:
        return (
            f"Web search failed ({type(error).__name__}). "
            "Please try again later."
        )

    if not results:
        return "No search results found."

    formatted_results = []
    seen_urls = set()

    # Step 3: Process each source
    for result in results:

        title = result.get("title") or "Untitled"
        url = result.get("href") or ""
        snippet = result.get("body") or ""

        if not url.startswith(("http://", "https://")):
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)

        # Step 4: Extract article or use snippet
        evidence = get_webpage_evidence(
            url,
            snippet
        )

        source_number = len(formatted_results) + 1

        formatted_results.append(
            f"Source {source_number}\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Evidence Type: {evidence['status']}\n"
            f"Evidence:\n{evidence['text']}"
        )

        if len(formatted_results) >= max_results:
            break

    if not formatted_results:
        return "No valid web sources were found."

    # Step 5: Return all gathered evidence
    return "\n\n---\n\n".join(formatted_results)
