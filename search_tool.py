
from ddgs import DDGS


def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the internet using DDGS.
    Return actual search titles, URLs and snippets.
    """

    # Step 1: Validate the search topic
    if not query or not query.strip():
        return "Error: Please enter a research topic."

    # Limit the number of results
    max_results = max(1, min(max_results, 10))

    # Step 2: Perform the web search
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

    # Step 3: Handle empty results
    if not results:
        return "No search results found for this topic."

    # Step 4: Format the search results
    formatted_results = []
    seen_urls = set()

    for result in results:
        title = result.get("title") or "Untitled"
        url = result.get("href") or ""
        snippet = result.get("body") or "No description available."

        # Accept valid web links only
        if not url.startswith(("http://", "https://")):
            continue

        # Avoid duplicate sources
        if url in seen_urls:
            continue

        seen_urls.add(url)

        source_number = len(formatted_results) + 1

        formatted_results.append(
            f"Source {source_number}\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Snippet: {snippet}"
        )

    if not formatted_results:
        return "No valid web sources were found."

    # Step 5: Return the collected information
    return "\n\n".join(formatted_results)
