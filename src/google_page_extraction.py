import requests

from bs4 import BeautifulSoup


def google_search(query: str, api_key: str, cse_id: str, n: int = 5, timeout: int = 8) -> list:
    """
    Perform a Google search using the Custom Search JSON API.

    Args:
        query (str): The search query.
        api_key (str): The API key for Google Custom Search.
        cse_id (str): The Custom Search Engine ID.
        n (int): The number of search results to return.

    Returns:
        list: A list of URLs from the search results.
    """
    r = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params={"key": api_key, "cx": cse_id, "q": query, "num": n},
        timeout=timeout
    )
    r.raise_for_status()
    data = r.json()
    items = data.get("items", [])
    return [item.get("link") for item in items if "link" in item]

def fetch_and_extract_text(url: str, timeout: int = 8) -> str:
    """
    Fetch the content of a webpage and extract the main text.

    Args:
        url (str): The URL of the webpage.
        timeout (int): The timeout for the HTTP request.

    Returns:
        str: The extracted text from the webpage.
    """
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
    except Exception:
        return ""

    soup = BeautifulSoup(r.text, "html.parser")

    for tag in soup(["script", "style", "header", "footer", "nav", "form", "aside"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = " ".join(text.split())
    return text

def extract_from_urls(urls: list) -> dict:
    """
    Fetch and extract text from a list of URLs.
    
    Args:
        urls (list): A list of URLs.

    Returns:
        dict: A dictionary mapping URLs to their extracted text.
    """
    out = {}
    for u in urls:
        txt = fetch_and_extract_text(u)
        if txt:
            out[u] = txt
    return out
