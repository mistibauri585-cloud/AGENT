from tavily import TavilyClient
import os

def search_the_web(query: str) -> str:
    """Triggers live Google search via Tavily if ChromaDB does not have the answer."""
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return "Web search utility unavailable due to missing API Key."
    try:
        tavily = TavilyClient(api_key=api_key)
        response = tavily.search(query=query, max_results=3)
        contexts = [result['content'] for result in response.get('results', [])]
        return "\n".join(contexts)
    except Exception as e:
        return f"Could not fetch online search contents: {str(e)}"
