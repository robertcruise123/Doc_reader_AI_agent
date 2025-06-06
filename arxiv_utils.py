import arxiv
import datetime

def search_arxiv_papers(query: str, max_results: int = 3) -> str:
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )

        results = []
        for i, result in enumerate(search.results()):
            authors = ", ".join([author.name for author in result.authors])
            published_date = result.published.strftime("%Y-%m-%d")
            results.append(
                f"Paper {i+1}:\n"
                f"  Title: {result.title}\n"
                f"  Authors: {authors}\n"
                f"  Published: {published_date}\n"
                f"  Abstract: {result.summary[:200]}...\n"
                f"  URL: {result.entry_id}\n"
            )
        
        if results:
            return "Found the following papers on Arxiv:\n\n" + "\n---\n".join(results)
        else:
            return "No papers found on Arxiv for your query."

    except Exception as e:
        return f"An error occurred while searching Arxiv: {e}"