from typing import Dict


def elastic_to_search_results(res: Dict):
    from ..search import Document, SearchResult, SearchResults

    hits = res.get("hits")

    max_score = hits.get("max_score")
    total_results = hits.get("total").get("value")

    search_results = []
    for hit in hits.get("hits"):
        # TODO: Separate to a function for json -> Document
        source = hit["_source"]
        score = hit["_score"]
        document = Document(source.get("content"), hit.get("_id"),
                            attributes=source.get("attributes"),
                            vector=source.get("vector"))
        search_result = SearchResult(document, score)
        search_results.append(search_result)

    return SearchResults(max_score, total_results, search_results)
