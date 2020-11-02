from typing import Dict


def elastic_to_search_results(res: Dict, score_modifier: float):
    from ..search import Document, SearchResult, SearchResults

    hits = res.get("hits")

    max_score = hits.get("max_score")
    total_results = hits.get("total").get("value")

    search_results = []
    for hit in hits.get("hits"):
        source = hit["_source"]

        # score_modifier was added when searching on elasticsearch
        score = hit["_score"] - score_modifier
        document = Document(source.get("content"), hit.get("_id"),
                            attributes=source.get("attributes"),
                            vector=source.get("vector"))
        search_result = SearchResult(document, score)
        search_results.append(search_result)

    return SearchResults(max_score, total_results, search_results)
