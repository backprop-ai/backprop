from typing import Dict


def elastic_to_search_results(res: Dict, score_modifier: float, doc_class):
    """Converts Elasticsearch result to Kiri SearchResults object

    Args:
        res: dictionary/JSON returned from Elasticsearch
        score_modifier: value to be subtracted from each result's score
        doc_class: Type of Document object to be included in SearchResult object

    Returns:
        SearchResults equivalent of Elastic result
    """

    from ..search import SearchResult, SearchResults

    hits = res.get("hits")

    max_score = hits.get("max_score")
    total_results = hits.get("total").get("value")

    search_results = []
    for hit in hits.get("hits"):
        source = hit["_source"]

        # score_modifier was added when searching on elasticsearch
        score = hit["_score"] - score_modifier

        document = doc_class.from_elastic(id=hit.get("_id"),
                                          **source)
        search_result = SearchResult(document, score)
        search_results.append(search_result)

    return SearchResults(max_score, total_results, search_results)
