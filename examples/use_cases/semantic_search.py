from kiri import Kiri, InMemoryDocStore, ElasticDocStore, ChunkedDocument, ElasticChunkedDocument
from sys import argv
from docs.tech_docs import tech_docs

"""
This example shows a simple (but powerful) semantic search on a collection of documents.

Included in this are some tech docs from an old search-based MVP we built.
It covers a range of topics. Some example searches:
- Python format style
- Deployment strategy
- Node dependencies
"""

elastic = False

if elastic:
    doc_store = ElasticDocStore("http://localhost:9000", doc_class=ElasticChunkedDocument, index="kiri_default")
    docs = [d["elastic"] for d in tech_docs]
else:
    doc_store = InMemoryDocStore(doc_class=ChunkedDocument)
    docs = [d["memory"] for d in tech_docs]

kiri = Kiri(doc_store, local=True)
kiri.upload(docs)

query = ""
if len(argv) == 1:
    print("Supply a query when running this script")
    print("Usage: python doc_search.py \"<your query here>\"")
    exit(0)
elif len(argv) == 2:
    query = argv[1]
else:
    query = " ".join(argv[1:])

print(f"Query: {query}")


results = kiri.search(query, max_results=3, min_score=0.01)

print("Total results:", results.total_results)
for result in results.results:
    # Can add any attributes you like to a doc
    print(result.document.attributes["title"])
    print(result.preview)
    print("Score: " + str(result.score))
    print("========")