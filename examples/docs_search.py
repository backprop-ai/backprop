from kiri import Kiri, InMemoryDocStore, ElasticDocStore, ChunkedDocument, ElasticChunkedDocument
from sys import argv

"""
This example shows a simple (but powerful) semantic search on a collection of documents.

"""

elastic = False

if elastic:
    doc_store = ElasticDocStore("http://localhost:9000", doc_class=ElasticChunkedDocument, index="kiri_default")
else:
    doc_store = InMemoryDocStore(doc_class=ChunkedDocument)

kiri = Kiri(doc_store)

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


# TODO find some docs to use here...
documents = []
kiri.upload(documents)

results = kiri.search(query, max_results=50, min_score=0.01)

print("Total results:", results.total_results)
for result in results.results:
    print(result.preview, "Score: " + str(result.score))
    print("========")