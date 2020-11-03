from kiri import Kiri, ElasticDocStore, Document, ElasticDocument, ElasticChunkedDocument

doc_store = ElasticDocStore("http://localhost:9200",
                            doc_class=ElasticChunkedDocument, index="kiri_default")

kiri = Kiri(doc_store)

my_document = ElasticChunkedDocument("Hello. This is a document with some important content.", attributes={
    "title": "Test title", "url": "https://kiri.ai"}, chunking_level=1)
kiri.upload([my_document])

results = kiri.search("Hello there", max_results=50, min_score=0.01)

print("Total results:", results.total_results)
for result in results.results:
    print(result.preview, "Score: " + str(result.score))
    print("========")

# print(results.to_json())
