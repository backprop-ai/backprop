from kiri import Kiri, ElasticDocStore, Document

doc_store = ElasticDocStore("http://localhost:9200")

kiri = Kiri(doc_store)

my_document = Document("Where am I?", attributes={
                       "title": "Test title", "url": "https://kiri.ai"})
kiri.upload([my_document])


results = kiri.search("Hello there", max_results=15, min_score=0.1)

print("Total results:", results.total_results)
for result in results.results:
    print(result.document.content, "Score: " + str(result.score))
