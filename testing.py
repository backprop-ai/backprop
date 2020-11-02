from kiri import Kiri, ElasticDocStore, Document
import json

doc_store = ElasticDocStore("http://localhost:9200")

kiri = Kiri(doc_store)

my_document = Document("Hello there 123", attributes={
                       "title": "Test title", "url": "https://kiri.ai"})
kiri.upload([my_document])


results = kiri.search("Hello there")

print(results.num_results)
for result in results.results:
    print(result.document.content, result.score)
