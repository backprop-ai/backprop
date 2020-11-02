from kiri import Kiri, ElasticDocStore, Document

doc_store = ElasticDocStore("http://localhost:9200")

kiri = Kiri(doc_store)

my_document = Document("Hello this is a sample document")
kiri.upload([my_document])


print(kiri.search("Hello document"))
