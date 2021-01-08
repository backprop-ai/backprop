from typing import List
from ..search import Document


def check_duplicate_documents(documents: List[Document]):
    seen = set()
    duplicates = set()
    for d in documents:
        if d.id not in seen:
            seen.add(d.id)
        else:
            duplicates.add(d.id)

    duplicates = list(duplicates)

    if len(duplicates) != 0:
        raise ValueError(
            f"list of documents contains duplicate ids, example: {duplicates[0]}")


def check_document_types(documents: List[Document]):
    # set
    types = {type(d) for d in documents}
    if len(types) != 1:
        raise ValueError("all documents must have the same type")
