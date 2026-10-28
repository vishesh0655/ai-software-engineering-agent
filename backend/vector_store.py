import re
import chromadb
from backend.settings import settings


class VectorStore:
    @staticmethod
    def delete_project_collection(project_id):
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,54}", project_id):
            raise ValueError("project_id is not a valid collection identifier")
        client = chromadb.PersistentClient(path=settings.chroma_path)
        collection_name = f"codebase_{project_id}"
        collections = client.list_collections()
        names = {item.name if hasattr(item, "name") else str(item) for item in collections}
        if collection_name not in names:
            return False
        client.delete_collection(name=collection_name)
        return True

    def __init__(self, path=None, project_id=None):
        path = path or settings.chroma_path
        self.client = chromadb.PersistentClient(path=path)

        if project_id is None:
            collection_name = "codebase"
        else:
            if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,54}", project_id):
                raise ValueError("project_id is not a valid collection identifier")
            collection_name = f"codebase_{project_id}"

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_documents(self, documents, embeddings, metadatas, ids):
        self.collection.upsert(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query_embedding, n_results=5):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        return results

    def count(self):
        return self.collection.count()

    def remove_documents_except(self, retained_ids):
        """Remove stale vectors from this collection after a successful upsert."""
        current_ids = self.collection.get()["ids"]
        retained = set(retained_ids)
        stale_ids = [document_id for document_id in current_ids if document_id not in retained]
        if stale_ids:
            self.collection.delete(ids=stale_ids)

    def delete_collection(self):
        name = self.collection.name
        self.client.delete_collection(name=name)
