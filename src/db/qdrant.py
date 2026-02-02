"""Qdrant vector store client for semantic search."""

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)

from config.settings import settings


class QdrantVectorStore:
    """Qdrant client wrapper for vector operations."""

    def __init__(self, url: str | None = None):
        """Initialize Qdrant client.

        Args:
            url: Qdrant server URL. Defaults to settings.qdrant_url.
        """
        self.url = url or settings.qdrant_url
        self.client = QdrantClient(url=self.url)

    def create_collection(
        self,
        collection_name: str,
        vector_size: int = 384,
        distance: Distance = Distance.COSINE,
    ) -> None:
        """Create a new collection if it doesn't exist.

        Args:
            collection_name: Name of the collection.
            vector_size: Dimension of vectors.
            distance: Distance metric to use.
        """
        existing = self.client.get_collections().collections
        existing_names = [c.name for c in existing]

        if collection_name not in existing_names:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance,
                ),
            )

    def upsert_vectors(
        self,
        collection_name: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        """Upsert vectors into collection.

        Args:
            collection_name: Target collection name.
            ids: Vector IDs.
            vectors: Vector embeddings.
            payloads: Optional metadata for each vector.
        """
        points = []
        for i, (id_, vector) in enumerate(zip(ids, vectors)):
            payload = payloads[i] if payloads else {}
            points.append(
                PointStruct(
                    id=id_,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=collection_name,
            points=points,
        )

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 10,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors.

        Args:
            collection_name: Collection to search.
            query_vector: Query embedding.
            top_k: Number of results to return.
            filter_conditions: Optional filter conditions.

        Returns:
            List of search results with scores and payloads.
        """
        query_filter = None
        if filter_conditions:
            conditions = []
            for key, value in filter_conditions.items():
                if isinstance(value, dict) and "range" in value:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            range=Range(**value["range"]),
                        )
                    )
                else:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value),
                        )
                    )
            query_filter = Filter(must=conditions)

        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in results.points
        ]

    def search_papers_by_abstract(
        self,
        query_vector: list[float],
        top_k: int = 10,
        year_range: tuple[int, int] | None = None,
        categories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search papers by abstract similarity.

        Args:
            query_vector: Query embedding.
            top_k: Number of results.
            year_range: Optional (min_year, max_year) filter.
            categories: Optional category filter.

        Returns:
            List of similar papers with scores.
        """
        filter_conditions = {}
        if year_range:
            filter_conditions["year"] = {
                "range": {"gte": year_range[0], "lte": year_range[1]}
            }

        return self.search(
            collection_name="paper_abstracts",
            query_vector=query_vector,
            top_k=top_k,
            filter_conditions=filter_conditions if filter_conditions else None,
        )

    def search_methods_by_description(
        self,
        query_vector: list[float],
        top_k: int = 10,
        method_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search methods by description similarity.

        Args:
            query_vector: Query embedding.
            top_k: Number of results.
            method_type: Optional method type filter.

        Returns:
            List of similar methods with scores.
        """
        filter_conditions = {}
        if method_type:
            filter_conditions["method_type"] = method_type

        return self.search(
            collection_name="method_descriptions",
            query_vector=query_vector,
            top_k=top_k,
            filter_conditions=filter_conditions if filter_conditions else None,
        )

    def get_collection_info(self, collection_name: str) -> dict[str, Any]:
        """Get collection information.

        Args:
            collection_name: Name of collection.

        Returns:
            Collection information dictionary.
        """
        info = self.client.get_collection(collection_name)
        return {
            "name": collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.value,
        }

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection.

        Args:
            collection_name: Name of collection to delete.
        """
        self.client.delete_collection(collection_name)
