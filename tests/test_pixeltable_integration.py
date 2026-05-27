"""Tests that exercise real Pixeltable operations without API keys.

Uses deterministic fake embeddings so these run in CI with zero config.
"""

import hashlib
from typing import Any

import numpy as np
import pixeltable as pxt
import pytest
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from llama_index.vector_stores.pixeltable import PixeltableVectorStore

EMBED_DIM = 32
TABLE_DIR = "test_li_ci"


def _fake_embedding(text: str) -> list[float]:
    h = hashlib.sha256(text.encode()).digest()
    vec = np.frombuffer(h, dtype=np.uint8)[:EMBED_DIM].astype(np.float32)
    return (vec / (np.linalg.norm(vec) + 1e-9)).tolist()


def _make_node(node_id: str, text: str, metadata: dict | None = None, ref_doc_id: str = "") -> TextNode:
    return TextNode(
        id_=node_id,
        text=text,
        metadata=metadata or {},
        embedding=_fake_embedding(text),
        relationships={} if not ref_doc_id else {},
    )


@pytest.fixture()
def table_name() -> str:
    return f"{TABLE_DIR}.docs"


@pytest.fixture(autouse=True)
def clean_table() -> Any:
    try:
        pxt.drop_dir(TABLE_DIR, force=True)
    except Exception:
        pass
    yield
    try:
        pxt.drop_dir(TABLE_DIR, force=True)
    except Exception:
        pass


# ---- CRUD ----


class TestCRUD:
    def test_add_and_count(self, table_name: str) -> None:
        vs = PixeltableVectorStore(table_name=table_name, embed_dim=EMBED_DIM)
        nodes = [_make_node(f"n{i}", f"text-{i}") for i in range(3)]
        ids = vs.add(nodes)
        assert len(ids) == 3
        assert vs.table.count() == 3

    def test_delete_by_ref_doc_id(self, table_name: str) -> None:
        vs = PixeltableVectorStore(table_name=table_name, embed_dim=EMBED_DIM)
        node = TextNode(
            id_="n1",
            text="hello",
            embedding=_fake_embedding("hello"),
            relationships={"1": {"node_id": "doc_1", "node_type": "4"}},
        )
        vs.add([node])
        assert vs.table.count() == 1
        vs.delete("doc_1")

    def test_delete_nodes_by_id(self, table_name: str) -> None:
        vs = PixeltableVectorStore(table_name=table_name, embed_dim=EMBED_DIM)
        vs.add([_make_node("a", "alpha"), _make_node("b", "beta"), _make_node("c", "gamma")])
        vs.delete_nodes(node_ids=["a", "c"])
        assert vs.table.count() == 1

    def test_clear(self, table_name: str) -> None:
        vs = PixeltableVectorStore(table_name=table_name, embed_dim=EMBED_DIM)
        vs.add([_make_node("n1", "text1")])
        assert vs.table.count() == 1
        vs.clear()
        assert vs.table.count() == 0


# ---- Query ----


class TestQuery:
    def test_basic_query(self, table_name: str) -> None:
        vs = PixeltableVectorStore(table_name=table_name, embed_dim=EMBED_DIM)
        vs.add(
            [
                _make_node("n1", "Pixeltable processes multimodal data"),
                _make_node("n2", "Weather forecast for tomorrow"),
                _make_node("n3", "AI model training pipeline"),
            ]
        )
        q = VectorStoreQuery(query_embedding=_fake_embedding("data processing"), similarity_top_k=2)
        result = vs.query(q)
        assert len(result.nodes) <= 2
        assert len(result.similarities) == len(result.nodes)
        assert all(isinstance(s, float) for s in result.similarities)

    def test_query_returns_correct_top_k(self, table_name: str) -> None:
        vs = PixeltableVectorStore(table_name=table_name, embed_dim=EMBED_DIM)
        vs.add([_make_node(f"n{i}", f"document {i}") for i in range(5)])
        q = VectorStoreQuery(query_embedding=_fake_embedding("document"), similarity_top_k=3)
        result = vs.query(q)
        assert len(result.nodes) == 3


# ---- MetadataFilter ----


class TestMetadataFilter:
    def _make_store(self, table_name: str) -> PixeltableVectorStore:
        vs = PixeltableVectorStore(table_name=table_name, embed_dim=EMBED_DIM)
        vs.add(
            [
                _make_node("n1", "alpha", {"category": "science", "priority": 3}),
                _make_node("n2", "beta", {"category": "art", "priority": 1}),
                _make_node("n3", "gamma", {"category": "science", "priority": 5}),
                _make_node("n4", "delta", {"category": "tech", "priority": 4}),
            ]
        )
        return vs

    def test_filter_eq(self, table_name: str) -> None:
        vs = self._make_store(table_name)
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="category", value="science", operator=FilterOperator.EQ),
            ]
        )
        q = VectorStoreQuery(query_embedding=_fake_embedding("test"), similarity_top_k=10, filters=filters)
        result = vs.query(q)
        assert len(result.nodes) == 2

    def test_filter_ne(self, table_name: str) -> None:
        vs = self._make_store(table_name)
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="category", value="art", operator=FilterOperator.NE),
            ]
        )
        q = VectorStoreQuery(query_embedding=_fake_embedding("test"), similarity_top_k=10, filters=filters)
        result = vs.query(q)
        assert len(result.nodes) == 3

    def test_filter_gt(self, table_name: str) -> None:
        vs = self._make_store(table_name)
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="priority", value=3, operator=FilterOperator.GT),
            ]
        )
        q = VectorStoreQuery(query_embedding=_fake_embedding("test"), similarity_top_k=10, filters=filters)
        result = vs.query(q)
        assert len(result.nodes) == 2

    def test_filter_lt(self, table_name: str) -> None:
        vs = self._make_store(table_name)
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="priority", value=4, operator=FilterOperator.LT),
            ]
        )
        q = VectorStoreQuery(query_embedding=_fake_embedding("test"), similarity_top_k=10, filters=filters)
        result = vs.query(q)
        assert len(result.nodes) == 2

    def test_filter_gte(self, table_name: str) -> None:
        vs = self._make_store(table_name)
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="priority", value=4, operator=FilterOperator.GTE),
            ]
        )
        q = VectorStoreQuery(query_embedding=_fake_embedding("test"), similarity_top_k=10, filters=filters)
        result = vs.query(q)
        assert len(result.nodes) == 2

    def test_filter_lte(self, table_name: str) -> None:
        vs = self._make_store(table_name)
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="priority", value=1, operator=FilterOperator.LTE),
            ]
        )
        q = VectorStoreQuery(query_embedding=_fake_embedding("test"), similarity_top_k=10, filters=filters)
        result = vs.query(q)
        assert len(result.nodes) == 1

    def test_filter_and(self, table_name: str) -> None:
        vs = self._make_store(table_name)
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="category", value="science", operator=FilterOperator.EQ),
                MetadataFilter(key="priority", value=3, operator=FilterOperator.GT),
            ],
            condition=FilterCondition.AND,
        )
        q = VectorStoreQuery(query_embedding=_fake_embedding("test"), similarity_top_k=10, filters=filters)
        result = vs.query(q)
        assert len(result.nodes) == 1

    def test_filter_or(self, table_name: str) -> None:
        vs = self._make_store(table_name)
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="category", value="art", operator=FilterOperator.EQ),
                MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ),
            ],
            condition=FilterCondition.OR,
        )
        q = VectorStoreQuery(query_embedding=_fake_embedding("test"), similarity_top_k=10, filters=filters)
        result = vs.query(q)
        assert len(result.nodes) == 2


# ---- .table Escape Hatch ----


class TestTableProperty:
    def test_table_returns_pxt_table(self, table_name: str) -> None:
        vs = PixeltableVectorStore(table_name=table_name, embed_dim=EMBED_DIM)
        vs.add([_make_node("n1", "test")])
        t = vs.table
        assert t is not None
        assert t.count() == 1

    def test_table_data_accessible(self, table_name: str) -> None:
        vs = PixeltableVectorStore(table_name=table_name, embed_dim=EMBED_DIM)
        vs.add([_make_node("n1", "hello"), _make_node("n2", "world")])
        t = vs.table
        rows = t.select(t.text, t.node_id).collect()
        assert len(rows) == 2
        texts = {r["text"] for r in rows}
        assert texts == {"hello", "world"}

    def test_where_on_table(self, table_name: str) -> None:
        vs = PixeltableVectorStore(table_name=table_name, embed_dim=EMBED_DIM)
        vs.add(
            [
                _make_node("n1", "doc1", {"score": 10}),
                _make_node("n2", "doc2", {"score": 20}),
                _make_node("n3", "doc3", {"score": 30}),
            ]
        )
        t = vs.table
        rows = t.where(t.metadata["score"] > 15).select(t.text).collect()
        assert len(rows) == 2
