"""Integration tests for PixeltableVectorStore.

Requires:
- A running Pixeltable instance
- OPENAI_API_KEY environment variable
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason='OPENAI_API_KEY not set',
)


@pytest.fixture
def clean_table():
    """Ensure a clean table for each test, and clean up after."""
    import pixeltable as pxt
    table_name = 'test_li_integration.docs'
    try:
        pxt.drop_table(table_name, force=True)
    except Exception:
        pass
    try:
        pxt.drop_dir('test_li_integration', force=True)
    except Exception:
        pass
    yield table_name
    try:
        pxt.drop_table(table_name, force=True)
    except Exception:
        pass
    try:
        pxt.drop_dir('test_li_integration', force=True)
    except Exception:
        pass


class TestPixeltableVectorStoreIntegration:
    def test_add_and_query(self, clean_table):
        from llama_index.core.schema import TextNode
        from llama_index.core.vector_stores.types import VectorStoreQuery
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.vector_stores.pixeltable import PixeltableVectorStore

        embed_model = OpenAIEmbedding(model_name='text-embedding-3-small')

        store = PixeltableVectorStore(table_name=clean_table, embed_dim=1536)

        texts = [
            'Pixeltable is multimodal data infrastructure',
            'LlamaIndex builds RAG applications',
            'Vector stores hold embeddings',
        ]
        nodes = []
        for i, text in enumerate(texts):
            embedding = embed_model.get_text_embedding(text)
            node = TextNode(text=text, id_=f'node_{i}')
            node.embedding = embedding
            nodes.append(node)

        ids = store.add(nodes)
        assert len(ids) == 3

        query_embedding = embed_model.get_query_embedding('multimodal data')
        query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=2)
        result = store.query(query)

        assert len(result.nodes) == 2
        assert 'Pixeltable' in result.nodes[0].get_content()
        assert len(result.similarities) == 2
        assert all(0.0 <= s <= 1.0 for s in result.similarities)

    def test_delete_by_ref_doc_id(self, clean_table):
        import pixeltable as pxt
        from llama_index.core.schema import TextNode
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.vector_stores.pixeltable import PixeltableVectorStore

        embed_model = OpenAIEmbedding(model_name='text-embedding-3-small')
        store = PixeltableVectorStore(table_name=clean_table, embed_dim=1536)

        from llama_index.core.schema import RelatedNodeInfo, NodeRelationship
        node = TextNode(text='doc to delete', id_='del_node')
        node.embedding = embed_model.get_text_embedding('doc to delete')
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id='doc_1')

        store.add([node])

        t = pxt.get_table(clean_table)
        assert t.count() == 1

        store.delete('doc_1')
        assert t.count() == 0

    def test_delete_nodes(self, clean_table):
        import pixeltable as pxt
        from llama_index.core.schema import TextNode
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.vector_stores.pixeltable import PixeltableVectorStore

        embed_model = OpenAIEmbedding(model_name='text-embedding-3-small')
        store = PixeltableVectorStore(table_name=clean_table, embed_dim=1536)

        nodes = []
        for i in range(3):
            node = TextNode(text=f'node {i}', id_=f'n_{i}')
            node.embedding = embed_model.get_text_embedding(f'node {i}')
            nodes.append(node)

        store.add(nodes)
        t = pxt.get_table(clean_table)
        assert t.count() == 3

        store.delete_nodes(node_ids=['n_0', 'n_2'])
        assert t.count() == 1

    def test_clear(self, clean_table):
        import pixeltable as pxt
        from llama_index.core.schema import TextNode
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.vector_stores.pixeltable import PixeltableVectorStore

        embed_model = OpenAIEmbedding(model_name='text-embedding-3-small')
        store = PixeltableVectorStore(table_name=clean_table, embed_dim=1536)

        node = TextNode(text='clear test', id_='clear_node')
        node.embedding = embed_model.get_text_embedding('clear test')
        store.add([node])

        t = pxt.get_table(clean_table)
        assert t.count() == 1

        store.clear()
        assert t.count() == 0
