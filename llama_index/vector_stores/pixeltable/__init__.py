"""LlamaIndex VectorStore backed by Pixeltable.

Install:
    pip install llama-index-vector-stores-pixeltable

Usage:
    from llama_index.vector_stores.pixeltable import PixeltableVectorStore

    vector_store = PixeltableVectorStore(table_name="mydir.docs", embed_dim=1536)
"""

from llama_index.vector_stores.pixeltable.base import PixeltableVectorStore

__all__ = ['PixeltableVectorStore']
__version__ = '0.1.1'
