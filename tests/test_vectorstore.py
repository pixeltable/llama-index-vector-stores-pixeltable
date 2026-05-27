"""Unit tests for PixeltableVectorStore."""

from llama_index.vector_stores.pixeltable import PixeltableVectorStore


def test_import():
    """Verify the package exports PixeltableVectorStore."""
    assert hasattr(PixeltableVectorStore, "add")
    assert hasattr(PixeltableVectorStore, "delete")
    assert hasattr(PixeltableVectorStore, "query")
    assert hasattr(PixeltableVectorStore, "delete_nodes")
    assert hasattr(PixeltableVectorStore, "clear")


def test_version():
    """Verify version is set."""
    import llama_index.vector_stores.pixeltable

    assert llama_index.vector_stores.pixeltable.__version__ == "0.1.0"
