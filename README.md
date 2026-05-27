# llama-index-vector-stores-pixeltable

LlamaIndex VectorStore integration backed by [Pixeltable](https://github.com/pixeltable/pixeltable) -- multimodal data infrastructure with built-in embedding indexes, metadata filtering, computed column lineage, and incremental computation.

## Installation

```bash
pip install llama-index-vector-stores-pixeltable
```

## Quick Start

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.pixeltable import PixeltableVectorStore

# Create the vector store
vector_store = PixeltableVectorStore(
    table_name="mydir.docs",
    embed_dim=1536,
)

# Load documents and build index
storage_context = StorageContext.from_defaults(vector_store=vector_store)
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is Pixeltable?")
print(response)
```

## Filtered Queries with MetadataFilters

`MetadataFilters` on `query()` map to Pixeltable's `.where()` clause -- predicates are evaluated **before** ranking:

```python
from llama_index.core.vector_stores.types import (
    VectorStoreQuery, MetadataFilters, MetadataFilter, FilterOperator,
)

filters = MetadataFilters(filters=[
    MetadataFilter(key="category", value="science", operator=FilterOperator.EQ),
])
result = vector_store.query(VectorStoreQuery(
    query_embedding=embedding,
    similarity_top_k=5,
    filters=filters,
))
```

Supported operators: `==`, `!=`, `>`, `<`, `>=`, `<=` with AND/OR conditions.

## Access the Underlying Pixeltable Table

The `.table` property gives direct access to the Pixeltable table for operations beyond the VectorStore interface -- computed columns, lineage, version history, and arbitrary predicates:

```python
import pixeltable as pxt

t = vector_store.table

# Inspect all data
t.select(t.text, t.metadata, t.embedding).collect()

# Add a computed column -- auto-backfills all existing rows
t.add_computed_column(modality=extract_modality(t.metadata))

# WHERE on computed columns + similarity
import numpy as np
sim = t.embedding.similarity(vector=np.array(query_vec, dtype=np.float32))
results = (
    t.where(t.modality == "image")
    .order_by(sim, asc=False)
    .limit(5)
    .select(t.text, t.modality, sim=sim)
    .collect()
)
```

## Connect to an Existing Pixeltable Table

```python
vector_store = PixeltableVectorStore(table_name="mydir.existing_docs")
index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine()
```

## Why Pixeltable as a Vector Backend?

- **Metadata filtering via `.where()`**: Filter on metadata fields *before* ranking, not post-hoc
- **Computed column lineage**: Add derived columns that auto-backfill and auto-compute on new inserts
- **Persistent and versioned**: Data survives restarts; every change is tracked
- **Incremental**: Only new/changed rows get re-embedded
- **Multimodal native**: Images, video, audio, and documents alongside text
- **Any embedding model**: Works with OpenAI, Hugging Face, or any local model
- **No external services**: Embedded PostgreSQL, no Docker required

## Links

- [Pixeltable Docs](https://docs.pixeltable.com/)
- [LlamaIndex Integration Docs](https://docs.pixeltable.com/libraries/llamaindex)
- [GitHub](https://github.com/pixeltable/pixeltable)
- [Discord](https://discord.gg/QPyqFYx2UN)
