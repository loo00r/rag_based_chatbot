from core.config import PG_CONN, EMBED_MODEL, COLLECTION, CHUNK_SIZE, CHUNK_OVERLAP

# TODO: load docs → chunk → embed → upsert into pgvector
# from langchain_postgres import PGVector
# from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter

if __name__ == "__main__":
    print(f"Ingesting into {PG_CONN}, collection={COLLECTION}")
