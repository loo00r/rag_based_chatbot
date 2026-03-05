import re
import requests
from bs4 import BeautifulSoup
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.config import settings, PG_CONN

URL = "https://zakon.rada.gov.ua/laws/show/1306-2001-п/print"
RULE_RE = re.compile(r"^(\d+\.\d+(?:\.\d+)*)\.")


def is_section(text: str) -> bool:
    # "7. ПРІОРИТЕТ..." — число + крапка + пробіл + текст у верхньому регістрі
    match = re.match(r"^\d+\.\s+(.+)$", text)
    if not match:
        return False
    body = match.group(1).replace("'", "").replace("'", "").replace(" ", "").replace("-", "")
    return len(body) > 2 and body.isupper()


def parse_docs(html: str) -> list[Document]:
    soup = BeautifulSoup(html, "html.parser")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP, separators=["\n\n", "\n", " ", ""]
    )
    docs = []
    current_section = "ПДР"
    current_rule_id = ""
    current_lines: list[str] = []

    def flush():
        if not (current_rule_id and current_lines):
            return
        content = f"{current_section}\n{' '.join(current_lines)}"
        if len(content) > settings.CHUNK_SIZE * 2:
            for i, chunk in enumerate(splitter.split_text(content)):
                docs.append(Document(
                    page_content=chunk,
                    metadata={"section": current_section, "rule_id": f"{current_rule_id}.{i}"},
                ))
        else:
            docs.append(Document(
                page_content=content,
                metadata={"section": current_section, "rule_id": current_rule_id},
            ))

    for tag in soup.find_all("p"):
        text = tag.get_text(" ", strip=True)
        if not text:
            continue

        if is_section(text):
            flush()
            current_section = text
            current_rule_id = ""
            current_lines = []
            continue

        rule_match = RULE_RE.match(text)
        if rule_match:
            flush()
            current_rule_id = rule_match.group(1)
            current_lines = [text]
        elif current_rule_id:
            current_lines.append(text)

    flush()
    return docs


if __name__ == "__main__":
    print("Fetching ПДР з zakon.rada.gov.ua...")
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"}
    html = requests.get(URL, timeout=30, headers=headers).text
    docs = parse_docs(html)
    print(f"Parsed {len(docs)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBED_MODEL)
    store = PGVector(embeddings=embeddings, collection_name=settings.COLLECTION, connection=PG_CONN)
    store.delete_collection()
    store.create_collection()
    store.add_documents(docs)
    print(f"Ingested {len(docs)} chunks into '{settings.COLLECTION}'")
