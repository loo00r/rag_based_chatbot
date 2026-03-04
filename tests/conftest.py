import sys, os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../app"))

# Mock heavy external deps before core.rag is imported (module-level constructors)
for _m in ["langchain_community.embeddings", "langchain_postgres", "langchain_openai"]:
    sys.modules.setdefault(_m, MagicMock())
