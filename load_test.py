import asyncio
import time
import statistics
from core.agent import graph

QUERIES = [
    "What is LangGraph?",
    "How does RAG work?",
    # TODO: expand to 100 queries
] * 50


async def run_query(q: str) -> float:
    t = time.perf_counter()
    await asyncio.to_thread(
        graph.invoke,
        {"query": q, "sub_queries": [], "rag_answers": [], "web_results": [], "final_answer": "", "iterations": 0}
    )
    return time.perf_counter() - t


async def main():
    times = await asyncio.gather(*[run_query(q) for q in QUERIES])
    s = sorted(times)
    n = len(s)
    print(f"n={n}  p50={s[n//2]:.1f}s  p95={s[int(n*.95)]:.1f}s  p99={s[int(n*.99)]:.1f}s")
    print(f"throughput={n/sum(times)*n:.2f} req/s")


if __name__ == "__main__":
    asyncio.run(main())
