import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.agent import AgentState, MinimalDocumentQAAgent
from app.config import EVAL_FILE
from app.pipeline import answer_question, ingest_documents
from app.session import SessionMemory
from app.tracing import read_traces

app = typer.Typer(help="静态 workflow RAG CLI")
console = Console()


def _print_citations(citations: list[dict]) -> None:
    table = Table(title="Citations")
    table.add_column("#")
    table.add_column("source_file")
    table.add_column("chunk_index")
    table.add_column("rrf")
    table.add_column("rerank")
    for c in citations:
        rerank_score = "" if c.get("rerank_score") is None else f"{c['rerank_score']:.3f}"
        table.add_row(
            str(c["index"]),
            c["source_file"],
            str(c["chunk_index"]),
            f"{c.get('rrf_score', 0):.4f}",
            rerank_score,
        )
    console.print(table)


@app.command()
def ingest(path: str):
    """Import .txt/.md/.docx docs and build local indexes."""
    result = ingest_documents(path)
    console.print(f"[green]Ingest done[/green] files={result['files']} chunks={result['chunks']}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    language: str = typer.Option("zh-CN", "--language", "-l", help="zh-CN / en-US"),
    top_k: int = typer.Option(5, help="Top chunks for final context"),
    reranker: bool = typer.Option(False, help="Enable true reranker"),
    rerank_top_n: int = typer.Option(8, help="How many candidates to rerank"),
    rewrite: bool = typer.Option(False, help="Enable conversational query rewrite"),
    show_rewrite: bool = typer.Option(False, help="Show rewritten query"),
    trace: bool = typer.Option(False, help="Enable trace logging"),
    debug: bool = typer.Option(False, help="Show retrieval/rerank debug details"),
):
    """Single-turn hybrid RAG query with optional rewrite/rerank/trace."""
    rewritten = None
    if rewrite:
        rewritten = SessionMemory().rewrite_question(question)
    result = answer_question(
        question,
        language=language,
        top_k=top_k,
        enable_reranker=reranker,
        rerank_top_n=rerank_top_n,
        rewrite_question=rewritten,
        trace_enabled=trace,
        debug=debug,
    )

    if show_rewrite and rewritten:
        console.print(f"[yellow]Rewritten Query:[/yellow] {result['retrieval_question']}")

    console.print(Panel(result["answer"], title=f"Answer (trace={result['trace_id']})", expand=False))
    _print_citations(result["citations"])

    if reranker and result.get("rerank_before"):
        console.print("[bold]Rerank Before[/bold]: " + ", ".join(result["rerank_before"]))
        console.print("[bold]Rerank After [/bold]: " + ", ".join(result["rerank_after"]))

    if debug:
        console.print_json(data=result["debug"])


@app.command()
def chat(
    language: str = typer.Option("zh-CN", "--language", "-l", help="zh-CN / en-US"),
    top_k: int = typer.Option(5, help="Top chunks for final context"),
    reranker: bool = typer.Option(False, help="Enable true reranker"),
    rerank_top_n: int = typer.Option(8, help="How many candidates to rerank"),
    trace: bool = typer.Option(False, help="Enable trace logging"),
    show_rewrite: bool = typer.Option(True, help="Show rewritten query each turn"),
):
    """Conversational RAG chat with session memory, rewrite, and history compression."""
    session = SessionMemory()
    console.print("[bold cyan]Conversational RAG chat started. 输入 exit 结束。[/bold cyan]")
    while True:
        question = typer.prompt("Q").strip()
        if question.lower() in {"exit", "quit"}:
            break

        rewritten = session.rewrite_question(question)
        result = answer_question(
            question,
            language=language,
            top_k=top_k,
            enable_reranker=reranker,
            rerank_top_n=rerank_top_n,
            rewrite_question=rewritten,
            trace_enabled=trace,
        )
        session.add_turn(question, result["answer"])

        if show_rewrite:
            console.print(f"[yellow]Rewritten Query:[/yellow] {result['retrieval_question']}")

        console.print(Panel(result["answer"], title=f"Answer (trace={result['trace_id']})", expand=False))
        _print_citations(result["citations"])


@app.command("agent-chat")
def agent_chat(
    language: str = typer.Option("zh-CN", "--language", "-l", help="zh-CN / en-US"),
    reranker: bool = typer.Option(True, help="Enable reranker in agent mode"),
    trace: bool = typer.Option(True, help="Enable trace in agent mode"),
    show_steps: bool = typer.Option(True, help="Show agent step/tool execution"),
):
    """Minimal Document QA Agent chat (state + router + planner + loop + stop)."""
    state = AgentState(language=language)
    agent = MinimalDocumentQAAgent(state=state)
    console.print("[bold magenta]Agent chat started. 输入 exit 结束。[/bold magenta]")
    while True:
        question = typer.prompt("AgentQ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        result = agent.run(question, show_steps=show_steps, enable_reranker=reranker, trace_enabled=trace)
        console.print(Panel(result["answer"], title=f"Agent Answer (trace={result['trace_id']})", expand=False))
        _print_citations(result.get("citations", []))
        if show_steps:
            console.print_json(data={"agent_steps": result.get("agent_steps", [])})


@app.command("trace-show")
def trace_show(limit: int = typer.Option(10, help="How many recent traces to show")):
    """Show recent local traces for observability/diagnostics."""
    traces = read_traces(limit=limit)
    if not traces:
        console.print("[yellow]No traces found.[/yellow]")
        return
    for item in traces:
        console.print_json(data=item)


@app.command()
def evaluate(
    language: str = typer.Option("zh-CN", "--language", "-l"),
    reranker: bool = typer.Option(False),
    trace: bool = typer.Option(False),
):
    """Run a minimal offline eval from data/eval_set.jsonl for A/B style checks."""
    if not EVAL_FILE.exists():
        console.print(f"[red]Missing eval file: {EVAL_FILE}[/red]")
        raise typer.Exit(code=1)

    total = 0
    answered = 0
    grounded = 0
    with EVAL_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            result = answer_question(
                row["question"],
                language=language,
                enable_reranker=reranker,
                trace_enabled=trace,
            )
            if "证据不足" not in result["answer"] and "Insufficient evidence" not in result["answer"]:
                answered += 1
            if result["citations"]:
                grounded += 1

    table = Table(title="Minimal Eval")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("total", str(total))
    table.add_row("answered_rate", f"{answered / max(total, 1):.2%}")
    table.add_row("citation_rate", f"{grounded / max(total, 1):.2%}")
    console.print(table)


if __name__ == "__main__":
    app()
