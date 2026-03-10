import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.pipeline import answer_question, ingest_documents

app = typer.Typer(help="静态 workflow RAG CLI")
console = Console()


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
    reranker: bool = typer.Option(False, help="Enable optional reranker placeholder"),
):
    """Single-turn hybrid RAG query."""
    result = answer_question(question, language=language, top_k=top_k, enable_reranker=reranker)

    console.print(Panel(result["answer"], title="Answer", expand=False))

    table = Table(title="Citations")
    table.add_column("#")
    table.add_column("source_file")
    table.add_column("chunk_index")
    for c in result["citations"]:
        table.add_row(str(c["index"]), c["source_file"], str(c["chunk_index"]))
    console.print(table)


@app.command()
def chat(
    language: str = typer.Option("zh-CN", "--language", "-l", help="zh-CN / en-US"),
    top_k: int = typer.Option(5, help="Top chunks for final context"),
    reranker: bool = typer.Option(False, help="Enable optional reranker placeholder"),
):
    """Multi-turn terminal chat."""
    console.print("[bold cyan]Static workflow RAG CLI chat started. 输入 exit 结束。[/bold cyan]")
    while True:
        question = typer.prompt("Q").strip()
        if question.lower() in {"exit", "quit"}:
            break
        result = answer_question(question, language=language, top_k=top_k, enable_reranker=reranker)
        console.print(Panel(result["answer"], title="Answer", expand=False))


if __name__ == "__main__":
    app()
