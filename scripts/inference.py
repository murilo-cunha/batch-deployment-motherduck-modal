"""Run batch inference - inspired by `https://modal.com/docs/examples/batch_inference_using_huggingface`."""

from datetime import datetime
import os
import modal
from pathlib import Path

stub = modal.Stub(
    "batch-duckdb-huggingface",
    image=modal.Image.debian_slim().pip_install_from_pyproject(
        str(Path(__file__).parents[1] / "pyproject.toml")
    ),
)


@stub.cls(cpu=8, retries=3)
class ToxicityAnalysis:
    """Toxic comment classification using Huggingface's transformers."""

    @modal.enter()
    def setup_pipeline(self):
        """Download and setup pretrained pipeline."""
        from transformers import pipeline

        self.pipe = pipeline(
            "text-classification", model="martin-ha/toxic-comment-model"
        )

    @modal.method()
    def predict(self, phrase: str):
        """Compute probabilites for positive class."""
        pred = self.pipe(phrase, truncation=True, max_length=512, top_k=2)
        # pred will look like: [{'label': 'toxic', 'score': 0.99}, {'label': 'non-toxic', 'score': 0.01}]
        probs = {p["label"]: p["score"] for p in pred}
        return probs["toxic"]


@stub.function(secrets=[modal.Secret.from_name("motherduck-token")])
def read_data():
    """Read data from DuckDB."""
    import duckdb

    with duckdb.connect(
        f"md:mlops-demo?motherduck_token={os.environ['MOTHERDUCK_TOKEN']}"
    ) as conn:
        return conn.sql(
            "SELECT id, text, timestamp"
            " FROM main.hn_starter LEFT JOIN main.predictions USING(id)"
            " WHERE prediction IS NULL LIMIT 10"
        ).fetchall()


@stub.function(secrets=[modal.Secret.from_name("motherduck-token")])
def write_data(table: str, values: list[tuple[int, float, datetime]]) -> None:
    """Write data to target table in DuckDB."""
    import duckdb

    with duckdb.connect(
        f"md:mlops-demo?motherduck_token={os.environ['MOTHERDUCK_TOKEN']}"
    ) as conn:
        conn.executemany(f"INSERT INTO {table} VALUES (?, ?, ?)", values)


@stub.local_entrypoint()
def main():
    print("Downloading data...")
    data = read_data.remote()
    print("Got", len(data), "reviews")
    ids, content, _ = zip(*data)
    predictor = ToxicityAnalysis()
    print("Running batch prediction...")
    predictions = list(predictor.predict.map(content))
    print("Writing results...")
    write_data.remote(
        table="main.predictions",
        values=zip(ids, predictions, [datetime.now()] * len(ids)),
    )
