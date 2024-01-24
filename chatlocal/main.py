from pathlib import Path

from chatlocal.builder import DocumentstoreBuilder
from chatlocal.settings import Job, Settings


def main(job: Job):
    ada_model = "text-embedding-ada-002"

    settings = Settings(
        by="word",
        length=100,
        docstorepath=Path.home() / ".cache/chatlocal",
        retrievertag=ada_model,
        embedding_dim=1536,
        max_seq_length=1536,
        api_key_string="OPENAI_API_KEY_SCEPA",
    )

    builder = DocumentstoreBuilder(settings=settings)
    builder.add_files(job=job)


if __name__ == "__main__":
    job = Job(datadir=Path.home() / "Downloads/research", tag="scepa_100w")
    main(job=job)
