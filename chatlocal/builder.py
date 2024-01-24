import os

from loguru import logger
from typing import Optional

from chatlocal.settings import Job, Settings
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import (
    DocxToTextConverter,
    EmbeddingRetriever,
    FileTypeClassifier,
    MarkdownConverter,
    PDFToTextConverter,
    PreProcessor,
    TextConverter,
)
from haystack.pipelines import Pipeline


class DocumentstoreBuilder:
    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.api_key_string is None:
            logger.info(
                f"No api key found with {settings.api_key_string}."
                "The builder will only work with local models."
            )

        file_type_classifier = FileTypeClassifier()
        text_converter = TextConverter()
        pdf_converter = PDFToTextConverter()
        md_converter = MarkdownConverter()
        docx_converter = DocxToTextConverter()
        preprocessor = PreProcessor(
            split_by=settings.by,
            split_length=settings.length,
            split_respect_sentence_boundary=True,
            language=settings.language,
            add_page_number=settings.add_page,
        )

        p = Pipeline()
        p.add_node(
            component=file_type_classifier, name="FileTypeClassifier", inputs=["File"]
        )
        p.add_node(
            component=text_converter,
            name="TextConverter",
            inputs=["FileTypeClassifier.output_1"],
        )
        p.add_node(
            component=pdf_converter,
            name="PdfConverter",
            inputs=["FileTypeClassifier.output_2"],
        )
        p.add_node(
            component=md_converter,
            name="MarkdownConverter",
            inputs=["FileTypeClassifier.output_3"],
        )
        p.add_node(
            component=docx_converter,
            name="DocxConverter",
            inputs=["FileTypeClassifier.output_4"],
        )

        p.add_node(
            component=preprocessor,
            name="Preprocessor",
            inputs=[
                "TextConverter",
                "PdfConverter",
                "MarkdownConverter",
                "DocxConverter",
            ],
        )
        self.pipeline = p
        self.document_store: Optional[FAISSDocumentStore] = None
        self.retriever = None

    def run_preprocessor(self, job: Job):
        files = [*job.datadir.glob("*")]
        logger.info(f"found {len(files)} files in {job.datadir}.")
        metadata = [{"filename": f.name} for f in files]
        result = self.pipeline.run(file_paths=files, meta=metadata)
        logger.info(f"retrieved {len(result['documents'])} document snippets.")
        return result

    def add_files(self, job: Job):
        result = self.run_preprocessor(job)

        if not self.document_store:
            self.get_docstore(job)
        assert self.document_store is not None
        self.document_store.write_documents(documents=result["documents"])

        if not self.retriever:
            self.get_retriever(top_k=self.settings.top_k)
        logger.info("updating embeddings...")
        self.document_store.update_embeddings(
            retriever=self.retriever, update_existing_embeddings=False, batch_size=64
        )
        logger.info("saving docstore...")
        _, index_path, config_path = self._get_paths(job.tag)
        self.document_store.save(index_path=index_path, config_path=config_path)

    def _get_paths(self, tag: str):
        docstore = self.settings.docstorepath
        sql_url = f"sqlite:///{docstore}/{tag}.db"
        index_path = docstore / f"{tag}.faiss"
        config_path = docstore / f"{tag}.json"
        return sql_url, index_path, config_path

    def get_docstore(self, job: Job):
        docstore = self.settings.docstorepath
        if not docstore.exists():
            logger.info(f"creating docstorefolder at {docstore}")
            docstore.mkdir()

        sql_url, index_path, config_path = self._get_paths(job.tag)

        if not index_path.exists():
            logger.info(f"creating FAISS docstore {sql_url}")
            self.document_store = FAISSDocumentStore(
                sql_url=sql_url,
                faiss_index_factory_str="Flat",
                embedding_dim=self.settings.embedding_dim,
            )
        else:
            logger.info(f"loading existing FAISS docstore {job.tag} from {index_path}")
            self.document_store = FAISSDocumentStore.load(
                index_path=index_path, config_path=config_path
            )

        logger.info(f"docstore has {self.document_store.get_document_count()} docs.")

    def get_retriever(self, top_k: int):
        logger.info(f"Using {self.settings.api_key_string} as api key")
        API_KEY = os.environ.get(self.settings.api_key_string, None)

        self.retriever = EmbeddingRetriever(
            document_store=self.document_store,
            embedding_model=self.settings.retrievertag,
            batch_size=self.settings.retriever_batch_size,
            api_key=API_KEY,
            top_k=top_k,
            max_seq_len=self.settings.max_seq_length,
        )