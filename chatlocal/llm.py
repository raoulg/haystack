import os
import random
from pathlib import Path

import tiktoken
from builder import DocumentstoreBuilder
from loguru import logger
from settings import Clusters, PromtOptions
from tqdm import tqdm

from haystack import Document
from haystack.nodes import AnswerParser, PromptNode, PromptTemplate
from haystack.pipelines import Pipeline


class KeywordExtraction:
    def __init__(self, options: PromtOptions, apikey: str):
        self.apikey = apikey
        self.enc = self.get_encoder(options.keywordmodel)
        self.options = options

    def __call__(
        self, clusters: Clusters, builder: DocumentstoreBuilder, outputfile: Path
    ) -> list[str]:
        if not outputfile.exists():
            sample = self.sample_texts(clusters, builder)
            keywords = self.extract_keywords(sample, outputfile)
        else:
            logger.info(f"Loading keywords from {outputfile}")
            with outputfile.open("r") as f:
                keywords = f.read().splitlines()
        return keywords

    def sample_texts(self, clusters: Clusters, builder: DocumentstoreBuilder) -> dict:
        sample = {}
        for key in clusters.keys():
            sample[key] = self.text_from_cluster(clusters[key], builder)
        return sample

    def text_from_cluster(
        self, cluster: Clusters, builder: DocumentstoreBuilder
    ) -> dict:
        d = {}
        for i in range(len(cluster.center)):
            d[i] = builder.document_store.get_documents_by_id(ids=cluster.center[i])
        for i in range(len(cluster.random)):
            d[i] += builder.document_store.get_documents_by_id(ids=cluster.random[i])
        return d

    def select_chunks(self, maxlen: int, documents: list[Document]) -> list[Document]:
        # TODO this throws away the unselected documents;
        # could be usefull to keep them for a second round
        random.shuffle(documents)
        selected = []
        dlen = 0
        for d in documents:
            dlen += len(self.enc.encode(d.content))
            if dlen > maxlen:
                break
            selected.append(d)
        return selected

    def update_maxlen(self, prompt: str) -> int:
        headersize = len(self.enc.encode(prompt))
        return 3988 - headersize

    def get_encoder(self, retrievertag: str) -> tiktoken.Encoding:
        return tiktoken.encoding_for_model(retrievertag)

    def create_pipe(self, prompt: str, API_KEY: str) -> Pipeline:
        template = PromptTemplate(
            prompt=prompt,
            output_parser=AnswerParser(),
        )
        node = PromptNode(
            self.options.keywordmodel, default_prompt_template=template, api_key=API_KEY
        )
        pipe = Pipeline()
        pipe.add_node(component=node, name="prompt_node", inputs=["Query"])
        return pipe

    def extract_keywords(self, sample: dict, outputfile: Path) -> list[str]:
        logger.info(f"Retrieving API_KEY from {self.apikey}")
        API_KEY = os.environ.get(self.apikey)
        assert API_KEY is not None, f"API_KEY {self.apikey} not found"
        keywords = set()
        for key in sample.keys():
            logger.info(f"Running {key}")
            for setnum, vals in tqdm(sample[key].items()):
                logger.info(f"Running set {setnum} with {len(vals)} documents")
                prompt = self.build_promt(self.options)
                maxlen = self.update_maxlen(prompt)
                documents = self.select_chunks(maxlen, vals)
                logger.info(
                    f"Selected {len(documents)}/{len(vals)} documents, maxlen {maxlen}"
                )
                pipe = self.create_pipe(prompt, API_KEY)
                output = pipe.run(documents=documents)
                keywords.update(output["answers"][0].answer.split(","))

                if len(keywords) > 300:
                    avoid = set(random.sample(list(keywords), 300))
                else:
                    avoid = set(keywords)
                self.options.avoid = avoid
            kwlist = list(keywords)
            self.save(kwlist, outputfile)
        return kwlist

    def save(self, keywords: list[str], outputfile: Path) -> None:
        logger.info(f"Saving keywords to {outputfile}")
        with outputfile.open("w") as f:
            for kw in sorted(keywords):
                f.write(f"{kw}\n")

    def build_promt(self, options: PromtOptions):
        return f"""
        You are specialized in analyzing various pieces of information and providing precise summaries.
        Please determine the core theme in the following series of *-separated information fragments, which are delimited by triple backticks.
        Ensure your answer focuses on the topic and avoids including unrelated content. DO NOT write complete sentences.
        You should obey the following rules when doing this task:
        1. Keywords in your answer should related to the topic {options.main_topic};
        2. Your answer should include at most {options.l1} keywords;
        3. Each keyword should be at most {options.l2} words long;
        4. avoid already appeared theme keywords, marked inside ⟨⟩;
        5. Separate your output keywords with commas (,);
        Information:’ ’ ’{options.docfun}’ ’ ’
        Please avoid the following already appeared theme terms: ⟨{", ".join([k for k in options.avoid])}⟩
        Your response:
        """
