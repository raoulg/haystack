import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken
from builder import DocumentstoreBuilder
from loguru import logger
from openai import OpenAI
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
                answer = output["answers"][0].answer
                if re.search(r"[eE]nergy [pP]overty", answer):
                    logger.debug(f"Answer: {answer}")
                    logger.debug(f"documnets: \n {[d.content for d in documents]}")
                keywords.update(answer.split(","))

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


@dataclass
class KeywordCleanup:
    header: str
    dedup: str
    splitting: str
    deletion: str

    def __call__(
        self, keywords: list[str], API_KEY: str, outputfile: Path
    ) -> list[str]:
        if outputfile.exists():
            logger.info(f"Loading cleaned keywords from {outputfile}")
            keywords = self.load(outputfile)
            return keywords

        client = OpenAI(api_key=API_KEY)
        logger.info(f"Starting with {len(keywords)} keywords")
        for task in ["dedup", "splitting", "deletion"]:
            logger.info(f"Running {task}")
            p = self.create(keywords, task)
            kw = self.get_completion(client, p)
            keywords = kw.split(",")
            logger.info(f"Got {len(keywords)} keywords")
        kw_clean = self.clean(keywords)
        with outputfile.open("w") as f:
            f.write("\n".join(kw_clean))
        logger.info(f"Saved {len(kw_clean)} keywords to {outputfile}")
        return kw_clean

    def load(self, outputfile: Path) -> list[str]:
        with outputfile.open("r") as f:
            keywords = f.read().splitlines()
        return keywords

    def create(self, keywords: list[str], taskname: str) -> list[dict[str, str]]:
        task = getattr(self, taskname)
        prompt = f"""Instructions:
                {task}
                Input Keywords:
                {", ".join(keywords)}
                """
        msg = [
            {"role": "system", "content": self.header},
            {"role": "user", "content": prompt},
        ]

        return msg

    @staticmethod
    def get_completion(client, p) -> str:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=p,
        )
        msg = completion.choices[0].message.content
        return msg

    @staticmethod
    def clean(keywords: list[str]) -> list[str]:
        return [re.sub(r"^\s+", "", kw.lower()) for kw in keywords]


HEADER = """You have been provided a list of keywords, each separated by a comma. Your task is to process this list according to a set of guidelines designed to refine and improve the list's utility.
Upon completion of your task, the output should be a processed list of keywords. Each keyword should be separated by a comma.
It's crucial that you maintain the same formatting (separate keywords by comma) to ensure the usability of the processed list.
Don't response to anything other than processed keywords. Each of your processed keywords should have at most 3 words."""
DEDUP = """'Concentration and Deduplication':
Your task is to examine each keyword in the list and identify those that are either identical or extremely similar in meaning. These should be treated in two ways:
- For keywords that are closely related, expressing different aspects of the same core idea, you need to 'concentrate' them into a single term that best captures the overall concept. For instance, consider the keywords <Styrofoam recycling>, <Styrofoam packaging recycling>, <Styrofoam recycling machine>, <Foam recycling>, <recycled Styrofoam products>. These all relate to the central concept of 'Styrofoam recycling' and should be consolidated into this single keyword.
- For keywords that are identical or nearly identical in meaning, remove the duplicates so that only one instance remains. The guideline here is: if two keywords convey almost the same information, retain only one.
Remember, the objective of this step is to trim the list by eliminating redundancy and focusing on the core concepts each keyword represents."""
SPLITTING = """'Splitting':
Sometimes, a keyword might be made up of an entity and another keyword, each of which independently conveys a meaningful concept. In these cases, you should split them into two separate keywords.
For instance, consider a keyword like 'Apple recycling'. Here, 'Apple' is a distinct entity, and 'recycling' is a separate concept. Therefore, it's appropriate to split this keyword into two: 'Apple' and 'recycling'.
However, when you split a keyword, be sure to check if the resulting terms are already present in the list you have generated. If they are, remove the duplicates to ensure the list remains concise and unique. Always aim to avoid redundancy in the list."""
DELETION = """'Deletion':
You will encounter keywords that are either too vague or represent an overly broad concept. These should be removed from the list.
For instance, a keyword like 'things' or 'stuff' is generally too vague to be useful. Similarly, a keyword like 'technology' might be too broad unless it's used in a specific context or in conjunction with other terms."""


keywordcleanup = KeywordCleanup(
    header=HEADER,
    dedup=DEDUP,
    splitting=SPLITTING,
    deletion=DELETION,
)
