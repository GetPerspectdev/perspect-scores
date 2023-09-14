from langchain import PromptTemplate, LLMChain, SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler

from typing import Dict

from datasets import load_dataset, Dataset

import chainlit as cl

import json

from gpt4all import Embed4All
import numpy as np

import os

import nest_asyncio
nest_asyncio.apply()

from Archetypes.web_app import predict_archetype
from DesignPatternDetector.app import get_github, get_files, CODE_FILES, code_prompt_template
from SlackSentiment.app import allowed, Perspective, Comment, Attribute, Span, PerspectiveAPIException
from PerspecToolformer.chainlit_test import number_template, embed_texts, get_relevant_documents, RetrievalAugmentedQAPipeline, predict_convo

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"inputs": prompt, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generation"]


content_handler = ContentHandler()

llm=SagemakerEndpoint(
        endpoint_name="jumpstart-dft-meta-textgeneration-llama-2-7b",
        region_name="us-west-2",
        model_kwargs={"parameters": {"max_new_tokens": 50}},
        content_handler=content_handler,
        endpoint_kwargs={"CustomAttributes":"accept_eula=true"}
    )

EMBEDDER = Embed4All()