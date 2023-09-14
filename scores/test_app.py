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

arch_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ###Instruction:
    You are an expert programming assistant who is also an expert in the Star Wars universe and all the main heroes and villains from all the movies and shows. You will tell the truth, even if the truth is that you don't know.
    Given a person's GitHub activity data in JSON format, you must determine what Jedi class does the user fit based on their GitHub activity data (the four classes in order from best to worst are Jedi Master, Jedi Knight, Jedi Apprentice, and Jedi Padawan).
    The data includes the following fields:
    -repo_name: this has the name of the GitHub repository that the user has worked on.
    -branch_name: this has the name of each branch in the GitHub repository that the user has worked on.
    -commit_count: this shows the number of commits the user has made in the respective branch.
    -pull_count: this shows the number of pulls the user has made in the respective branch.
    -pull_file_count: this shows the total number of files affected by the user's pulls made in the respective branch.

    ###Input:
    GitHub Activity Data:
    {examples}
    {message_history}
    {current_message}

    ###Example Response:
    Jedi Knight, Anakin Skywalker, because you get a lot done as evidenced by the number of your commits.
    Jedi Master, Jocasta Nu, because you have large quantity of pull requests.
    Jedi Padawan, Qui Gon Jinn, because you have very few repositories in your activity history, but you have great potential to grow.

    ###Response:
    The Star Wars class and character that personifies this person is"""

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"inputs": prompt, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generation"]

@cl.author_rename
def rename(orig_author: str):
    rename_dict = {"Perspect Scores": "Get your scores from Perspect!"}
    return rename_dict.get(orig_author, orig_author)

cl.on_chat_start
async def init():
    msg = cl.Message(content=f"Setting up scores...")
    await msg.send()

    #build datasets and FAISS
    msg = cl.Message(content=f"Building FAISS indexes and loading data...")
    await msg.send()
    archetypes_ds = load_dataset('csv', data_files="./Archetypes/data/embedded_llama_faiss_ds.csv", split="train")
    archetypes_ds.load_faiss_index('embedding', './Archetypes/data/gpt_index.faiss')

    designpatterns_ds = load_dataset('csv', data_files="./DesignPatternDetector/data/embedded_gpt_faiss_ds.csv", split="train")
    designpatterns_ds.load_faiss_index('embedding', './DesignPatternDetector/data/gpt_index.faiss')

    professionalism_ds = load_dataset('csv', data_files='./PerspecToolformer/data/embedded_dataset.csv', split='train')
    professionalism_ds.load_faiss_index('embedding', './PerspecToolformer/data/professionalism_index.faiss')


    msg = cl.Message(content=f"Loading Models...")
    await msg.send()

    content_handler = ContentHandler()

    llm=SagemakerEndpoint(
            endpoint_name="jumpstart-dft-meta-textgeneration-llama-2-7b",
            region_name="us-west-2",
            model_kwargs={"parameters": {"max_new_tokens": 50}},
            content_handler=content_handler,
            endpoint_kwargs={"CustomAttributes":"accept_eula=true"}
        )

    EMBEDDER = Embed4All()

    msg = cl.Message(content=f"chaining models...")
    await msg.send()

    archetype_chain = RetrievalAugmentedQAPipeline(vector_db=archetypes_ds, llm=llm, verbose=False, embedder=EMBEDDER, template=)
    designpattern_chain = RetrievalAugmentedQAPipeline()
    professionalism_chain = RetrievalAugmentedQAPipeline()