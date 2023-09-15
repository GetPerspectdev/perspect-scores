from langchain import PromptTemplate, LLMChain, SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler

import requests
from bs4 import BeautifulSoup
from collections import Counter

from urllib.parse import urljoin

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
from DesignPatternDetector.app import CODE_FILES
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
    {data}

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
    
def get_github(repo_url, dev_key='', branch="main", verbose=False, dataset=None, llm=None, embedder=None):
    url_tree = [repo_url]
    file_urls = []
    contents = []
    for i in url_tree:
        r = requests.get(i)
        if r.status_code == 200:
            if verbose:
                print(f"Creating url tree: {i}")
            soup = BeautifulSoup(r.content, 'html.parser')
            if len(soup.find_all('a')) > 0:
                for j in soup.find_all('a'):
                        try:
                            if f"tree/{branch}" in j.get('href'):
                                url = urljoin(i, j.get('href'))
                                if url not in url_tree:
                                    url_tree.append(url)
                        except TypeError:
                            if verbose:
                                print(f"Error on original repo")
            else:
              stuff = json.loads(str(soup))
              try:
                for j in stuff['payload']['tree']['items']:
                    try:
                        if f"{i.split('/')[-1]}" in j['path'] and len(j['path'].split(".")) == 1:
                            url = f"{i}/{j['name']}"
                            if url not in url_tree:
                                    url_tree.append(url)
                    except TypeError:
                        if verbose:
                            print(f"Error on {j}")
              except KeyError:
                    if verbose:
                        print(f"Error on {i}")
        else:
            print(f"Problem with {repo_url}")
    for tree_url in url_tree:
        if verbose:
                print(f"Tree: {tree_url}")
        r = requests.get(tree_url)
        soup = BeautifulSoup(r.content, 'html.parser')
        if len(soup.find_all('a')) > 0:
            for i in soup.find_all('a'):
                try:
                    if f"blob/{branch}" in i.get('href'):
                        url = urljoin(repo_url, i.get('href'))
                        if url not in file_urls:
                            file_urls.append(url)
                            if verbose:
                                print(f"File: {url.split('/')[-1]}")
                except TypeError:
                    if verbose:
                        print(f"Error on {i}")
        else:
            stuff = json.loads(str(soup))
            try:
                for j in stuff['payload']['tree']['items']:
                    try:
                        if j['contentType'] == 'file':
                            url = f"{tree_url}/{j['name']}"
                            if url not in file_urls:
                                    file_urls.append(url)
                                    if verbose:
                                        print(f"File: {url.split('/')[-1]}")
                    except TypeError:
                        if verbose:
                            print(f"Error on {j}")
            except KeyError:
                if verbose:
                    print(f"Error on {i}")

    for url in file_urls:
        if url != repo_url:
            url = url.replace ("tree/", "")
        url = (
            url
            .replace("github.com", "raw.githubusercontent.com")
            .replace(" ", "%20")
            .replace("blob/", "")
        )

        if any(ext in url for ext in CODE_FILES) or url.endswith('.ts') or url.endswith('.js'):
            try:
                r = requests.get(url)
                if verbose:
                    print(f"Content: {url}\nStatus: {r.status_code}")
                if r.status_code == 200:
                    file_content = BeautifulSoup(r.content, 'html.parser')
                    contents.append(file_content)
                else:
                    contents.append('')
            except:
                pass
        else:
            contents.append('')
            continue

        
    
    files = {}
    code_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            ###Instruction:
            You are an expert programming assistant who will tell the truth, even if the truth is that they don't know.
            Given some code, you must determine if it is close to a particular design pattern in that programming language, and if so, how they can refactor that code into code that follows a design pattern.
            If the code does not match a design pattern, you should give a recommendation as to which design pattern should be used in this instance.

            ###Input:
            {code}

            ###Response:
            The closest design pattern to this code is"""
    code_prompt_template = PromptTemplate(input_variables=['code'], template=code_template)
    code_chain = LLMChain(llm=llm, prompt=code_prompt_template)

    for file, content in zip(file_urls, contents):
        if content == '':
             continue
        embed = embedder.embed(content)
        query = np.array(embed, dtype=np.float32)
        score, samples = dataset.get_nearest_examples('embedding', query, k=1)
        files[file] = {'score': score, 'samples': samples, 'content': content, 'llm_out': ''}
        if verbose:
            print(f"Score: {score} {file.split('/')[-1]}")
        if score <= 1.3 and score >= 1.1:
            files[file]['llm_out'] = code_chain.run(content)
    num_files = len(file_urls)

    pp = []
    scores = []
    patterns = []
    resources = []
    for k, v in files.items():
        scores.append(v['score'])
        patterns.append(v['samples']['Design Pattern'])
        # try: 
        #     pp.append(f"Name: {k.split('/')[-1]} | Score: {v['score']} | Closest: {v['samples']['Language']} {v['samples']['Design Pattern']} | Model: {v['model_out']} |")
        # except KeyError:
        if v['score'] > 1.0:
            pp.append(
                {
                      "name": k.split('/')[-1], 
                      "score": float(v['score']), 
                      "language": v['samples']['Language'], 
                      "pattern": v['samples']['Design Pattern'],
                      "resource": v['samples']['Unnamed: 4'],
                      "llm_out": v['llm_out']
                }
            )
        else:
            pp.append(
                {
                    "name": k.split('/')[-1], 
                    "score": float(v['score']), 
                    "language": v['samples']['Language'], 
                    "pattern": v['samples']['Design Pattern'],
                    "llm_out": v['llm_out']
                }
            )

         
    if verbose:
        print("Getting Average Score and Highest Pattern Likelihood")
    score = float(0)
    if len(scores) > 0:
        score = np.mean(scores)
    eval = score>0.75
    top_pattern = "nothing"
    bot_pattern = "nothing"
    if len(patterns) > 0:
        occurence = Counter()
        for i in patterns:
            occurence.update(i)
        top_pattern = occurence.most_common(3)
        bot_pattern = occurence.most_common()[-3:]
    if len(resources) > 0:
        resource = max(resources, key=resources.count)
    else:
        resource = "No resource"

    if verbose:
        print({
        "design_pattern": eval, 
        "repo_url": repo_url, 
        "num_files": num_files, 
        "overall_score": str(score), 
        "top_3_patterns": top_pattern,
        "bot_3_patterns": bot_pattern, 
        "resource": resource, 
        "files": np.asarray(pp).tolist()
    })
    return {
        "design_pattern": eval, 
        "repo_url": repo_url, 
        "num_files": num_files, 
        "overall_score": str(score), 
        "top_3_patterns": top_pattern,
        "bot_3_patterns": bot_pattern, 
        "resource": resource, 
        "files": np.asarray(pp).tolist()
    }

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
    archetypes_ds = load_dataset('csv', data_files="./scores/Archetypes/data/embedded_llama_faiss_ds.csv", split="train")
    archetypes_ds.load_faiss_index('embedding', './scores/Archetypes/data/gpt_index.faiss')

    designpatterns_ds = load_dataset('csv', data_files="./scores/DesignPatternDetector/data/embedded_gpt_faiss_ds.csv", split="train")
    designpatterns_ds.load_faiss_index('embedding', './scores/DesignPatternDetector/data/gpt_index.faiss')

    professionalism_ds = load_dataset('csv', data_files='./scores/PerspecToolformer/data/embedded_dataset.csv', split='train')
    professionalism_ds.load_faiss_index('embedding', './scores/PerspecToolformer/data/professionalism_index.faiss')


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

    professionalism_template = PromptTemplate(input_variables=["examples", "message_history", "current_message"], template=number_template)

    professionalism_chain = RetrievalAugmentedQAPipeline(vector_db=professionalism_ds, llm=llm, verbose=False, embedder=EMBEDDER, template=professionalism_template)

    cl.user_session.set("llm", llm)
    cl.user_session.set("embedder", EMBEDDER)
    cl.user_session.set("design_ds", designpatterns_ds)
    cl.user_session.set("professionalism", professionalism_chain)

@cl.on_message
async def main(message: str):
    llm = cl.user_session.get("llm")
    embedder = cl.user_session.get("embedder")
    design_ds = cl.user_session.get("design_ds")
    professionalism = cl.user_session.get("professionalism")

    archetype = await cl.make_async(predict_archetype)(user_token="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", desired_repo="perspect-scores")
    designpattern = await cl.make_async(get_github)("https://github.com/rphovley/storymakerevents", branch='main', verbose=False, dataset=design_ds, llm=llm, embedder=embedder)
    prof_score = await cl.make_async(professionalism.run_pipeline)(message, "")

    res = {"archetype": archetype, "design_pattern": designpattern, "professionalism": prof_score}

    await cl.Message(content=res).send()