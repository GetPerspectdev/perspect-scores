import requests 
from typing import Dict
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import json
from collections import Counter
import math

from langchain import PromptTemplate
from langchain.llms import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.chains import LLMChain #, SimpleSequentialChain
#from langchain.embeddings import LlamaCppEmbeddings
from gpt4all import Embed4All
from datasets import load_dataset
import numpy as np

ds = load_dataset('csv', data_files="./data/embedded_gpt_faiss_ds.csv", split='train')
ds.load_faiss_index('embedding', './data/gpt_index.faiss')
knn = 1
# rn, nl = "\r\n\r\n", "\n"

embedder = Embed4All()
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
# llm = LlamaCpp(
#     model_path="./models/hermes-llongma-2-7b-8k.ggmlv3.q5_K_M.bin",
#     n_gpu_layers=0,
#     n_batch=512,
#     n_ctx=8192,
#     verbose=True
#     )
# code_chain = LLMChain(llm=llm, prompt=code_prompt_template)

CODE_FILES = ['.cgi','.cmd','.pl','.class','.cpp','.css','.h','.html','.java','.php','.py','.ipynb','.sh','.swift']

def get_files(paths, file_name):
    for i in paths:
        if i in file_name:
            return "a"
    return None

# def parse_tree_urls():
#     pass

# def parse_file_urls_from_tree():
#     pass

# def get_content_from_file_urls():
#     pass

# def embed_content():
#     pass

# def query_and_score():
#     pass

# def format_and_record_files():
#     pass

# def print_nice():
#     pass
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

code_chain = LLMChain(llm=llm, prompt=code_prompt_template)

def get_github(repo_url, branch="main", verbose=False, dataset=ds):
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
    for file, content in zip(file_urls, contents):
        if content == '':
             continue
        embed = embedder.embed(content)
        query = np.array(embed, dtype=np.float32)
        score, samples = dataset.get_nearest_examples('embedding', query, k=knn)
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
    resource_names = []
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
            resources.append(v['samples']['Unnamed: 4'])
            resource_names.append(v['samples']['Design Pattern'])
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
        resource = max(sorted(resources), key=resources.count)
        resource_name = max(sorted(resource_names), key=resource_names.count)
    else:
        resource = "No resource"
        resource_name = "No resource"

    if verbose:
        print({
        "design_pattern": eval, 
        "repo_url": repo_url, 
        "num_files": num_files, 
        "overall_score": str(score), 
        "top_3_patterns": top_pattern,
        "bot_3_patterns": bot_pattern, 
        "resource": resource, 
        "resource_name": resource_name, 
        "files": np.asarray(pp).tolist()
    })
    return {
        "design_pattern": bool(eval), 
        "repo_url": repo_url, 
        "num_files": num_files, 
        "overall_score": str(score), 
        "top_3_patterns": top_pattern,
        "bot_3_patterns": bot_pattern, 
        "resource": resource, 
        "files": np.asarray(pp).tolist()
    }