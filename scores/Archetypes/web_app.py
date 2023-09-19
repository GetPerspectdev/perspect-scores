# Needed to install brew
# pip3 install PyGithub
# pip install urllip3==1.26.6
from typing import Dict

import os
import boto3
from github import Github
import pandas as pd
from langchain import PromptTemplate, SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain, LLMMathChain, TransformChain, SequentialChain, SimpleSequentialChain

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import  ConversationSummaryBufferMemory, ConversationBufferWindowMemory

import json

print("we got here")
ENDPOINT_NAME = os.environ.get('SM_ENDPOINT_NAME')
runtime= boto3.client('runtime.sagemaker', region_name='us-west-2',
                      aws_access_key_id='AKIAUP67XQF3XIOJBBP2',
                      aws_secret_access_key='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')

print("now here")
# Authentication is defined via github.Auth
from github import Auth

# Briton token to test
# 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

def predict_archetype(user_token, desired_repo):
    auth = Auth.Token(user_token)
    target_repo = desired_repo
    # Public Web Github
    g = Github(auth=auth)
    user_login = g.get_user().login

    data = []
    repo = ''
    try:
        repo = g.get_user().get_repo(target_repo)
    except:
        print("Repo not found")
        obj = json.dumps({"repo": "not found"}, indent=4)
        return obj
    
    if(repo):
        # repo name
        print('Looking at data from repo', repo.name)
        repo_name = {"repo_name": repo.name}
        # Date of last push
        # print('Pushed at:', repo.pushed_at)
        # pushed_at = repo.pushed_at
        has_branch = False
        print('Retrieving data from repo', repo.name)
        for branch in repo.get_branches():
            # goes through each branch
            if len(branch.name) > 0:
                has_branch = True
            branch_name = branch.name
        commit_count = 0
        if has_branch == True:
            for commit in repo.get_commits():
                print('Retrieving your commits...')
                author = str(commit.author)
                if (user_login in author) == True:
                    # number of commits by user
                    commit_count += 1
        pull_count = 0
        pull_file_count = 0
        for pull in repo.get_pulls():
            #number of pulls and num files changed in each pull
            pull_count =+ 1
            pull_count =+ pull.changed_files
        item = {"repo_name": repo_name,
                # "pushed_at": pushed_at,
                "branch_name": branch.name,
                "commit_count": commit_count,
                "pull_count": pull_count,
                "pull_file_count": pull_file_count}
        data.append(item)

    gitData = json.dumps(data)

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

    arch_prompt_template = PromptTemplate(input_variables=['data'], template=arch_template)

    # llm = LlamaCpp(
    #     model_path="./models/llongma-7b-gguf-q5_K_M.bin",
    #     n_gpu_layers=35,
    #     n_batch=512,
    #     n_ctx=8000,
    #     verbose=True
    #     )

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

    llm = SagemakerEndpoint(
        endpoint_name="jumpstart-dft-meta-textgeneration-llama-2-7b",
        region_name="us-west-2",
        model_kwargs={"parameters": {"max_new_tokens": 100}},
        content_handler=content_handler,
        endpoint_kwargs={"CustomAttributes":"accept_eula=true"}
    )

    arch_chain = LLMChain(llm=llm, prompt=arch_prompt_template)

    archetype = arch_chain.run(gitData)
    obj = json.dumps({"archetype": archetype}, indent=4)
    return obj