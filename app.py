from langchain import LlamaCpp, SagemakerEndpoint, PromptTemplate, LLMChain
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from gpt4all import Embed4All
from datasets import load_dataset
import numpy as np

from bs4 import BeautifulSoup
from collections import Counter

from urllib.parse import urljoin
import os
import json
import requests

class One_Class_To_Rule_Them_All():
    def __init__(self, 
                 and_in: LlamaCpp, 
                 the: str, 
                 darkness: str, 
                 bind: bool, 
                 them: Embed4All):
        self.llm = and_in
        self.ds_folder = the
        self.index_folder = darkness
        self.verbose = bind
        self.embedder = them

    def load_ds_and_idx(self, idx):
        _, _, files = os.walk(self.ds_folder)
        self.vectorDB = load_dataset('csv', data_files=files[idx], split='train')
        _, _, files = os.walk(self.index_folder)
        self.vectorDB.load_faiss_index('embedding', files[idx])

    def embed_documents(self, examples):
        embedding = self.embedder.embed(examples['text'])
        return {'embedding': embedding}

    def get_relevant_documents(self, query: str, knn: int):
        embedding = self.embedder.embed(query)
        q = np.array(embedding, dtype=np.float32)
        _, samples = self.vectorDB.get_nearest_examples("embedding", q, k=knn)
        return [samples]

    def archetype_score(self, user_token: str, repo_name: str):
        from github import Github, Auth

        template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
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

        if self.verbose:
            print("Authenticating...")
        auth = Auth.Token(user_token)
        g = Github(auth=auth)
        user_login = g.get_user().login

        data = []
        repo = ''
        try:
            repo = g.get_user().get_repo(repo_name)
        except:
            if self.verbose:
                print(f"{repo_name} not found")
            obj = json.dumps({"repo": "not found"}, indent=4)
            return obj
        
        if(repo):
            # repo name
            if self.verbose:
                print('Looking at data from repo ', repo.name)
            repo_name = {"repo_name": repo.name}
            # Date of last push
            # print('Pushed at:', repo.pushed_at)
            # pushed_at = repo.pushed_at
            has_branch = False
            if self.verbose:
                print(f'Retrieving data from {repo.name}')
            for branch in repo.get_branches():
                # goes through each branch
                if len(branch.name) > 0:
                    has_branch = True
                branch_name = branch.name
            commit_count = 0
            if has_branch == True:
                for commit in repo.get_commits():
                    if self.verbose:
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
        arch_prompt_template = PromptTemplate(input_variables=['data'], template=template)
        
        if self.verbose:
            print("Running chain")
        arch_chain = LLMChain(llm=self.llm, prompt=arch_prompt_template)
        archetype = arch_chain.run(gitData)

        obj = json.dumps({"archetype": archetype})
        return obj
    
    def designpatterns_score(self, repo_url: str, branch="main"):
        CODE_FILES = [
            '.cgi',
            '.cmd',
            '.pl',
            '.class',
            '.cpp',
            '.css',
            '.h',
            '.html',
            '.java',
            '.php',
            '.py',
            '.ipynb',
            '.sh',
            '.swift',
            ]

        url_tree = [repo_url]
        file_urls = []
        contents = []
        for i in url_tree:
            r = requests.get(i)
            if r.status_code == 200:
                if self.verbose:
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
                                if self.verbose:
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
                                if self.verbose:
                                    print(f"Error on {j}")
                    except KeyError:
                            if self.verbose:
                                print(f"Error on {i}")
            else:
                print(f"Problem with {repo_url}")
        for tree_url in url_tree:
            if self.verbose:
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
                                if self.verbose:
                                    print(f"File: {url.split('/')[-1]}")
                    except TypeError:
                        if self.verbose:
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
                                        if self.verbose:
                                            print(f"File: {url.split('/')[-1]}")
                        except TypeError:
                            if self.verbose:
                                print(f"Error on {j}")
                except KeyError:
                    if self.verbose:
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
                    if self.verbose:
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
        # code_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
        #         ###Instruction:
        #         You are an expert programming assistant who will tell the truth, even if the truth is that they don't know.
        #         Given some code, you must determine if it is close to a particular design pattern in that programming language, and if so, how they can refactor that code into code that follows a design pattern.
        #         If the code does not match a design pattern, you should give a recommendation as to which design pattern should be used in this instance.

        #         ###Input:
        #         {code}

        #         ###Response:
        #         The closest design pattern to this code is"""
        # code_prompt_template = PromptTemplate(input_variables=['code'], template=code_template)
        # code_chain = LLMChain(llm=self.llm, prompt=code_prompt_template)

        files_to_analyze = []

        for file, content in zip(file_urls, contents):
            if content == '':
                continue
            embed = self.embedder.embed(content)
            query = np.array(embed, dtype=np.float32)
            score, samples = self.vectorDB.get_nearest_examples('embedding', query, k=1)
            files[file] = {
                'score': score, 
                'samples': samples, 
                'content': content, 
                'llm_out': ''
                }
            if self.verbose:
                print(f"Score: {score} {file.split('/')[-1]}")
            if score <= 1.3 and score >= 1.1:
                files_to_analyze.append(f"{file}, {content}")
                #files[file]['llm_out'] = code_chain.run(content)
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

            
        if self.verbose:
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

        if self.verbose:
            print({
            "design_pattern": eval, 
            "repo_url": repo_url, 
            "num_files": num_files, 
            "overall_score": str(score), 
            "top_3_patterns": top_pattern,
            "bot_3_patterns": bot_pattern, 
            "resource": resource, 
            "files": np.asarray(pp).tolist(),
            "files_to_analyze_with_llm": files_to_analyze
        })
        return json.dumps({
            "design_pattern": eval, 
            "repo_url": repo_url, 
            "num_files": num_files, 
            "overall_score": str(score), 
            "top_3_patterns": top_pattern,
            "bot_3_patterns": bot_pattern, 
            "resource": resource, 
            "files": np.asarray(pp).tolist(),
            "files_to_analyze_with_llm": files_to_analyze
        })
    
    def professionalism_score(self, slack_token: str):
        import slack_sdk
        import pandas as pd

        number_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
        ###Instruction:
        You are an expert witness specializing in empathy, toxicity, and professionalism.
        Given a person's message history, some already-rated examples as context, and a current message, rate the messages on a scale of 1-100 for how professional they are.
        Please respond with only an integer between 1 and 100 where 1 is super toxic, 100 is super professional, and 50 is completely neutral.
        Then give a short explanation of how the person could be more professional.

        ###Input:

        #Examples: {examples}
        
        #Message History: {message_history}
        
        Current Message:
        {current_message}


        ###Response:
        Your Professionalism rating from 1-100 is """

        client=slack_sdk.WebClient(token=slack_token)
        dm_channels_response = client.conversations_list(types="im")
        
        all_messages = {}

        for channel in dm_channels_response["channels"]:
            # Get conversation history
            history_response = client.conversations_history(channel=channel["id"])

            # Store messages
            all_messages[channel["id"]] = history_response["messages"]

        txts = []

        for channel_id, messages in all_messages.items():
            for message in messages:
                try:
                    text = message["text"]
                    user = message["user"]
                    timestamp = message["ts"]
                    txts.append([timestamp,user,text])
                except:
                    pass

        df = pd.DataFrame(txts)
        df.columns =  ['timestamp','user','text']
        self_user = df['user'].value_counts().idxmax()
        df = df[df.user == self_user]

        messages = df['text'].values.tolist()

        embeddings_list = []
        for message in messages:
            embed = self.embed_documents({"text": message})
            embeddings_list.append(embed)
        df['embedding'] = embeddings_list

        message_history = []
        scores = []
        for message in messages:
            if self.verbose:
                print()
            print(f"Searching VectorDB for {message[:10]}...")
            db_query = self.embedder.embed(message)
            db_query = np.array(db_query, dtype=np.float32)
            _, context_list = self.vectorDB.get_nearest_examples("embedding", db_query, k=3)

            if self.verbose:
                print("Gathering Context...")
            score_string = ""
            for similar_message, rating, comment in zip(
                context_list['text'], 
                context_list['rating'], 
                context_list['comment']
                ):
                score_string += f"Example: {similar_message}, Rating: {rating}, Reasoning{comment}\n"
            if self.verbose:
                print(f"Similar Messages from DB: {score_string}")
            
            formatted_prompt_template = PromptTemplate(
                input_variables=['examples', 'message_history', 'current_message'],
                template=number_template
            )
            chain = LLMChain(llm=self.llm, prompt=formatted_prompt_template)
            if self.verbose:
                print("Running Chain...")

            if len(message_history) < 1:
                message_history = ["No message history"][0]
            elif len(message_history) == 1:
                message_history = message_history[0]
            else:
                message_history = ",\n".join(message_history)

            obj = chain.run({
                "examples": score_string,
                "message_history": message_history,
                "current_message": message,
            })
            message_history.append(message)
            scores.append(obj)

        df['scores'] = scores
        df.to_csv(f"./user_slack_data/{self_user}_messages.csv")

