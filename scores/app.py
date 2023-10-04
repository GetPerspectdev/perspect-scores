from langchain import LlamaCpp, SagemakerEndpoint, PromptTemplate, LLMChain
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np

from bs4 import BeautifulSoup
from collections import Counter

from urllib.parse import urljoin
import os
import subprocess
from shutil import rmtree
import json
import requests

class One_Class_To_Rule_Them_All():
    def __init__(self, 
                 and_in: LlamaCpp, 
                 the: str, 
                 darkness: str, 
                 bind: bool, 
                 them: SentenceTransformer):
        self.llm = and_in
        self.ds_folder = the
        self.index_folder = darkness
        self.verbose = bind
        self.embedder = them
        if self.verbose:
            print("Loaded init!")

    def __repr__(self) -> str:
        return f"One Class To Rule Them All and_in the darkness bind them\n{self.llm}\n{self.embedder}"

    def load_ds_and_idx(self, idx):
        files = os.listdir(self.ds_folder)
        files.sort()
        # try:
        self.vectorDB = load_dataset('csv', data_files=f"{self.ds_folder}/{files[idx]}", split='train')
        # except:
        #     print(f"Could not load {files[idx]}")
        files = os.listdir(self.index_folder)
        files.sort()
        # try:
        self.vectorDB.load_faiss_index('embedding', f"{self.index_folder}/{files[idx]}")
        # except:
        #     print(f"Could not load {files[idx]}")

    def embed_documents(self, examples):
        embedding = self.embedder.encode(examples['text'])
        return {'embedding': embedding}

    def get_relevant_documents(self, query: str, knn: int):
        embedding = self.embedder.encode(query)
        q = np.array(embedding, dtype=np.float32)
        _, samples = self.vectorDB.get_nearest_examples("embedding", q, k=knn)
        return [samples]
    
    def _get_git(self, user_token: str):
        from github import Github, Auth

        auth = Auth.Token(user_token)
        g = Github(auth=auth)

        repos = []
        for repo in g.get_user().get_repos():
            repos.append([f"https://{user_token}@github.com/{repo.full_name}.git", f"{'private' if repo.private else 'public'}", f"{repo.full_name}"])
            # print(dir(repo))
        return repos
    
    def _get_slack_diff(self, user_id: str = None, slack_token: str = None):
        import slack_sdk
        import pandas as pd

        client=slack_sdk.WebClient(token=slack_token)
        dm_channels_response = client.conversations_list(types="im")
        
        all_messages = {}

        if self.verbose:
            print("Getting Diff")

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
        new_df = pd.DataFrame(txts)
        new_df.columns =  ['timestamp','user','text']
        self_user = new_df['user'].value_counts().idxmax()
        new_df = new_df[new_df.user == self_user]

        if self.verbose:
            print("New Messages collected, comparing and concatenating...")

        try:
            files = os.listdir("./scores/user_slack_data/")
            file = [i for i in files if user_id in i]
            old_df = pd.read_csv(f"./scores/user_slack_data/{file[0]}")
            old_df = old_df['text'].values.tolist()
            
            df = pd.concat([new_df, old_df], ignore_index=True).drop_duplicates('timestamp')
            messages = df['text'].values.tolist()
        except Exception as e:
            print(e)
            df = new_df
            messages = df['text'].values.tolist()

        if self.verbose:
            print("Done getting messages!")

        return df, messages

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
            repo = g.get_repo(repo_name)
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
    
    def _designpatterns_local_score_archive(self, user_token: str):
        from github import Github, Auth
        import base64

        if self.verbose:
            print("Authenticating...")
        auth = Auth.Token(user_token)
        g = Github(auth=auth)
        user_login = g.get_user().login

        
        repos = {}
        for repo in g.get_user().get_repos():
            if self.verbose:
                print(f"Analyzing {repo.full_name}")

            skips = (
                ".md",
                ".yml",
                ".txt",
                ".toml",
                ".in",
                ".rst",
                ".jpg",
                ""
            )

            files = {}
            pp = []
            scores = []
            patterns = []
            resources = []
            avgs = {}
            repo = g.get_repo("PyGithub/PyGithub")
            contents = repo.get_contents("")
            while contents:
                file_content = contents.pop(0)
                if file_content.type == "dir":
                    contents.extend(repo.get_contents(file_content.path))
                elif file_content.name.startswith(".") or file_content.name.endswith(skips):
                    continue
                else:
                    if self.verbose:
                        print(f"File: {file_content.name}")
                    readable = base64.b64decode(file_content.content)
                    embed = self.embedder.encode(str(readable))
                    query = np.array(embed, dtype=np.float32)
                    score, samples = self.vectorDB.get_nearest_examples('embedding', query, k=1)
                    files[file_content.name] = {
                        'score': score, 
                        'samples': samples, 
                        'content': readable, 
                    }
            for k, v in files.items():
                scores.append(v['score'])
                patterns.append(v['samples']['Design Pattern'])
                if v['samples']['Design Pattern'][0] not in avgs:
                    avgs[v['samples']['Design Pattern'][0]] = [v['score']]
                else:
                    avgs[v['samples']['Design Pattern'][0]] += [v['score']]
                # try: 
                #     pp.append(f"Name: {k.split('/')[-1]} | Score: {v['score']} | Closest: {v['samples']['Language']} {v['samples']['Design Pattern']} | Model: {v['model_out']} |")
                # except KeyError:
                if v['score'] > 1.0:
                    pp.append(
                        {
                            "name": k.split('/')[-1], 
                            "score": float(v['score']), 
                            "language": v['samples']['Language'][0], 
                            "pattern": v['samples']['Design Pattern'][0],
                            "resource": v['samples']['Unnamed: 4'][0],
                        }
                    )
                else:
                    pp.append(
                        {
                            "name": k.split('/')[-1], 
                            "score": float(v['score']), 
                            "language": v['samples']['Language'][0], 
                            "pattern": v['samples']['Design Pattern'][0],
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
            for key in avgs.keys():
                avgs[key] = float(sum(avgs[key])/len(avgs[key]))

            if self.verbose:
                print({
                "design_pattern": bool(eval), 
                "repo_url": repo.full_name, 
                "overall_score": str(score), 
                "top_3_patterns": top_pattern,
                "bot_3_patterns": bot_pattern, 
                "resource": resource, 
                "files": np.asarray(pp).tolist(),
                "occurance": dict(occurence),
                "averages": avgs,
            })
            repos[repo.full_name] = ({
                "design_pattern": bool(eval), 
                "repo_url": repo.full_name, 
                "overall_score": str(score), 
                "top_3_patterns": top_pattern,
                "bot_3_patterns": bot_pattern, 
                "resource": resource, 
                "files": np.asarray(pp).tolist(),
                "occurance": dict(occurence),
                "averages": avgs,
            })
        return repos

    def designpatterns_local_score(self, repo_url: str = ""):
        repo_path = "/tmp/curr_repo"
        rmtree(repo_path, ignore_errors=True)
        result = subprocess.run(["git", "clone", "--depth=1", repo_url, repo_path])
        repo_files = [os.path.join(repo_path, f) for f in os.listdir(repo_path)]
        contents = [f for f in repo_files if os.path.isfile(f)]
        files = {}
        for file in contents:
            try:
                with open(file, 'r') as f:
                    content = f.read()
                    embed = self.embedder.encode(content)
                    query = np.array(embed, dtype=np.float32)
                    score, samples = self.vectorDB.get_nearest_examples('embedding', query, k=1)
                    files[file] = {
                        'score': score, 
                        'samples': samples, 
                        'content': content, 
                        }
            except UnicodeDecodeError:
                pass # binary or ascii file

        pp = []
        scores = []
        patterns = []
        resources = []
        avgs = {}
        for k, v in files.items():
            scores.append(v['score'])
            patterns.append(v['samples']['Design Pattern'])

            if v['samples']['Design Pattern'][0] not in avgs:
                avgs[v['samples']['Design Pattern'][0]] = [v['score']]
            else:
                avgs[v['samples']['Design Pattern'][0]] += [v['score']]

            if v['score'] > 1.0:
                pp.append(
                    {
                        "name": k.split('/')[-1], 
                        "score": float(v['score']), 
                        "language": v['samples']['Language'][0], 
                        "pattern": v['samples']['Design Pattern'][0],
                        "resource": v['samples']['Unnamed: 4'][0],
                    }
                )
            else:
                pp.append(
                    {
                        "name": k.split('/')[-1], 
                        "score": float(v['score']), 
                        "language": v['samples']['Language'][0], 
                        "pattern": v['samples']['Design Pattern'][0],
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
        for key in avgs.keys():
            avgs[key] = float(sum(avgs[key])/len(avgs[key]))
        
        rmtree("/tmp/curr_repo", ignore_errors=True)

        if self.verbose:
            print({
            "design_pattern": bool(eval), 
            "repo_url": repo_url, 
            "overall_score": str(score), 
            "top_3_patterns": top_pattern,
            "bot_3_patterns": bot_pattern, 
            "resource": resource, 
            "files": np.asarray(pp).tolist(),
            "occurance": dict(occurence),
            "averages": avgs,
        })
        return {
            "design_pattern": bool(eval), 
            "repo_url": repo_url, 
            "overall_score": str(score), 
            "top_3_patterns": top_pattern,
            "bot_3_patterns": bot_pattern, 
            "resource": resource, 
            "files": np.asarray(pp).tolist(),
            "occurance": dict(occurence),
            "averages": avgs,
        }
    
    def professionalism_score(self, slack_token: str, user_id=None):
        if user_id == None:
            import slack_sdk
        import pandas as pd
        import tiktoken
        tiktoker = tiktoken.encoding_for_model('gpt-3.5-turbo')

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

        if user_id == None:
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
            df.to_csv(f"./scores/user_slack_data/{self_user}_messages.csv")

            messages = df['text'].values.tolist()
        elif user_id and slack_token:
            df, messages = self._get_slack_diff(user_id=user_id, slack_token=slack_token)
        else:
            files = os.listdir("./scores/user_slack_data/")
            file = [i for i in files if user_id in i]
            df = pd.read_csv(f"./scores/user_slack_data/{file[0]}")
            messages = df['text'].values.tolist()

        embeddings_list = []
        for message in messages:
            message = str(message)
            message = message.encode("ascii", 'ignore').decode('utf-8').strip()
            if len(message)>0:
                embed = self.embedder.encode(message)
                embeddings_list.append(embed)
            else:
                embed = self.embedder.encode("Likely an emoji")
                embeddings_list.append(embed)
        df['embedding'] = embeddings_list

        message_history = []
        scores = []
        i = 1
        for message in messages:
            message = str(message)
            message = message.encode("ascii", 'ignore').decode('utf-8').strip()
            if self.verbose:
                print(f"Searching VectorDB for {message[:10]}...")
            if len(message)>0:
                db_query = self.embedder.encode(message)
            else:
                db_query = self.embedder.encode("Emoji")
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
                score_string += f"Example: {similar_message}, Rating: {rating}, Reasoning: {comment}\n"
            # if self.verbose:
            #     print(f"Similar Messages from DB: {score_string}")
            
            formatted_prompt_template = PromptTemplate(
                input_variables=['examples', 'message_history', 'current_message'],
                template=number_template
            )
            chain = LLMChain(llm=self.llm, prompt=formatted_prompt_template)
            if self.verbose:
                print("Running Chain...")

            dumb_message = "No Message History"
            if len(message_history) == 1:
                dumb_message = message_history[0]
            
            num_tokens = len(tiktoker.encode(f"{score_string},\n" + ",\n".join(message_history) + message))
            while num_tokens > 3800:
                message_history.pop(0)
                num_tokens = len(tiktoker.encode(f"{score_string},\n" + ",\n".join(message_history) + message))
            
            if self.verbose:
                print(f"Message token count: {num_tokens}")
                print(f"examples_from_db: {score_string}\ncurr_message: {message}")
            obj = chain.run({
                "examples": score_string,
                "message_history": ",\n".join(message_history) if len(message_history) > 1 else dumb_message,
                "current_message": message,
            })
            message_history.append(message)
            scores.append(obj)
            if self.verbose:
                print(f"Finished Message {i} of {len(messages)}\n{i//len(messages)}% complete")
            i += 1

        df['scores'] = scores
        df.to_csv(f"./scores/user_slack_data/{user_id if user_id else self_user}_messages.csv")
        return df.to_json()
    
    def process_scores(self, user_id: str = ""):
        pass