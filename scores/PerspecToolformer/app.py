from typing import Dict
from langchain import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.llms import SagemakerEndpoint
from datasets import load_dataset

import json

from gpt4all import Embed4All
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable
import asyncio

import os
from typing import List

import nest_asyncio
nest_asyncio.apply()
import slack_sdk
import pandas as pd

# # Utils
# class TextFileLoader:
#     def __init__(self, path: str, encoding: str = "utf-8"):
#         self.documents = []
#         self.path = path
#         self.encoding = encoding

#     def load(self):
#         if os.path.isdir(self.path):
#             self.load_directory()
#         elif os.path.isfile(self.path) and self.path.endswith(".txt"):
#             self.load_file()
#         else:
#             raise ValueError(
#                 "Provided path is neither a valid directory nor a .txt file."
#             )

#     def load_file(self):
#         with open(self.path, "r", encoding=self.encoding) as f:
#             self.documents.append(f.read())

#     def load_directory(self):
#         for root, _, files in os.walk(self.path):
#             for file in files:
#                 if file.endswith(".txt"):
#                     with open(
#                         os.path.join(root, file), "r", encoding=self.encoding
#                     ) as f:
#                         self.documents.append(f.read())

#     def load_documents(self):
#         self.load()
#         return self.documents


# class CharacterTextSplitter:
#     def __init__(
#         self,
#         chunk_size: int = 1000,
#         chunk_overlap: int = 200,
#     ):
#         assert (
#             chunk_size > chunk_overlap
#         ), "Chunk size must be greater than chunk overlap"

#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap

#     def split(self, text: str) -> List[str]:
#         # chunks = []
#         # for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
#         #     chunks.append(text[i : i + self.chunk_size])
#         return text.split("\n")

#     def split_texts(self, texts: List[str]) -> List[str]:
#         chunks = []
#         for text in texts:
#             chunks.extend(self.split(text))
#         return chunks


# def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
#     """Computes the cosine similarity between two vectors."""
#     dot_product = np.dot(vector_a, vector_b)
#     norm_a = np.linalg.norm(vector_a)
#     norm_b = np.linalg.norm(vector_b)
#     return dot_product / (norm_a * norm_b)


# class VectorDatabase:
#     def __init__(self, embedding_model: Embed4All = None):
#         self.vectors = defaultdict(np.array)
#         self.embedding_model = embedding_model or Embed4All()
#         self.verbose = True

#     def insert(self, key: str, vector: np.array) -> None:
#         self.vectors[key] = vector

#     def search(
#         self,
#         query_vector: np.array,
#         k: int,
#         distance_measure: Callable = cosine_similarity,
#     ) -> List[Tuple[str, float]]:
#         if self.verbose:
#             print("Searching VectorDB")
#         scores = [
#             (key, distance_measure(query_vector, vector))
#             for key, vector in self.vectors.items()
#         ]
#         return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

#     def search_by_text(
#         self,
#         query_text: str,
#         k: int,
#         distance_measure: Callable = cosine_similarity,
#         return_as_text: bool = False,
#     ) -> List[Tuple[str, float]]:
#         if self.verbose:
#             print(f"Embedding {query_text[:10]}...")
#         query_vector = self.embedding_model.embed(query_text)
#         results = self.search(query_vector, k, distance_measure)
#         ratings = [(" ".join(rating[0].split(",")[-2:]), rating[1]) for rating in results]
#         return [rating[0] for rating in ratings] if return_as_text else ratings

#     def retrieve_from_key(self, key: str) -> np.array:
#         return self.vectors.get(key, None)

#     async def abuild_from_list(self, list_of_text: List[str]) -> "VectorDatabase":
#         embeddings = []
#         for text in list_of_text:
#             embedding = self.embedding_model.embed(text)
#             embeddings.append(embedding)
#         for text, embedding in zip(list_of_text, embeddings):
#             self.insert(text, np.array(embedding))
#         return self

# class RetrievalAugmentedQAPipeline:
#     def __init__(self, llm: LlamaCpp, vector_db_retriever: VectorDatabase, template=None, verbose=False) -> None:
#         self.llm = llm
#         self.template = template
#         self.vector_db_retriever = vector_db_retriever
#         self.verbose = verbose

#     def run_pipeline(self, user_query: str) -> str:
#         if self.verbose:
#             print(f"Searching VectorDB for {user_query[:10]}...")
#         context_list = self.vector_db_retriever.search_by_text(user_query, k=2)
        
#         if self.verbose:
#             print("Gathering context...")
#         context_prompt = ""
#         for context in context_list:
#             context_prompt += context[0] + "\n"

#         formatted_prompt_template = PromptTemplate(input_variables=['message_history', 'context'], template=self.template)
#         chain = LLMChain(llm=self.llm, prompt=formatted_prompt_template)
#         if self.verbose:
#             print("Running Chain")

#         professionalism = chain.run({"message_history":user_query, "context": context_prompt})
#         obj = json.dumps({"professionalism": professionalism}, indent=4)
#         return obj


# if __name__ == "__main__":
#     message_history = ["Bring in from slack here", "any number of strings in an array"]

#     text_loader = TextFileLoader("./data/professionalism.txt")
#     documents = text_loader.load_documents()

#     text_splitter = CharacterTextSplitter()
#     split_documents = text_splitter.split_texts(documents)

#     vector_db = VectorDatabase()
#     vector_db = asyncio.run(vector_db.abuild_from_list(split_documents))

#     number_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
#     ###Instruction:
#     You are an expert witness specializing in empathy, toxicity, and professionalism.
#     Given a summary from another expert, rate the messages on a scale of 1-100 for how professional they are (higher scores indicate more professional messages).
#     Please respond with only an integer between 1 and 100 and end after that integer.

#     ###Input:
#     Message History: {message_history}

#     Context:
#     {context}


#     ###Response:
#     Your Professionalism rating from 1-100 is """

#     llm = LlamaCpp(
#                 model_path="./models/llongma-7b-gguf-q2_K.bin",
#                 n_gpu_layers=1,
#                 n_batch=512,
#                 n_ctx=8000,
#                 verbose=True
#                 )

#     raqa_pipeline = RetrievalAugmentedQAPipeline(
#         llm=llm,
#         vector_db_retriever=vector_db,
#         template=number_template,
#         verbose=True
#     )

#     object = raqa_pipeline.run_pipeline(",\n".join(message_history))
#     print(object)

def professionalism_score(slack_token: str, verbose: bool = False):
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

    embedder = Embed4All()

    vectorDB = load_dataset('csv', data_files='./data/professionalism.csv', split='train')

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
    # df.to_csv(f"./scores/user_slack_data/{self_user}_messages.csv")

    messages = df['text'].values.tolist()
    

    embeddings_list = []
    for message in messages:
        if len(message)>0:
            embed = embedder.embed(message)
            embeddings_list.append(embed)
        else:
            embed = embedder.embed("Likely an emoji")
            embeddings_list.append(embed)
    df['embedding'] = embeddings_list

    message_history = []
    scores = []
    i = 1
    for message in messages:
        if verbose:
            print(f"Searching VectorDB for {message[:10]}...")
        db_query = embedder.embed(message)
        db_query = np.array(db_query, dtype=np.float32)
        _, context_list = vectorDB.get_nearest_examples("embedding", db_query, k=3)

        if verbose:
            print("Gathering Context...")
        score_string = ""
        for similar_message, rating, comment in zip(
            context_list['text'], 
            context_list['rating'], 
            context_list['comment']
            ):
            score_string += f"Example: {similar_message}, Rating: {rating}, Reasoning: {comment}\n"
        if verbose:
            print(f"Similar Messages from DB: {score_string}")
        
        formatted_prompt_template = PromptTemplate(
            input_variables=['examples', 'message_history', 'current_message'],
            template=number_template
        )
        chain = LLMChain(llm=llm, prompt=formatted_prompt_template)
        if verbose:
            print("Running Chain...")

        dumb_message = "No Message History"
        if len(message_history) == 1:
            dumb_message = message_history[0]

        obj = chain.run({
            "examples": score_string,
            "message_history": ",\n".join(message_history) if len(message_history) > 1 else dumb_message,
            "current_message": message,
        })
        message_history.append(message)
        scores.append(obj)
        if verbose:
            print(f"Finished Message {i} of {len(messages)}")
        i += 1

    df['scores'] = scores
    # df.to_csv(f"./scores/user_slack_data/{self_user}_messages.csv")
    return df.to_json()