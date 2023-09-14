from langchain import PromptTemplate, LLMChain, LlamaCpp

from datasets import load_dataset, Dataset

import chainlit as cl

import json

from gpt4all import Embed4All
import numpy as np

import os

import nest_asyncio
nest_asyncio.apply()

    

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

embedder = Embed4All()

def embed_texts(examples):
        embedding = embedder.embed(examples['text'])
        return {"embedding": embedding}

def get_relevant_documents(self, query: str, vectorizer: Embed4All, docs: Dataset):
        embedding = vectorizer.embed(query)
        q = np.array(embedding, dtype=np.float32)
        _, samples = docs.get_nearest_examples("embedding", q, k=self.k)
        return [samples]

class RetrievalAugmentedQAPipeline:
    def __init__(self, vector_db: Dataset, llm: LlamaCpp, verbose: bool, embedder: Embed4All, template=None,) -> None:
        self.vector_db = vector_db
        self.template = template
        self.llm = llm
        self.verbose = verbose
        self.embedder = embedder

    def run_pipeline(self, user_query: str, message_history: list) -> str:
        if self.verbose:
            print(f"Searching VectorDB for {user_query[:10]}...")
        db_query = self.embedder.embed(user_query)
        db_query = np.array(db_query, dtype=np.float32)
        _, context_list = self.vector_db.get_nearest_examples("embedding", db_query, k=5)

        if self.verbose:
            print("Gathering context...")
        score_string = ""
        for similar_message, rating, comment in zip(context_list['text'], context_list['rating'], context_list['comment']):
             score_string += f"Message: {similar_message}\nRating: {rating}\nReasoning: {comment}\n"
        print(score_string)

        formatted_prompt_template = PromptTemplate(input_variables=['examples', 'message_history', 'current_message'], template=self.template)
        chain = LLMChain(llm=self.llm, prompt=formatted_prompt_template)
        if self.verbose:
            print("Running Chain")

        return chain.run({"examples": score_string, "message_history": "\n".join(message_history), "current_message": user_query})

def predict_convo(input: str, message_history: list, review_chain=None):
    professionalism = review_chain.run_pipeline(input, message_history)
    obj = json.dumps({"professionalism": professionalism}, indent=4)
    return obj

@cl.author_rename
def rename(orig_author: str):
    rename_dict = {"RetrievalPerspectScores": "LLM advice"}
    return rename_dict.get(orig_author, orig_author)

@cl.on_chat_start
async def init():
    msg = cl.Message(content=f"Building Index...")
    await msg.send()

    #build FAISS
    dataset = load_dataset('csv', data_files="./data/embedded_dataset.csv", split='train')
    if not os.path.exists("./data/professionalism_index.faiss"):
        emb_ds = dataset.map(embed_texts, batched=False)
        emb_ds.to_csv("./data/embedded_dataset.csv")
        emb_ds.add_faiss_index("embedding")
        emb_ds.save_faiss_index('embedding', './data/professionalism_index.faiss')
        dataset = load_dataset('csv', data_files="./data/embedded_dataset.csv", split='train')
    dataset.load_faiss_index('embedding', './data/professionalism_index.faiss')

    #LLM
    llm = LlamaCpp(
            model_path="./models/llongma-7b-gguf-q4_0.bin",
            n_gpu_layers=1,
            n_batch=512,
            n_ctx=8192,
            verbose=True
            )

    #RAG class
    llm_chain = RetrievalAugmentedQAPipeline(
         dataset, 
         llm, 
         True, 
         embedder, 
         number_template
    )

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)

    # Get Slack message history
    message_history = ['Hey',"What's up"]

    cl.user_session.set("history", message_history)

@cl.on_message
async def main(message: str):
    # Retrieve the chain and history from the user session
    llm_chain = cl.user_session.get("llm_chain")

    history = cl.user_session.get("history")

    # Call the chain asyncronously
    res = await cl.make_async(predict_convo)(message, history, llm_chain)

    await cl.Message(content=res).send()