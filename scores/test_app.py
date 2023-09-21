from app import One_Class_To_Rule_Them_All
from langchain import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from gpt4all import Embed4All


if __name__ == "__main__":
    the_finger = CallbackManager([StreamingStdOutCallbackHandler])

    the_ring = LlamaCpp(
        model_path="./scores/test_model/llongma-7b-gguf-q4_0.bin",
        temperature=0.75,
        max_tokens=100,
        top_p=1,
        callback_manager=the_finger,
        verbose=True,
        n_ctx=8000,
        n_gpu_layers=1,
        n_batch=512,
        n_threads=1,
        seed=8855,
    )

    the_forge = Embed4All()

    scorer = One_Class_To_Rule_Them_All(
        and_in=the_ring,
        the="./scores/data",
        darkness="./scores/indexes",
        bind=True,
        them=the_forge
    )

    scorer.load_ds_and_idx(1)
    print("Loaded embeddings")

    token = "xoxp-3014184694567-4200801421619-5918204969079-dc7524383be15002b7fe9f12deb4d119"
    obj = scorer.professionalism_score(slack_token=token)
    print(obj)

    scorer.load_ds_and_idx(0)
    obj = scorer.designpatterns_score(repo_url="https://github.com/IMJONEZZ/NLP", branch="master")
    print(obj)

    github_token = ""
    repo_name = ""
    obj = scorer.archetype_score(user_token=github_token, repo_name=repo_name)
    print(obj)