from app import One_Class_To_Rule_Them_All
from langchain import LlamaCpp, OpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from gpt4all import Embed4All
import tiktoken


if __name__ == "__main__":
    the_finger = CallbackManager([StreamingStdOutCallbackHandler])
    the_size = 8000

    the_ring = LlamaCpp(
        model_path="./scores/test_model/llongma-7b-gguf-q4_0.bin",
        temperature=0.75,
        max_tokens=100,
        top_p=1,
        callback_manager=the_finger,
        verbose=True,
        n_ctx=the_size,
        n_gpu_layers=100,
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

    # scorer.load_ds_and_idx(1)
    # print("Loaded embeddings")
    # token = "SLACK-TOKEN-HERE"
    # obj = scorer.professionalism_score(slack_token="", user_id="U045WPKCDJ7")
    # #print(obj)
    # print("professionalism secure")

    
    scorer.load_ds_and_idx(0)
    github_token = ""
    repo_name = "perspect-scores"

    repos = scorer._get_git(github_token)
    for repo in repos:
        obj = scorer.designpatterns_local_score(repo_url=repos[2])
        print(f"Design patterns secure for {repo}")

  
    

    # obj = scorer.archetype_score(user_token=github_token, repo_name=repo_name)
    # print(obj)
    # print("archetype secure")