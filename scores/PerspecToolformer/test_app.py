from langchain import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain, LLMMathChain, TransformChain, SequentialChain, SimpleSequentialChain

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import  ConversationSummaryBufferMemory, ConversationBufferWindowMemory

import gradio as gr
import json


toxic_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
###Instruction:
You are an expert psychologist with a specialty in sociology.
Given someone's private message history, you can tell them how to improve their empathy and decrease their toxicity citing examples from the text.
You are constructively critical and will avoid being too nice, as that could increase toxicity. Professionalism dictates that nearly everyone can improve.
If you are unable to find anything, you will truthfully tell them they are doing great or that you don't know.

###Input:
Message History:
{messages}

###Response:
Thinking through this step-by-step, you can improve your toxicity by"""

toxic_prompt_template = PromptTemplate(input_variables=['messages'], template=toxic_template)

llm = LlamaCpp(
    model_path="./models/llongma-7b-gguf-q2_k.bin",
    n_gpu_layers=0,
    n_batch=512,
    n_ctx=8000,
    verbose=True
    )

toxicity_chain = LLMChain(llm=llm, prompt=toxic_prompt_template)

number_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
###Instruction:
You are an expert witness specializing in toxicity and professionalism.
Given a person's message history, rate the messages on a scale of 1-100 for how professional they are.
Please respond first with only an integer between 1 and 100, where 1 is always toxic and 100 is always professional.
Then add an example on how to improve.

###Input:
Message History:
{synopsis}


###Response:
Your professionalism rating from 1-100 is """

number_prompt_template = PromptTemplate(input_variables=["synopsis"], template=number_template)

review_chain = LLMChain(llm=llm, prompt=number_prompt_template)

# This is the overall chain where we run these two chains in sequence.

#overall_chain = SimpleSequentialChain(chains= [toxicity_chain, review_chain], verbose=True)





def predict_convo(input=""):
    professionalism = review_chain.run(input)
    obj = json.dumps({"professionalism": professionalism}, indent=4)
    return obj

with gr.Blocks() as app:
    messages = gr.Textbox(label="messages")

    gr.Examples(
        [
            ["Brent:You're the best!\nJake:We can do it.\nBrent:You've never been good enough.\nJake:I can take care of that whenever.\nBrent:sup bro\nJake:How do we know when we're done?\nBrent:Are you going to be in the office today?\nJake:How long does it take to get there?\nBrent:I'll be there in about 30 minutes\nJake:Why are you always constantly late?"],
        ],
        inputs=[messages]
    )

    start_btn = gr.Button("Detect Professionalism")
    start_btn.click(
        fn=predict_convo,
        inputs=[messages],
        outputs=gr.Textbox(label="output"),
        api_name="professionaldetect"
    )
    
app.queue(concurrency_count=1)
app.launch()#inbrowser=True, share=True,