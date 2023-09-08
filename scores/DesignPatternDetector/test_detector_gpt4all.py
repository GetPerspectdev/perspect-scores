from datasets import load_dataset
from langchain import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from gpt4all import Embed4All
import numpy as np

ds = load_dataset('csv', data_files="./data/Design_Patterns.csv", split='train')
#print(ds[0])

#USE WITH NVIDIA GPU AND BITSANDBYTES
# model_ckpt = "meta-llama/Llama-2-7b-hf"
# model_path = "./models/hermes-llongma-2-13b-8k.ggmlv3.q5_K_M.bin"
# tokenizer = AutoTokenizer.from_pretrained("./tokenizer/")
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.save_pretrained("./tokenizer/")
# model = AutoModel.from_pretrained(model_ckpt, torch_dtype=torch.bfloat16)

# def mean_pooling(model_output, attention_mask):
#     token_embeds = model_output[0]
#     input_mask_expanded = (
#         attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
#     )
#     sum_embeddings = torch.sum(token_embeds * input_mask_expanded, 1)
#     sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#     return sum_embeddings / sum_mask

# def embed_texts(examples):
#     print(f"tokenizing {examples['Language']} {examples['Design Pattern']}")
#     inputs = tokenizer(
#         examples['Code Example'],
#         padding=True,
#         truncation=True,
#         max_length=4096,
#         return_tensors="pt",
#     )
#     print("Sending to model")
#     with torch.no_grad():
#         model_output = model(**inputs)
#     print("pooling")
#     pooled_embeds = mean_pooling(model_output, inputs['attention_mask'])
#     return {"embedding": pooled_embeds.cpu().numpy()}
embedder = Embed4All()

def embed_texts(examples):
    print(f"Embedding {examples['Language']} {examples['Design Pattern']}")
    embedding = embedder.embed(examples['Code Example'])
    return {"embedding": embedding}

emb_ds = ds.map(embed_texts, batched=False)
emb_ds.to_csv("./data/Embedded.csv")

emb_ds.add_faiss_index("embedding")
emb_ds.save_faiss_index('embedding', 'my_index.faiss')


idx, knn = 1, 3

rn, nl = "\r\n\r\n", "\n"

query = np.array(emb_ds[idx]['embedding'], dtype=np.float32)
scores, samples = emb_ds.get_nearest_examples("embedding", query, k=knn)

print(f"QUERY LABEL: {emb_ds[idx]['Design Pattern']}")
print(
    f"QUERY TEXT: {emb_ds[idx]['Code Example'][:200].replace(rn, nl)} [...]\n"
)
print("=" * 50)
print("Retrieved Documents:")
for score, label, text in zip(
    scores, samples["Design Pattern"], samples["Code Example"]
):
    print("=" * 50)
    print(f"TEXT:\n{text[:200].replace(rn, nl)} [...]")
    print(f"SCORE: {score:.2f}")
    print(f"LABEL: {label}")

print("="*80)
print("MODEL SHOWCASE:")

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
llm = LlamaCpp(
    model_path="./models/hermes-llongma-2-7b-8k.ggmlv3.q2_K.bin",
    n_gpu_layers=0,
    n_batch=512,
    n_ctx=8192,
    verbose=True
    )
code_chain = LLMChain(llm=llm, prompt=code_prompt_template)

code = """def dummy_function(foo, bar):
            #Take foo, multiply it by bar, then return the output
            out = foo * bar
            return out"""
code_chain.run(code)


# with model.chat_session():
#     tokens = list(model.generate(prompt='You are a helpful and expert assistant who is extensively aware of all programming design patterns.', max_tokens=0, streaming=True))
#     model.current_chat_session.append({'role': 'assistant', 'content': ''.join(tokens)})

#     tokens = list(model.generate(prompt='''Detect which programming language the following code is written in and which design pattern it matches most:
#                                 from __future__ import annotations
#                                 from abc import ABC, abstractmethod


#                                 class AbstractFactory(ABC):
#                                     """"""
#                                     The Abstract Factory interface declares a set of methods that return
#                                     different abstract products. These products are called a family and are
#                                     related by a high-level theme or concept. Products of one family are usually
#                                     able to collaborate among themselves. A family of products may have several
#                                     variants, but the products of one variant are incompatible with products of
#                                     another.
#                                     """"""
#                                     @abstractmethod
#                                     def create_product_a(self) -> AbstractProductA:
#                                         pass

#                                     @abstractmethod
#                                     def create_product_b(self) -> AbstractProductB:
#                                         pass


#                                 class ConcreteFactory1(AbstractFactory):
#                                     """"""
#                                     Concrete Factories produce a family of products that belong to a single
#                                     variant. The factory guarantees that resulting products are compatible. Note
#                                     that signatures of the Concrete Factory's methods return an abstract
#                                     product, while inside the method a concrete product is instantiated.
#                                     """"""

#                                     def create_product_a(self) -> AbstractProductA:
#                                         return ConcreteProductA1()

#                                     def create_product_b(self) -> AbstractProductB:
#                                         return ConcreteProductB1()


#                                 class ConcreteFactory2(AbstractFactory):
#                                     """"""
#                                     Each Concrete Factory has a corresponding product variant.
#                                     """"""

#                                     def create_product_a(self) -> AbstractProductA:
#                                         return ConcreteProductA2()

#                                     def create_product_b(self) -> AbstractProductB:
#                                         return ConcreteProductB2()


#                                 class AbstractProductA(ABC):
#                                     """"""
#                                     Each distinct product of a product family should have a base interface. All
#                                     variants of the product must implement this interface.
#                                     """"""

#                                     @abstractmethod
#                                     def useful_function_a(self) -> str:
#                                         pass


#                                 """"""
#                                 Concrete Products are created by corresponding Concrete Factories.
#                                 """"""


#                                 class ConcreteProductA1(AbstractProductA):
#                                     def useful_function_a(self) -> str:
#                                         return ""The result of the product A1.""


#                                 class ConcreteProductA2(AbstractProductA):
#                                     def useful_function_a(self) -> str:
#                                         return ""The result of the product A2.""


#                                 class AbstractProductB(ABC):
#                                     """"""
#                                     Here's the the base interface of another product. All products can interact
#                                     with each other, but proper interaction is possible only between products of
#                                     the same concrete variant.
#                                     """"""
#                                     @abstractmethod
#                                     def useful_function_b(self) -> None:
#                                         """"""
#                                         Product B is able to do its own thing...
#                                         """"""
#                                         pass

#                                     @abstractmethod
#                                     def another_useful_function_b(self, collaborator: AbstractProductA) -> None:
#                                         """"""
#                                         ...but it also can collaborate with the ProductA.

#                                         The Abstract Factory makes sure that all products it creates are of the
#                                         same variant and thus, compatible.
#                                         """"""
#                                         pass


#                                 """"""
#                                 Concrete Products are created by corresponding Concrete Factories.
#                                 """"""


#                                 class ConcreteProductB1(AbstractProductB):
#                                     def useful_function_b(self) -> str:
#                                         return ""The result of the product B1.""

#                                     """"""
#                                     The variant, Product B1, is only able to work correctly with the variant,
#                                     Product A1. Nevertheless, it accepts any instance of AbstractProductA as an
#                                     argument.
#                                     """"""

#                                     def another_useful_function_b(self, collaborator: AbstractProductA) -> str:
#                                         result = collaborator.useful_function_a()
#                                         return f""The result of the B1 collaborating with the ({result})""


#                                 class ConcreteProductB2(AbstractProductB):
#                                     def useful_function_b(self) -> str:
#                                         return ""The result of the product B2.""

#                                     def another_useful_function_b(self, collaborator: AbstractProductA):
#                                         """"""
#                                         The variant, Product B2, is only able to work correctly with the
#                                         variant, Product A2. Nevertheless, it accepts any instance of
#                                         AbstractProductA as an argument.
#                                         """"""
#                                         result = collaborator.useful_function_a()
#                                         return f""The result of the B2 collaborating with the ({result})""


#                                 def client_code(factory: AbstractFactory) -> None:
#                                     """"""
#                                     The client code works with factories and products only through abstract
#                                     types: AbstractFactory and AbstractProduct. This lets you pass any factory
#                                     or product subclass to the client code without breaking it.
#                                     """"""
#                                     product_a = factory.create_product_a()
#                                     product_b = factory.create_product_b()

#                                     print(f""{product_b.useful_function_b()}"")
#                                     print(f""{product_b.another_useful_function_b(product_a)}"", end="""")


#                                 if __name__ == ""__main__"":
#                                     """"""
#                                     The client code can work with any concrete factory class.
#                                     """"""
#                                     print(""Client: Testing client code with the first factory type:"")
#                                     client_code(ConcreteFactory1())

#                                     print(""\n"")

#                                     print(""Client: Testing the same client code with the second factory type:"")
#                                     client_code(ConcreteFactory2())

#                                  Respond with only the name of the programming language and the design pattern''', max_tokens=20, streaming=True))
#     model.current_chat_session.append({'role': 'assistant', 'content': ''.join(tokens)})

#     for i in range(len(model.current_chat_session)):
#         if i % 2 != 0:
#             print(model.current_chat_session[i]['content'])