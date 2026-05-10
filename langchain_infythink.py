# All of the features of Infythink can be done in langchain without manually wiring everything
# LangChain gives:

# memory management
# chaining
# tool integration
# retrieval
# multi-agent orchestration

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp

MODEL_PATH = "/home/arindam/TinyLlama/tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf"

llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.7,
    max_tokens=256,
    n_ctx=2048
)

memory = ConversationBufferMemory()

template = """
Question:
{question}

Memory:
{history}

Reason carefully.
"""

prompt = PromptTemplate(
    input_variables=["question", "history"],
    template=template
)

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

question = "How would humanity colonize Europa?"

for i in range(5):

    result = chain.run(question=question)

    print(f"\nSTEP {i+1}\n")
    print(result)

    # Compress memory periodically
    if i % 2 == 1:

        summary = llm(
            f"""
Summarize this conversation briefly:

{memory.buffer}
"""
        )

        memory.clear()

        memory.save_context(
            {"input": "summary"},
            {"output": summary}
        )