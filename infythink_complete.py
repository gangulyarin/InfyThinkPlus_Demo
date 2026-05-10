# Final Full Code containing:
# faiss-cpu → vector DB for memory/retrieval.
# langchain → agent orchestration.
# llama-cpp-python → lightweight local inference.
# openai → optional external API for heavy reasoning or tool execution.

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import LlamaCppEmbeddings
import subprocess
import random
from llama_cpp import Llama

MODEL_PATH = "/home/arindam/TinyLlama/tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf"
llm = Llama(
    model_path=MODEL_PATH,  # download TinyLlama GGUF
    n_ctx=2048
)
def run_llm(prompt, temp=0.7):
    return llm(prompt, temperature=temp, max_tokens=200)["choices"][0]["text"]


# -----------------------------
# AGENT 1 — REASONER
# -----------------------------

def reason(question, memory):

    prompt = f"""
You are a careful reasoning agent.

Question:
{question}

Memory:
{memory}

Think step-by-step.
"""

    return run_llm(prompt)



# -----------------------------
# AGENT 2 — CRITIC
# -----------------------------

def critique(reasoning):
    prompt = f"""You are a strict critic.

Analyze the reasoning below.

Find:
- logical mistakes
- missing assumptions
- weak arguments
- contradictions

Reasoning:
{reasoning}

Critique:
"""

    return run_llm(prompt)



# Adding Vector Store:
# Embeddings Model
#embeddings = LlamaCppEmbeddings(model_path=MODEL_PATH)
# Use tiny SBERT embeddings (1-2 GB RAM)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector DB
#vector_memeory = FAISS(embedding_function=embeddings, index=None)
vector_memory = FAISS.from_texts([], embedding=embeddings)
# Optionally, remove dummy vector if you don't want it
vector_memory.delete(ids=[0])

# Planner
def planner(question):
    prompt = run_llm(prompt)
    return plan.split("\n")

def monte_carlo_explore(subproblem, iteration=3):
    paths = []
    for _ in range(iteration):
        reasoning = reason(subproblem, memory="")
        critique_text = critique(reasoning)
        score = reward(reasoning, critique_text)
        paths.append((score, reasoning))
    # Now pick best scoring reason
    paths.sort(reverse=True)
    return paths[0][1]

# Critic & Self-Reflection
def critique_with_memory(reasoning):
    # Retrieve relevant past memory
    relevant = vector_memeory.similarity_search(reasoning, k=3)
    context = "\n".join([r.page_content for r in relevant])
    prompt = f"""
Critique the reasoning below using past memory if relevant.
Reasoning: {reasoning}
Memory context: {context}
"""
    return run_llm(prompt)

# Refinement + Recursive Summarization
def refine(reasoning, critique_text):
    prompt = f"""
Given the reasoning:
{reasoning}

And the critique:
{critique_text}

Improve and rewrite the reasoning.
"""
    refined = run_llm(prompt)
    
    # Compress and store in vector DB
    vector_memory.add_texts([refined])
    
    return refined

# Tool Executor
def tool_executor(task):
    if "calculate" in task.lower() or "python" in task.lower():
        code_prompt = f"""
Generate Python code to solve: {task}
"""
        code = run_llm(code_prompt)
        try:
            # Simple exec sandbox
            local_vars = {}
            exec(code, {}, local_vars)
            return local_vars.get("result", "No result")
        except Exception as e:
            return str(e)
    return "No tools needed."

# Multi Agent Loop
def full_agent_loop(question, cycles=5):
    memory_summary = ""
    
    subproblems = planner(question)
    
    for cycle in range(cycles):
        print(f"\n=== CYCLE {cycle+1} ===")
        
        for sub in subproblems:
            reasoning = monte_carlo_explore(sub)
            critique_text = critique_with_memory(reasoning)
            refined = refine(reasoning, critique_text)
            tool_output = tool_executor(refined)
            
            print(f"\nSubproblem: {sub}")
            print(f"Refined Reasoning: {refined}")
            print(f"Tool Output: {tool_output}")
            
            # Update compressed memory summary
            memory_summary += "\n" + refined
    
    print("\n=== FINAL MEMORY SUMMARY ===")
    print(memory_summary)


if __name__ == "__main__":

    q = """
Design a fully autonomous research city
that can survive 20 years independently.
"""

    full_agent_loop(q)
