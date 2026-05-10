# Here we are using ollama cpp
# We will also use Reinfocement learning to improve as mentioned in the paper

import subprocess
import random
from llama_cpp import Llama

MODEL_PATH = "/home/arindam/TinyLlama/tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf"
llm = Llama(
    model_path=MODEL_PATH,  # download TinyLlama GGUF
    n_ctx=2048
)
def run_llm(prompt, temp=0.7):
    """cmd = [
        "./llama-cli",
        "-m", MODEL_PATH,
        "-p", prompt,
        "--temp", str(temp),
        "-n", "256"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout
    """
    return llm(prompt, max_tokens=200)["choices"][0]["text"]

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

# -----------------------------
# AGENT 3 — REWARD MODEL
# -----------------------------
def reward(reasoning, critique):
    score=100
    penalties = [
        "mistake",
        "contradiction",
        "incorrect",
        "missing",
        "weak"
    ]
    for p in penalties:
        score -= critique.lower().count(p) * 10
    
    score += random.randint(-5,5)
    return max(score,0)

# -----------------------------
# AGENT 4 — REFINER
# -----------------------------

def refine(question, reasoning, critique):

    prompt = f"""
Question:
{question}

Previous reasoning:
{reasoning}

Critique:
{critique}

Rewrite improved reasoning.
"""

    return run_llm(prompt)

# -----------------------------
# MEMORY COMPRESSION
# -----------------------------

def summarize(memory, new_reasoning):

    prompt = f"""
Compress the reasoning memory.

Keep:
- key facts
- conclusions
- unresolved issues

Old memory:
{memory}

New reasoning:
{new_reasoning}

Compressed memory:
"""

    return run_llm(prompt)


# -----------------------------
# MAIN LOOP
# -----------------------------

def infinite_reasoning(question, cycles=5):
    memory = ""
    for step in range(cycles):
        print(f"\n========================")
        print(f"CYCLE {step+1}")
        print(f"========================\n")
        reasoning = reason(question, memory)
        print("REASONING:\n")
        print(reasoning)
        critique_text = critique(reasoning)
        print("\nCRITIQUE:\n")
        print(critique_text)
        score = reward(reasoning, critique_text)
        print(f"\nREWARD SCORE: {score}")
        improved = refine(question,reasoning,critique_text)
        print("\nREFINED REASONING:\n")
        print(improved)
        memory = summarize(memory,improved)
        print("\nCOMPRESSED MEMORY:\n")
        print(memory)
    print("\n========================")
    print("FINAL MEMORY STATE")
    print("========================\n")

    print(memory)

if __name__ == "__main__":

    q = """
Design a fully autonomous underwater research city
that can survive 20 years independently.
"""

    infinite_reasoning(q)