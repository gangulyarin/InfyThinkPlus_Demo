# A very basic and lightweight demo to InfyThink+
# Using Ollama

from ollama import chat

MODEL = "qwen2.5-coder:3b-instruct-q4_K_M"

MAX_HISTORY = 3

def generate(prompt):
    response = chat(model=MODEL, messages=[
        {
            "role": "user",
            "content": prompt
        }
    ])
    return response["message"]["content"]

def summarize(history):
    prompt = f"""
You are a compression system.

Summarize the important reasoning information below.
Preserve facts, conclusions, plans, equations, and unresolved questions.

Reasoning:
{history}

Compressed Summary:
"""
    return generate(prompt)


def think(question, cycles=5):
    memory = ""
    reasoning_history = []
    current_prompt = f"""
Question:
{question}

Think step-by-step.
"""
    for i in range(cycles):
        print(f"\n=== Reasoning Cycle {i+1} ===\n")
        prompt = f"""
Previous memory:
{memory}

Current reasoning task:
{current_prompt}

Continue reasoning carefully.
"""
        output = generate(prompt)
        print(output)
        reasoning_history.append(output)

        #compress periodically
        if len(reasoning_history)>=MAX_HISTORY:
            combined = "\n\n".join(reasoning_history)
            memory = summarize(f"""
Existing memory:
{memory}

New reasoning:
{combined}
""")
            print("\n--- COMPRESSED MEMORY ---\n")
            print(memory)
            reasoning_history = []
        current_prompt = """
Continue solving the original question.
Focus on unresolved parts.
"""
    final_context = f"""
Question:
{question}

Compressed memory:
{memory}

Recent reasoning:
{reasoning_history}

Provide final answer.
"""
    final_answer = generate(final_context)
    print("\n=== FINAL ANSWER ===\n")
    print(final_answer)


if __name__ == "__main__":

    question = """
Design a plan for a Mars colony that can survive independently
for 15 years.
"""

    think(question)