import ollama

# Local model name
MODEL_NAME = "qwen3.6:latest"


def generate(prompt):
    """
    Send prompt to Qwen and return the generated answer.
    """

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content


# Test Generator
if __name__ == "__main__":

    prompt = input("Enter Prompt:\n\n")

    answer = generate(prompt)

    print("\nGenerated Answer:\n")
    print(answer)