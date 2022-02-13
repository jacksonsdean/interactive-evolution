import openai

openai.api_key = "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"

def finish_poem(text):
    response = openai.Completion.create(
    engine="text-davinci-001",
    prompt=f"Finish the poem:\n{text}",
    temperature=0.5,
    max_tokens=80,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["#", ";", "\"\"\""]
    )
    top_choice = text
    top_choice += response.choices[0].text
    top_choice = top_choice.replace("\n\n", "\n")
    return top_choice

if __name__ == '__main__':
    # example:
    out = finish_poem("roses are red") 
    print(out)
