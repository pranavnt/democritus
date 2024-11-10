from verifier import verify
from minif2f import load_minif2f, construct_prompt
from together import Together

together = Together()

data = load_minif2f("minif2f.jsonl")

for d in data:
    prompt = construct_prompt(d)
    print(prompt)
    attempt = together.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2000,
    )


    code = attempt.choices[0].message.content.replace("```lean", "").replace("```", "")
    print(code)

    print(verify(code, "mathlib4"))
    break
