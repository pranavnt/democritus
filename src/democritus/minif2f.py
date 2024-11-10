import json
from dataclasses import dataclass

@dataclass
class MiniF2FData:
    name: str
    split: str
    informal_prefix: str
    formal_statement: str
    goal: str
    header: str

def construct_prompt(data: MiniF2FData):
    return f"Prove the following formal statement in Lean, Give me the full lean file needed, including imports and all Do not use the `begin` keyword; use by instead as this is Lean4. Do not include any other text than the lean code, and no new additional imports are necessary.{data.header}\n{data.formal_statement}"

def load_minif2f(path: str):
    with open(path, "r") as f:
        return [MiniF2FData(**json.loads(line)) for line in f]

if __name__ == "__main__":
    data = load_minif2f("minif2f.jsonl")
    for d in data:
        print(construct_prompt(d))
        print()
        break
