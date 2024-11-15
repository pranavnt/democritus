# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List
import fire

from democritus.model.generation import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
    device: str = "cpu",
):
    """
    Examples to run with the pre-trained models (no fine-tuning). Prompts are
    usually in the form of an incomplete text prefix that the model can then try to complete.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.
    `max_gen_len` is needed because pre-trained models usually do not stop completions naturally.

    Args:
        ckpt_dir: Directory containing model checkpoint files
        tokenizer_path: Path to the tokenizer file
        temperature: Temperature for sampling (default: 0.6)
        top_p: Top-p sampling threshold (default: 0.9)
        max_seq_len: Maximum sequence length (default: 128)
        max_gen_len: Maximum generation length (default: 64)
        max_batch_size: Maximum batch size (default: 4)
        device: Device to run on ("cpu", "cuda", "mps", or "auto") (default: "auto")
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        device=device,
    )

    prompts: List[str] = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,

        I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]

    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")

    # Example of chat completion
    dialogs = [
        [
            {"role": "user", "content": "What's your favorite programming language and why?"}
        ],
        [
            {"role": "user", "content": "Write a haiku about artificial intelligence."}
        ]
    ]

    chat_results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    print("\nChat Completion Examples:\n")
    for dialog, result in zip(dialogs, chat_results):
        print(f"User: {dialog[0]['content']}")
        print(f"Assistant: {result['generation']['content']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)