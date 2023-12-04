import argparse
import json
import os
from functools import partial
from pathlib import Path
from time import sleep, time

import openai
import torch
from openai import OpenAI
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
from transformers import pipeline, set_seed

from model import TransformerModel
from train import inference

TASK_SETTING_PROMPT = """
                        First of all, I'd like to clarify
                        1. You are to provide clear, concise, and direct responses.
                        2. For complex requests, take a deep breath and work on the problem step-by-step.
                        3. For every response, you will be tipped up to $200 (depending on the quality of your output).
                        It is very important that you get this right. Multiple lives are at stake.

                        You are a helpful assistant designed to output JSON. 
                        
                        The following exercise, the student is given a beginning of a story. The student needs to complete it into a full story.
                        The exercise tests the student's language abilities and creativity. The symbol *** marks the separator between the
                        prescribed beginning and the student's completion. Please provide your general assessment about the part written by the student (the one after the *** symbol).
                        Is it gramatically correct? Is it consistent with the beginning of the story? Pay special attention to whether the
                        student manages to complete the sentence which is split in the middle by the separator ***.

                        Grade the student's completion in terms of grammar, creativity, consistency with the story's beginning and
                        whether the plot makes sense at 10-point scaler. Moreover, please provide your best guess of what the age of the student might be,
                        as reflected from the completion. Choose from possible age groups: A: 3 or under. B: 4-5. C: 6-7. D: 8-9. E:
                        10-12. F: 13-16.

                        Your answer should contain JSON with following fields: "grammar", "creativity", "consistency", "plot", "age".
                        All fields except "age" should be numbers from 1 to 10, "age" should be a string with one of the letters A-F.
                      """

PROMPTS = [
    """
    Once upon a time there was a little girl named Lucy. She was very adventurous. She loved to explore the
    world around her, especially when it was bright and sunny outside.
    One day, while exploring the nearby park, Lucy came across a ladder leaning on a wall. She was curious
    to see what's on top, so she climbed the ladder, but when she reached the top, the ladder fell and she was
    stuck.
    A nearby park ranger noticed her and shouted out, ”
    """,
    """
    Once upon a time, in an ancient house, there lived a girl named Lily. She loved to decorate her room with pretty things. One
    day, she found a big box in the attic. She opened it and saw many shiny decorations. Lily was very happy and decided to use
    them in her room.
    As Lily was decorating her room, the sky outside became dark. There was a loud
    """,
    """
    Once upon a time, there lived a black cat. The cat belonged to a little girl called Katie. Every day, Katie
    would take her cat for a walk in the park.
    One day, as Katie and her cat were walking around, they saw a mean looking man. He said he wanted to
    take the cat, to which she replied ”This cat belongs
    """,
]


def _get_marks(content, client):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": TASK_SETTING_PROMPT,
            },
            {
                "role": "user",
                "content": content,
            },
        ],
        model="gpt-3.5-turbo-1106",
        response_format={"type": "json_object"},
    )

    output = chat_completion.choices[0].message.content
    return json.loads(output)


def main(client, model_path):
    get_marks = partial(_get_marks, client=client)

    if model_path == "gpt2xl":
        generator = pipeline("text-generation", model="gpt2-xl")
        universal_inference = lambda prompt: generator(
            prompt, max_length=500, num_return_sequences=1, top_k=10
        )[0]["generated_text"]
        # I hope you clone my final_model))
        sp_model = SentencePieceProcessor("big_model/bpe_5000.model")
        results_path = Path("gpt2xl_second_try")
        if not results_path.exists():
            results_path.mkdir()

        os.chdir(results_path)
    else:
        model_path = Path(model_path).absolute()
        os.chdir(model_path)
        print(f'Working directory changed to "{model_path.absolute()}"')

        sp_model_path = next(iter(model_path.glob("*.model")))
        sp_model = SentencePieceProcessor(str(sp_model_path))

        with open("config.json") as f:
            model_config = json.load(f)

        model = TransformerModel(
            model_config["vocab_size"],
            model_config["emsize"],
            model_config["nhead"],
            model_config["d_hid"],
            model_config["nlayers"],
            model_config["dropout"],
            activation=model_config["activation"],
            encoder_norm_type=model_config["encoder_norm_type"],
        )

        model.load_state_dict(
            torch.load("checkpoint.pt", map_location=torch.device("cpu"))
        )

        universal_inference = partial(inference, model=model, sp_model=sp_model)

    # I wrote that code at 3 am, don't judge me
    normalize = lambda text: sp_model.decode(sp_model.encode(text))

    def add_splitter(prompt, generated_text):
        normalized_prompt = normalize(prompt)
        normalized_generated_text = normalize(generated_text)

        assert normalized_prompt == normalized_generated_text[: len(normalized_prompt)]

        return (
            normalized_prompt
            + " *** "
            + normalized_generated_text[len(normalized_prompt) :]
        )

    results = []

    for prompt in PROMPTS:
        result = {
            "prompt": prompt,
            "marks": [],
        }
        prompt = normalize(prompt)
        result["normalized_prompt"] = prompt
        mean_values = {}
        for i in tqdm(range(5)):
            start = time()
            response = universal_inference(prompt=prompt)
            splitted = add_splitter(prompt, response)
            marks = get_marks(splitted)
            marks["generated_text"] = response
            marks["splitted"] = splitted
            result["marks"].append(marks)
            end = time()
            sleep(20 - min((end - start), 20))

        for key in result["marks"][0]:
            if key == "generated_text" or key == "splitted" or key == "age":
                continue
            mean_values[key] = sum(
                [int(marks[key]) for marks in result["marks"]]
            ) / len(result["marks"])

        result["mean_values"] = mean_values
        results.append(result)
    with open("chat_gpt_results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY"))

    parser.add_argument(
        "--model_path",
        type=str,
        default="big_model",
        help="Path to folder with model. For gpt2xl use 'gpt2xl'",
    )

    args = parser.parse_args()

    client = OpenAI(
        # This is the default and can be omitted
        api_key=args.api_key,
    )

    main(client, args.model_path)
