import argparse
import json

import language_tool_python


def check_grammar(text):
    tool = language_tool_python.LanguageTool("en-US")
    matches = tool.check(text)
    non_capitalization_errors = [m for m in matches if "CASE" not in m.ruleId]
    print(f"{non_capitalization_errors}")
    error_count = len(non_capitalization_errors)

    return error_count / len(text.split(" "))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str)
    args = parser.parse_args()

    with open(args.results_path, "r") as f:
        results = json.load(f)

    errors_per_word = 0
    generations = 0
    for prompt_results in results[:-1]:
        for one_generation in results:
            generations += 1
            errors_per_word += check_grammar(one_generation["generated_text"])

    print(f"Mean grammar errors per word: {errors_per_word / generations}")
