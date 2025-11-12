# Example usage: python langchain_bws.py --model mistral:instruct --prompt specificity_task_instructions.txt --raters 12
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.output_parsers import CommaSeparatedListOutputParser
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import csv
import argparse

def reply_to_values(response):
    values_list = response.split(",")
    for idx, value in enumerate(values_list):
        values_list[idx] = "".join([c for c in value if c.isdigit()])
    return values_list


def annotate(metaphor, history=False):
    if history:
        request = {
            "chat_history": chat_history,
            "input": metaphor,
        }
    else:
        request = {
            "input": metaphor,
        }

    response = chain.invoke(request)

    return response

def write_out(out_file_name, results_dict):
    out_annotation_file = Path(str(out_file_name.absolute()))
    if not out_annotation_file.exists():
        with out_annotation_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results_dict.keys())
            writer.writeheader()
            writer.writerow(results_dict)
    else:
        with out_annotation_file.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results_dict.keys())
            writer.writerow(results_dict)


if __name__ == "__main__":
    start_time = datetime.now()

    parser = argparse.ArgumentParser(
        description="Metaphors Ratings Script with llms, langchain and Ollama API",
        usage="python langchain_met_ratings.py --model 'gemma3:1b' --metaphors_file new_MB.csv --prompt MB_task_instructions.txt --history --test",
    )

    parser.add_argument(
        "--metaphors_file",
        type=Path,
        help="study to replicate"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="mistral:instruct",
        help="Model name",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Path to data",
    )

    parser.add_argument("--history", action="store_true", help="Keep history")

    parser.add_argument(
        "--raters",
        type=int,
        default=1,
        help="number of raters to annotate each metaphor",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in testing mode",
    )

    args = parser.parse_args()

    MODEL = args.model

    if args.history:
        KEEP_HISTORY = True
    else:
        KEEP_HISTORY = False

    TASK_INSTRUCTIONS = open(args.prompt, "r").read()

    if args.test:
        TEST = True
    else:
        TEST = False

    DATA_PATH = "data/new_datasets/"
    RATERS = args.raters

    model_name = MODEL.replace(":", "-")
    if TEST:
        out_file_name = "_TEST_met_ratings_llm-langchain_"
    else:
        out_file_name = "met_ratings_llm-langchain_"

    if KEEP_HISTORY:
        out_file_name += "keep-history_"
    else:
        out_file_name += "no-history_"
    out_annotation_file = Path(
        DATA_PATH,
        "synthetic_annotations",
        out_file_name
        + model_name
        + "_"
        + str(start_time.isoformat().replace(":", "-").split(".")[0])
        + ".csv"
    )

    print(out_annotation_file)

    run_config = {
        "time": str(start_time.isoformat().replace(":", "-").split(".")[0]),
        "n_raters": RATERS,
        "method": "langchain-ollama",
        "model": MODEL,
        "keep_history": KEEP_HISTORY,
        "prompt": TASK_INSTRUCTIONS,
    }

    metaphors_file = Path(DATA_PATH, str(args.metaphors_file))
    df_metaphors = pd.read_csv(metaphors_file, encoding="utf-8")

    if args.test:
        df_metaphors=df_metaphors[:5]

    metaphor_list = df_metaphors["Metaphor"]

    for n in range(RATERS):
        rater_time = datetime.now()
        rater = f"rater_{n + 1}"
        print("rater: ", rater)

        llm = Ollama(model=MODEL, num_predict=48)  # , temperature=0.2)
        if KEEP_HISTORY:
            chat_history = []
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", TASK_INSTRUCTIONS),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("user", "{input}"),
                ]
            )
        else:
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", TASK_INSTRUCTIONS),
                    ("user", "{input}"),
                ]
            )
        chain = prompt_template | llm

        for idx, metaphor in list(enumerate(metaphor_list)):
            print(rater, idx + 1, "of", len(metaphor_list))

            reply = annotate(metaphor, history=KEEP_HISTORY)
            print("output: ", reply)
            values=reply_to_values(reply)
            print("values: ", values)

            if "MB" in args.prompt:

                row = {
                    "annotator": rater,
                    "metaphor": metaphor,
                    "familiarity" : int(values[0]),
                    "meaningfulness" : int(values[1]),
                    "body relatedness" : int(values[2])
                }

#################################################################

            if "ME" in args.prompt:

                row = {
                    "annotator": rater,
                    "metaphor": metaphor,
                    "familiarity" : values[0],
                    "meaningfulness" : values[1],
                    "difficulty" : values[2]
                }

            if "MI" in args.prompt:

                row = {
                    "annotator": rater,
                    "metaphor": metaphor,
                    "phisicality" : values[0],
                    "imageability" : values[1],
                }

            if "MM" in args.prompt:

                row = {
                    "annotator": rater,
                    "metaphor": metaphor,
                    "familiarity" : values[0],
                    "meaningfulness" : values[1],
                }

############################################################################

            if KEEP_HISTORY and None not in values:
                chat_history.append(HumanMessage(content=metaphor))
                chat_history.append(AIMessage(content=reply))

            write_out(out_annotation_file, row)

        print(f"{rater} completed in: {datetime.now() - rater_time}")

    with open(str(out_annotation_file.absolute()) + "_CONFIG.json", "w") as f:
        json.dump(run_config, f)

    print("Metaphor rating completed in: {}".format(datetime.now() - start_time))