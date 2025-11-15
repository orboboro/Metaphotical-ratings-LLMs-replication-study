import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import csv
import argparse
import os
from huggingface_hub import InferenceClient

def reply_to_values(response):
    values_list = response.split(",")
    for idx, value in enumerate(values_list):
        values_list[idx] = "".join([c for c in value if c.isdigit()])
    return values_list

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

def track_conversation(out_file_name, conversation):
    out_annotation_file = Path(str(out_file_name.absolute()))
    if not out_annotation_file.exists():
        with out_annotation_file.open("w", encoding="utf-8", newline="") as f:
            f.write(str(conversation + "\n\n"))
    else:
        with out_annotation_file.open("a", encoding="utf-8", newline="") as f:
            f.write(str(conversation + "\n\n"))

def main():

    start_time = datetime.now()

    parser = argparse.ArgumentParser(
        description="Metaphors Ratings Script with llms, langchain and Ollama API",
        usage="python huggingface_API_calls.py --model 'google/gemma-3-27b-it:nebius' --metaphors_file new_MB.csv --prompt MB_task_instructions.txt --history --test",
    )

    parser.add_argument(
        "--metaphors_file",
        type=Path,
        help="Target study for replication"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Path to data",
    )

    parser.add_argument(
        "--history",
        action="store_true",
        help="Keep history")

    parser.add_argument(
        "--raters",
        type=int,
        default=1,
        help="Number of raters to annotate each metaphor",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in testing mode",
    )

    args = parser.parse_args()

    MODEL = (args.model).replace(":", "-")

    if args.history:
        KEEP_HISTORY = True
    else:
        KEEP_HISTORY = False

    TASK_INSTRUCTIONS = open(args.prompt, "r", encoding="utf-8").read()

    if args.test:
        TEST = True
    else:
        TEST = False

    DATA_PATH = "data/new_datasets/"
    RATERS = args.raters

    if TEST:
        out_file_name = "_TEST_met_ratings_llm-langchain_"
    else:
        out_file_name = "met_ratings_llm-langchain_"

    if KEEP_HISTORY:
        out_file_name += "keep-history_"
    else:
        out_file_name += "no-history_"

    model_name = MODEL.replace(":", "-").replace("/", "-")

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
        "method": "API calls with huggingface_hub",
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
        print("RATER: ", rater, "\n")

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": TASK_INSTRUCTIONS}]
                },
            {
                "role" : "assistant",
                "content": [{"type": "text", "text": ""}]
                }
        ]

        for idx, metaphor in list(enumerate(metaphor_list)):
            print(rater, idx + 1, "of", len(metaphor_list))

            client = InferenceClient(api_key=os.environ["HF_TOKEN"], provider="nebius")

            conversation[-1]["content"][0]["text"] = metaphor

            completion = client.chat.completions.create(
                model=MODEL,
                messages=conversation
                )

            reply = completion.choices[0].message.content # content è un attributo dell'oggetto ChatCompletionOutputMessage
            print("output: ", reply)

            if KEEP_HISTORY:

                conversation.append({"role" : "assistant", "content": [{"type": "text", "text": reply}]})
                conversation.append({"role" : "assistant", "content": [{"type": "text", "text": ""}]})

            track_conversation(out_annotation_file.absolute() + "_CONVERSATION", conversation)

            values=reply_to_values(reply)
            print("values: ", values, "\n")

            if idx == len(metaphor_list) - 1:
                print("CONVERSATION:\n", conversation)

            if "MB" in args.prompt:

                row = {
                    "annotator": rater,
                    "metaphor": metaphor,
                    "familiarity" : int(values[0]),
                    "meaningfulness" : int(values[1]),
                    "body relatedness" : int(values[2])
                }

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

            write_out(out_annotation_file, row)

        print(f"{rater} completed in: {datetime.now() - rater_time}")

    with open(str(out_annotation_file.absolute()) + "_CONFIG.json", "w") as f:
        json.dump(run_config, f)

    print("Metaphor rating completed in: {}".format(datetime.now() - start_time))

if __name__ == "__main__": # La variabile speciale __name__ viene inizializzata uguale a "__main__" quando un file python viene eseguito
    main()                 # direttamente. Dunque la condizione __name__ == "__main__ è rispettata e quindi il contenuto delle funzione
                            # main viene eseguito. invece, se il file .py viene importato in un altro file, il suo contenuto non verrà
                            # eseguito, perché dal momento che il file non è eseguito direttamente, __name__ non sarà uguale alla stringa
                            # "__main__", ma al nome stesso del file .py. Insomma questa condizione serve a far sì che una funzione
                            # contenuta in un file venga eseguita solo quando è chiamata firettamente da terminale e nonquando è importata
                            # come modulo da altri file. 
                            # Reference: https://www.youtube.com/watch?v=sugvnHA7ElY