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
    out_annotation_file = Path(str(out_file_name.absolute()) + "_CONVERSATION")
    if not out_annotation_file.exists():
        with out_annotation_file.open("w", encoding="utf-8", newline="") as f:
            f.write(str(conversation) + "\n\n")
    else:
        with out_annotation_file.open("a", encoding="utf-8", newline="") as f:
            f.write(str(conversation) + "\n\n")

def main():

    start_time = datetime.now()

    parser = argparse.ArgumentParser(
        description="Metaphors Ratings Script with llms using Huggingface API",
        usage="python paced_huggingface_API_calls.py --model 'google/gemma-3-27b-it:nebius' --dataset clean_MB.csv --prompt MB_task_instructions.txt --history --test",
    )

    parser.add_argument(
        "--dataset",
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
        out_file_name = "_TEST_met_ratings_llm_"
    else:
        out_file_name = "met_ratings_llm_"

    if KEEP_HISTORY:
        out_file_name += "keep-history_"
    else:
        out_file_name += "no-history_"

    model_name = MODEL.replace(":", "-").replace("/", "-")

    out_annotation_file = Path(
        DATA_PATH,
        "synthetic_annotations",
        out_file_name,
        model_name,
        ".csv"
    )

    run_config = {
        "time": str(start_time.isoformat().replace(":", "-").split(".")[0]),
        "n_raters": RATERS,
        "method": "API calls with huggingface_hub",
        "model": MODEL,
        "keep_history": KEEP_HISTORY,
        "prompt": TASK_INSTRUCTIONS,
    }

    dataset = Path(DATA_PATH, str(args.dataset))
    dataset_df = pd.read_csv(dataset, encoding="utf-8")
    checkpoint_file = Path("checkpoint.csv")

    if not checkpoint_file.exists():
        dataset_df.to_csv(checkpoint_file, index = False)
        checkpoint_df = pd.read_csv(checkpoint_file, encoding="utf-8")

        with open("current_rater.txt", "W", encoding = "utf-8") as f:
            f.write(str(1))

    else:
        checkpoint_df = pd.read_csv(checkpoint_file, encoding = "utf-8")

        if not checkpoint_df:
            dataset_df.to_csv(checkpoint_file, index = False)

            with open("current_rater.txt", "r+", encoding = "utf-8") as f:
                previous_rater = int(f.read().strip)
                f.write(str(previous_rater + 1))

    with open("current_rater.txt", "r", encoding = "utf-8") as f:
        rater = f.read().strip()


    if rater < RATERS:

        metaphors_list = check_point_df["Metaphor"]
        structures_list = check_point_df["Met_structure"]

        rater_time = datetime.now()

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": TASK_INSTRUCTIONS}]
                },
            {
                "role" : "user",
                "content": [{"type": "text", "text": ""}]
                }
        ]

        for idx, metaphor in list(enumerate(metaphors_list)):
            
            print(rater, idx + 1, "of", len(metaphors_list))
            structure = structures_list[idx]

            client = InferenceClient(api_key=os.environ["HF_TOKEN"])

            conversation[-1]["content"][0]["text"] = metaphor

            completion = client.chat.completions.create(
                model=MODEL,
                messages=conversation
                )

            reply = completion.choices[0].message.content # content è un attributo dell'oggetto ChatCompletionOutputMessage
            print("output: ", reply)

            check_point_df = check_point_df[1:]

            if KEEP_HISTORY:

                conversation.append({"role" : "assistant", "content": [{"type": "text", "text": reply}]})
                conversation.append({"role" : "user", "content": [{"type": "text", "text": ""}]})

            track_conversation(f"rater_{rater}_conversation" + str(out_annotation_file), conversation)

            values=reply_to_values(reply)
            print("values: ", values, "\n")

            if "MB" in args.prompt:

                row = {
                    "annotator": rater,
                    "metaphor": metaphor,
                    "metaphor_structure" : structure,
                    "FAMILIARITY_synthetic" : int(values[0]),
                    "MEANINGFULNESS_synthetic" : int(values[1]),
                    "body relatedness" : int(values[2])
                }

            if "ME" in args.prompt:

                row = {
                    "annotator": rater,
                    "metaphor": metaphor,
                    "metaphor_structure" : structure,
                    "FAMILIARITY_synthetic" : int(values[0]),
                    "MEANINGFULNESS_synthetic" : int(values[1]),
                    "DIFFICULTY_synthetic" : int(values[2])
                }

            if "MI" in args.prompt:

                row = {
                    "annotator": rater,
                    "metaphor": metaphor,
                    "metaphor_structure" : structure,
                    "PHISICALITY_synthetic" : int(values[0]),
                    "IMAGEBILITY_synthetic" : int(values[1]),
                }

            if "MM" in args.prompt:

                row = {
                    "annotator": rater,
                    "metaphor": metaphor,
                    "metaphor_structure" : structure,
                    "FAMILIARITY_synthetic" : int(values[0]),
                    "MEANINGFULNESS_synthetic" : int(values[1]),
                }

            write_out(out_annotation_file, row)

        print(f"{rater} completed in: {datetime.now() - rater_time}")

    else:

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


# import time
# time.sleep(300)   # 300 secondi = 5 minuti
