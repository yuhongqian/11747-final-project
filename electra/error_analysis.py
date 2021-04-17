import sys
import csv
import json
from data import ANSWER_IDX
import pdb


def read_output(output_file, analysis_res_path, gold_std="../MuTual/data/mutual/dev/dev.json"):
    output = dict()

    with open(gold_std, "r") as f:
        gold = json.load(f)
    with open(output_file, "r") as f:
        reader1 = csv.reader(f, delimiter="\t")
        for row in reader1:
            output[row[0]] = row[1]

    with open(analysis_res_path, "w") as f:
        for example, contents in gold.items():
            output_answer = output[example]
            gold_answer = contents["answers"]
            options = contents["options"]
            if output_answer != gold_answer:
                res = {
                    "id": example,
                    "context": contents["article"],
                    "system_answer": options[ANSWER_IDX[output_answer]],
                    "gold_answer": options[ANSWER_IDX[gold_answer]]
                }
                f.write(json.dumps(res) + "\n")


if __name__ == '__main__':
    read_output(sys.argv[1], sys.argv[2])





