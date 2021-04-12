import sys
import json


def calculate_metrics(output_txt, json_data_path="../MuTual/data/mutual/dev/dev.json"):
    gold = dict()
    with open(json_data_path, "r") as f:
        all_data = json.load(f)
        for example_id, data in all_data.items():
            gold[example_id] = data["answers"]
    with open(output_txt, "r") as f:
        r1, r2, mrr = 0, 0, 0
        for i, l in enumerate(f):
            items = l.split()
            example_id = items[0]
            choices = items[1:]
            index = choices.index(gold[example_id])
            if index == 0:
                r1 += 1
            elif index == 1:
                r2 += 1
            mrr += 1 / (index + 1)
    total_examples = len(gold)
    print("R@1: %.3f \t R@2: %.3f \t MRR %.3f" %
          (r1 / total_examples, (r1 / total_examples + r2 / total_examples), mrr / total_examples))


def main():
    calculate_metrics(sys.argv[1])


if __name__ == '__main__':
    main()