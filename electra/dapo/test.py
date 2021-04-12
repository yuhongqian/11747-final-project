import json
from get_pretrain_data import utterance_insert


def length_test(in_file):
    with open(in_file, "r") as f:
        for l in f:
            conv = json.loads(l)
            assert(len(conv) <= 10)


def ui_test():
    print(utterance_insert([1,2,3,4]))


def main():
    # length_test("/bos/tmp10/hongqiay/mutual_pretrain/dailydialog.txt")
    ui_test()


if __name__ == '__main__':
    main()