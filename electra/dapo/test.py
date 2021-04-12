import json


def length_test(in_file):
    with open(in_file, "r") as f:
        for l in f:
            conv = json.loads(l)
            assert(len(conv) <= 10)


def main():
    length_test("/bos/tmp10/hongqiay/mutual_pretrain/dailydialog.txt")


if __name__ == '__main__':
    main()