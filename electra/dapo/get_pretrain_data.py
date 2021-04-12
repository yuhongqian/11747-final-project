"""
Writes and mix-up all pre-training dialog datasets into a specific format, where each line represents a conversation
as a list of utterances.

Four datasets are used: DailyDialog, PERSONA-CHAT, Topical-Chat, BlendedSkillTalk

TODO: if it doesn't work well on mutual, may be try adding f/m in the front & tokenize (stanford tokenizer) the data
    in the same way
"""

import json


def split_conversation(conversation):
    """
    Splits one long conversation into overlapping conversations of 10 utterances.
    :return: a list of conversations = a list of list of utterances
    """
    if len(conversation) <= 10:
        return
    else:
        conversations = []
        for start in range(0, len(conversation) - 10, 5):
            conversations.append(conversation[start:start+10])
    return conversations


def write_conversation(fp, conversation):
    conversations = split_conversation(conversation)
    if conversations is not None:
        for c in conversations:
            fp.write(json.dumps(c) + "\n")
    else:
        fp.write(json.dumps(conversation) + "\n")


def write_daily_dialog(fp, data_path="/bos/tmp10/hongqiay/dailydialog/dialogues_text.txt"):
    seperator = "__eou__"
    with open(data_path, "r") as fin:
        for l in fin:
            conversation = l.split(seperator)
            write_conversation(fp, conversation)


def write_persona_chat(fp, data_path="/bos/tmp10/hongqiay/personachat/train_none_original.txt"):
    curr_conv = []
    with open(data_path, "r") as fin:
        for l in fin:
            u1, u2 = l.split("\t")[:2]
            if u1.startswith("1"):
                write_conversation(fp, curr_conv)
                curr_conv = []
            curr_conv.append(u1[1:])    # remove the turn number
            curr_conv.append(u2)


def main():
    with open("/bos/tmp10/hongqiay/mutual_pretrain/dailydialog.txt", "w") as f:
        write_daily_dialog(f)
    with open("/bos/tmp10/hongqiay/mutual_pretrain/personachat.txt", "w") as f:
        write_persona_chat(f)


if __name__ == '__main__':
    main()