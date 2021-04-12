"""
Writes and mix-up all pre-training dialog datasets into a specific format, where each line represents a conversation
as a list of utterances.

Four datasets are used: DailyDialog, PERSONA-CHAT, Topical-Chat, BlendedSkillTalk

TODO: if it doesn't work well on mutual, may be try adding f/m in the front & tokenize (stanford tokenizer) the data
    in the same way
"""
import os
import math
import json
import random
from collections import Counter
import nltk
from nltk import trigrams


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
            conversation = [s.strip() for s in l.split(seperator) if s.strip() != ""]
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


def write_topical_chat(fp, data_path="/bos/tmp10/hongqiay/Topical-Chat/conversations/train.json"):
    with open(data_path, "r") as fin:
        all_data = json.load(fin)
        for example_id, example in all_data.items():
            conversation = [content["message"] for content in example["content"]]
            write_conversation(fp, conversation)


def write_blendedskill(fp, data_path="/bos/tmp10/hongqiay/blendedskill/train.json"):
    with open(data_path, "r") as fin:
        all_data = json.load(fin)
        for example in all_data:
            conversation = [example["free_turker_utterance"], example["guided_turker_utterance"]]
            conversation.extend([turn[1] for turn in example["dialog"]])
            write_conversation(fp, conversation)


def write_individual():
    with open("/bos/tmp10/hongqiay/mutual_pretrain/individual/dailydialog.txt", "w") as f:
        write_daily_dialog(f)
    with open("/bos/tmp10/hongqiay/mutual_pretrain/individual/personachat.txt", "w") as f:
        write_persona_chat(f)
    with open("/bos/tmp10/hongqiay/mutual_pretrain/individual/topicalchat.txt", "w") as f:
        write_topical_chat(f)
    with open("/bos/tmp10/hongqiay/mutual_pretrain/individual/blendedskill.txt", "w") as f:
        write_blendedskill(f)


def utterance_order(conversation):
    """
    Reorder the utterances
    """
    res = conversation.copy()
    random.shuffle(res)
    return res


def utterance_insert(conversation):
    """
    Randomly removes one utterance and reinserts it at another location in the conversation
    """
    old_idx = random.randrange(len(conversation))
    new_idx = old_idx
    while new_idx == old_idx:
        new_idx = random.randrange(len(conversation))

    if new_idx > old_idx:
        return conversation[:old_idx] + conversation[old_idx+1:new_idx] + [conversation[old_idx]] + conversation[new_idx:]
    else:
        return conversation[:new_idx] + [conversation[old_idx]] + conversation[new_idx:old_idx] + conversation[old_idx+1:]


def utterance_replace(conversation, utterance):
    """
    Replaces a randomly selected utterance with the input utterance
    """
    idx = random.randrange(len(conversation))
    res = conversation.copy()
    res[idx] = utterance
    return res


def write_conversation_pretrain(fp, conversation, label):
    """
    Writes a conversation example for pretraining
    """
    if type(conversation) == list:
        conversation = "".join(conversation)
    example = dict()
    example["label"] = label
    example["conversation"] = conversation
    fp.write(json.dumps(example) + "\n")


def nidf(ngram_count, num_convs, min_idf, max_idf):
    idf = math.log2(num_convs / ngram_count)
    return (idf - min_idf) / (max_idf - min_idf)


def get_all_positive_conversations(fp, individual_dir="/bos/tmp10/hongqiay/mutual_pretrain/individual"):
    """
    Gathers and writes all positive conversations. Counts the trigram document frequency.
    :param individual_dir:
    :return:
    """
    all_positive_conversations = []
    ngram_counts = Counter()
    for fname in os.listdir(individual_dir):
        with open(os.path.join(individual_dir, fname), "r") as f:
            for l in f:
                conversation = json.loads(l)
                conversation_text = "".join(conversation)
                tokens = nltk.word_tokenize(conversation_text)
                tri_tokens = trigrams(tokens)
                for trigram in set(tri_tokens):
                    ngram_counts[trigram] += 1
                all_positive_conversations.append(conversation)
    min_count = ngram_counts[min(ngram_counts)]
    min_idf = math.log2(len(all_positive_conversations) / min_count)
    max_count = ngram_counts[max(ngram_counts)]
    max_idf = math.log2(len(all_positive_conversations) / max_count)
    for k, v in ngram_counts:
        ngram_counts[k] = nidf(v, len(all_positive_conversations), min_idf, max_idf)    # now stores nidf
    return all_positive_conversations, ngram_counts


def get_all_negative_conversations(fp, all_positive_conversations):
    for pos_conv in all_positive_conversations:
        write_conversation_pretrain(fp, utterance_order(pos_conv), 0)
        write_conversation_pretrain(fp, utterance_insert(pos_conv), 0)
        replace_utterance = random.choice(random.choice(all_positive_conversations))
        write_conversation_pretrain(fp, utterance_replace(pos_conv, replace_utterance), 0)


def get_positive_score(conversation, nidfs):
    conversation_text = "".join(conversation)
    tokens = nltk.word_tokenize(conversation_text)
    tri_tokens = trigrams(tokens)
    score = 0
    num_trigrams = 0
    for trigram in tri_tokens:
        num_trigrams += 1
        score += nidfs[trigram]
    return score / num_trigrams


def get_pretrain_dataset(output_path="/bos/tmp10/hongqiay/mutual_pretrain/pretrain_data.txt"):
    with open(output_path, "w") as f:
        all_positive_conversations, nidfs = get_all_positive_conversations(f)
        for conversation in all_positive_conversations:
            pos_score = get_positive_score(conversation, nidfs)
            write_conversation_pretrain(f, conversation, round(pos_score, 4))
            get_all_negative_conversations(f, all_positive_conversations)


def main():
    seed = 42
    random.seed(seed)
    get_pretrain_dataset()


if __name__ == '__main__':
    main()