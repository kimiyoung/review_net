
import os
import re
import random

def camel_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def comment_tokenize(text):
    words = text.split()
    ret = []
    for word in words:
        t_word = ''
        for c in word:
            if c.isdigit():
                t_word += '0'
            elif c.isalpha():
                t_word += c
        names = camel_case(t_word).split('_')
        for name in names:
            if name != '':
                ret.append(name)
    return ret

def proc_file(filename):
    prev_i = - 100
    code_seq, comment_seq = [], []
    for i, line in enumerate(open(filename)):
        inputs = line.strip().split('\t')
        if len(inputs) > 1 and inputs[1] == 'TokenNameclass':
            if i - prev_i < 5:
                comment_seq = comment_tokenize(prev_comment)
        if len(inputs) > 2 and inputs[1] == 'TokenNameCOMMENT_JAVADOC':
            prev_i = i
            prev_comment = inputs[2].strip()
        if 'COMMENT' not in inputs[1]:
            if 'Identifier' in inputs[1]:
                code_seq += comment_tokenize(inputs[2])
            else:
                code_seq.append(inputs[1])
    if len(code_seq) >= 1 and len(comment_seq) >= 1:
        return [code_seq, comment_seq]
    return None

def proc_all_dir():
    PROJECTS = ['MinorThird', 'apache-ant-1.8.4', 'apache-cassandra-1.2.0', 'apache-log4j-1.2.17', 'apache-maven-3.0.4', \
        'batik-1.7', 'lucene-3.6.2', 'xalan-j-2.7.1', 'xerces-2.11.0']
    PREFIX = 'habeascorpus-data-withComments/habeascorpus_tokens/'
    seq_data = []
    for project in PROJECTS:
        dir = PREFIX + project
        for dir_name, _, file_list in os.walk(dir):
            for suffix in file_list:
                if not suffix.endswith('.java'): continue
                filename = os.path.join(dir_name, suffix)
                t_seq = proc_file(filename)
                if t_seq is not None:
                    seq_data.append(t_seq)
    return seq_data

# def print_data(train_file, test_file, ratio = 0.9):
#     random.seed(13)
#     seq_data = proc_all_dir()
#     fout_train, fout_test = open(train_file, 'w'), open(test_file, 'w')
#     for seq_datum in seq_data:
#         code_seq, comment_seq = tuple(seq_datum)
#         fout = fout_train if random.random() < ratio else fout_test
#         fout.write("{}\t{}\n".format(" ".join(code_seq), " ".join(comment_seq)))

def print_data(train_file, dev_file, test_file):
    random.seed(13)
    seq_data = proc_all_dir()
    fout_train, fout_dev, fout_test = open(train_file, 'w'), open(dev_file, 'w'), open(test_file, 'w')
    for seq_datum in seq_data:
        code_seq, comment_seq = tuple(seq_datum)
        fout = fout_train
        r = random.random()
        if r > 0.9:
            fout = fout_test
        elif r > 0.8:
            fout = fout_dev
        fout.write("{}\t{}\n".format(" ".join(code_seq), " ".join(comment_seq)))

if __name__ == '__main__':
    print_data('train.dat', 'dev.dat', 'test.dat')

