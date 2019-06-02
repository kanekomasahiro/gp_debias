import numpy as np
from numpy import dot
from numpy.linalg import norm
from gensim.models import KeyedVectors
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='')
    parser.add_argument('-g', '--sembias', type=str, default='SemBias/SemBias', help='')
    args = parser.parse_args()
    input_type = args.input[-3:]

    if input_type == 'txt':
        emb = KeyedVectors.load_word2vec_format(args.input,
                                                binary=False)
    elif input_type == 'bin':
        emb = KeyedVectors.load_word2vec_format(args.input,
                                                binary=True)

    def eval_bias_analogy(model):
        bias_analogy_f = open(args.sembias)

        definition_num = 0
        none_num = 0
        stereotype_num = 0
        total_num = 0
        sub_definition_num = 0
        sub_none_num = 0
        sub_stereotype_num = 0
        sub_size = 40

        sub_start = -(sub_size - sum(1 for line in open(args.sembias)))

        gender_v = model['he'] - model['she']
        for sub_idx, l in enumerate(bias_analogy_f):
            l = l.strip().split()
            max_score = -100
            for i, word_pair in enumerate(l):
                word_pair = word_pair.split(':')
                pre_v = model[word_pair[0]] - model[word_pair[1]]
                score = dot(gender_v, pre_v)/(norm(gender_v)*norm(pre_v))
                if score > max_score:
                    max_idx = i
                    max_score = score
            if max_idx == 0:
                definition_num += 1
                if sub_idx >= sub_start:
                    sub_definition_num += 1
            elif max_idx == 1 or max_idx == 2:
                none_num += 1
                if sub_idx >= sub_start:
                    sub_none_num += 1
            elif max_idx == 3:
                stereotype_num += 1
                if sub_idx >= sub_start:
                    sub_stereotype_num += 1
            total_num += 1
        if definition_num == 0:
            print('definition: 0')
        else:
            print('definition: {}'.format(definition_num / total_num))
        if stereotype_num == 0:
            print('stereotype: 0')
        else:
            print('stereotype: {}'.format(stereotype_num / total_num))
        if none_num == 0:
            print('none: 0')
        else:
            print('none: {}'.format(none_num / total_num))

        if sub_definition_num == 0:
            print('sub definition: 0')
        else:
            print('sub definition: {:.5f}'.format(sub_definition_num / sub_size))
        if sub_stereotype_num == 0:
            print('sub stereotype: 0')
        else:
            print('substereotype: {:.5f}'.format(sub_stereotype_num / sub_size))
        if sub_none_num == 0:
            print('sub none: 0')
        else:
            print('sub none: {:.5f}'.format(sub_none_num / sub_size))

    eval_bias_analogy(emb)

if __name__ == "__main__":
    main()
