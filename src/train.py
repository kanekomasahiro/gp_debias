import torch
import sys
import os
import time
import pickle
import copy
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
from gensim.models import KeyedVectors
from math import ceil
import random
from pre_train_autoencoder import pre_train_autoencoder
from pre_train_classifier import pre_train_classifier
import optim
import model

sys.path.append('./hyperparams/')
if sys.argv[1] == 'glove':
    from hyperparams_glove import Hyperparams as hp
elif sys.argv[1] == 'gn':
    from hyperparams_gn_glove import Hyperparams as hp

if hp.gpu:
    cuda.set_device(hp.gpu)
    cuda.manual_seed_all(hp.seed)
torch.manual_seed(hp.seed)
random.seed(hp.seed)
np.random.seed(hp.seed)

def shuffle_data(words, tags=None, limit_size=None, sampling=None):
    perm = torch.randperm(len(words))
    words = [words[idx.item()] for idx in perm]
    if tags != None:
        tags = [tags[idx.item()] for idx in perm]
    if sampling == 'under_sampling':
        words = words[:limit_size]
        if tags != None:
            tags = tags[:limit_size]
    elif sampling == 'over_sampling':
        limit_size -= len(words)
        words += [random.choice(words) for _ in range(limit_size)]
        if tags != None:
            tags += [tags[0] for _ in range(limit_size)]

    return words, tags

def create_train_dev(gender_words, no_gender_words, stereotype_words):
    stereotype_words = stereotype_words['female'] + stereotype_words['male']

    gender_pairs = [[female, male] for female, male in zip(gender_words['female'], gender_words['male'])]

    no_gender_words, _ = shuffle_data(no_gender_words)
    gender_pairs, _ = shuffle_data(gender_pairs)
    stereotype_words, _ = shuffle_data(stereotype_words)

    train_words = {}
    train_words['no gender'] = no_gender_words[:-hp.dev_num]
    train_words['female & male'] = gender_pairs[:-hp.dev_num]
    train_words['stereotype'] = stereotype_words[:-hp.dev_num]

    dev_words = {}
    dev_words['no gender'] = no_gender_words[-hp.dev_num:]
    dev_words['female & male'] = gender_pairs[-hp.dev_num:]
    dev_words['stereotype'] = stereotype_words[-hp.dev_num:]

    return train_words, dev_words

def trainModel(encoder, encoder_optim,
               female_classifier, female_classifier_optim,
               male_classifier, male_classifier_optim,
               decoder, decoder_optim,
               train_words, dev_words,
               word2emb):

    train_size = len(train_words['no gender']) + len(train_words['female & male']) * 2 + len(train_words['stereotype'])
    dev_size = len(dev_words['no gender']) + len(dev_words['female & male']) * 2 + len(dev_words['stereotype'])

    classifier_criterion = nn.BCELoss()
    decoder_criterion = nn.MSELoss()

    def run_model(words, gender_vektor, mode):
        if mode == 'train':
            encoder.train()
            female_classifier.train()
            male_classifier.train()
            decoder.train()
        elif model == 'eval':
            encoder.eval()
            female_classifier.eval()
            male_classifier.eval()
            decoder.eval()

        total_loss = 0
        total_female_classifier_loss = 0
        total_male_classifier_loss = 0
        total_decoder_loss = 0
        total_gender_stereotype_loss = 0
        total_gender_no_gender_loss = 0
        total_gender_vektor_loss = 0

        female_classifier_correct = 0
        male_classifier_correct = 0

        if mode == 'train':
            limit_size = max([len(words['no gender']),
                            len(words['female & male']),
                            len(words['stereotype'])])

            words['no gender'], _ = \
                shuffle_data(words['no gender'],
                limit_size=limit_size,
                sampling=hp.sampling)
            words['female & male'], _ = \
                shuffle_data(words['female & male'],
                limit_size=limit_size,
                sampling=hp.sampling)
            words['stereotype'], _ = \
                shuffle_data(words['stereotype'],
                limit_size=limit_size,
                sampling=hp.sampling)
        elif mode == 'eval':
            limit_size = hp.dev_num

        inputs = [[gender[0], gender[1], stereotype, no_gender] for gender, stereotype, no_gender in zip(words['female & male'], words['stereotype'], words['no gender'])]

        data_size = len(inputs) * 4

        def make_gold_label():
            gold = torch.FloatTensor(range(2)).view(2, 1)

            return gold.cuda() if hp.gpu >= 0 else gold

        def classify(hidden, gold, classifier):
            classifier.zero_grad()
            pre = classifier(hidden)
            loss = classifier_criterion(pre, gold)
            return pre, loss

        for i, words in enumerate(inputs):
            emb = torch.stack([torch.from_numpy(word2emb[word]) for word in words])
            if hp.gpu >= 0:
                emb = emb.cuda()
            encoder.zero_grad()
            hidden = encoder(emb)

            decoder.zero_grad()
            pre = decoder(hidden)
            decoder_loss = decoder_criterion(pre, emb)

            female_gold = torch.FloatTensor([1, 0, 0, 0]).view(-1, 1) # female, male, stereotype, no gender
            male_gold = torch.FloatTensor([0, 1, 0, 0]).view(-1, 1) # female, male, stereotype, no gender
            if hp.gpu >= 0:
                female_gold = female_gold.cuda()
                male_gold = male_gold.cuda()
            #female_pre, female_loss = classify(hidden[:3], female_gold, female_classifier)
            if hp.classifier_loss:
                female_pre, female_loss = classify(hidden, female_gold, female_classifier)
                male_pre, male_loss = classify(hidden, male_gold, male_classifier)
            gender_stereotype_loss = torch.sum(gender_vektor * hidden[2])**2
            if hp.gender_no_gender_loss:
                gender_no_gender_loss = torch.sum(gender_vektor * hidden[3])**2
            else:
                gender_no_gender_loss = 0
            if hp.gender_vektor_loss:
                gender_vektor_loss = decoder_criterion(gender_vektor, (hidden[1] - hidden[0]))
            else:
                gender_vektor_loss = 0

            decoder_loss *= hp.decoder_loss_rate
            if hp.classifier_loss:
                female_loss *= hp.female_loss_rate
                male_loss *= hp.male_loss_rate
            else:
                female_loss = 0
                male_loss = 0
            gender_stereotype_loss *= hp.gender_stereotype_loss_rate
            gender_no_gender_loss *= hp.gender_no_gender_loss_rate
            gender_vektor_loss *= hp.gender_vektor_loss_rate

            loss = decoder_loss \
                 + female_loss \
                 + male_loss \
                 + gender_stereotype_loss \
                 + gender_no_gender_loss \
                 + gender_vektor_loss

            total_decoder_loss += decoder_loss.item()
            total_gender_stereotype_loss += gender_stereotype_loss.item()
            if hp.classifier_loss:
                total_female_classifier_loss += female_loss.item()
                total_male_classifier_loss += male_loss.item()
            else:
                total_female_classifier_loss = 0
                total_male_classifier_loss = 0
            if hp.gender_no_gender_loss:
                total_gender_no_gender_loss += gender_no_gender_loss.item()
            else:
                total_gender_no_gender_loss = 1 # dammy
            if hp.gender_vektor_loss:
                total_gender_vektor_loss += gender_vektor_loss.item()
            else:
                total_gender_vektor_loss = 1 # dammy
            total_loss += loss.item()

            if hp.classifier_loss:
                female_classifier_correct += torch.sum(torch.eq(female_gold, (female_pre > 0.5).float().view(-1))).item()
                male_classifier_correct += torch.sum(torch.eq(male_gold, (male_pre > 0.5).float().view(-1))).item()
            else:
                female_classifier_correct = 0
                male_classifier_correct = 0

            if mode == 'train':
                loss.backward()
                encoder_optim.step()
                if hp.classifier_loss:
                    female_classifier_optim.step()
                    male_classifier_optim.step()
                decoder_optim.step()

        return total_decoder_loss / data_size, \
               total_female_classifier_loss / data_size, \
               total_male_classifier_loss / data_size, \
               total_gender_stereotype_loss / data_size, \
               total_gender_no_gender_loss / data_size, \
               total_gender_vektor_loss / data_size, \
               total_loss / data_size, \
               female_classifier_correct / data_size, \
               male_classifier_correct / data_size

    def calculate_gender_vektor(encoder, word2emb, gender_pairs):
        encoder.eval()
        encoder.zero_grad()
        females = []
        males = []
        for female, male in gender_pairs:
            females += [female]
            males += [male]
        female_embs = torch.stack([encoder(torch.FloatTensor(word2emb[word]).cuda()).data if hp.gpu >= 0 else encoder(torch.FloatTensor(emb[word])).data for word in females], dim=0)
        male_embs = torch.stack([encoder(torch.FloatTensor(word2emb[word]).cuda()).data if hp.gpu >= 0 else encoder(torch.FloatTensor(emb[word])).data for word in males], dim=0)

        gender_vektor = torch.sum(male_embs - female_embs, 0) / male_embs.size(0)
        return gender_vektor

    print('Start training')
    best_loss = float('inf')
    decoder_loss_list = []
    female_classifier_loss_list = []
    male_classifier_loss_list = []
    gender_stereotype_loss_list = []
    gender_no_gender_loss_list = []
    gender_vektor_loss_list = []
    total_loss_list = []
    female_classifier_acc_list = []
    male_classifier_acc_list = []
    for epoch in range(1, hp.epochs + 1):
        gender_vektor = calculate_gender_vektor(encoder, word2emb, train_words['female & male'])
        results = run_model(copy.deepcopy(train_words), gender_vektor, mode='train')
        decoder_loss_list += [results[0]]
        female_classifier_loss_list += [results[1]]
        male_classifier_loss_list += [results[2]]
        gender_stereotype_loss_list += [results[3]]
        if hp.gender_no_gender_loss:
            gender_no_gender_loss_list += [results[4]]
        if hp.gender_vektor_loss:
            gender_vektor_loss_list += [results[5]]
        total_loss_list += [results[6]]
        female_classifier_acc_list += [results[7]]
        male_classifier_acc_list += [results[8]]

        results = run_model(dev_words, gender_vektor, mode='eval')
        decoder_loss_list += [results[0]]
        female_classifier_loss_list += [results[1]]
        male_classifier_loss_list += [results[2]]
        gender_stereotype_loss_list += [results[3]]
        if hp.gender_no_gender_loss:
            gender_no_gender_loss_list += [results[4]]
        if hp.gender_vektor_loss:
            gender_vektor_loss_list += [results[5]]
        total_loss = results[6]
        total_loss_list += [total_loss]
        female_classifier_acc_list += [results[7]]
        male_classifier_acc_list += [results[8]]

        if total_loss < best_loss:
            best_epoch = epoch
            best_loss = total_loss
            encoder_state_dict = encoder.state_dict()
            checkpoint = {
                'encoder': encoder_state_dict,
                'hp': hp,
                'encoder_optim': encoder_optim
            }
            torch.save(checkpoint, '{}model_checkpoint'.format(hp.save_model))

    checkpoint = torch.load('{}model_checkpoint'.format(hp.save_model))
    torch.save(checkpoint,
               '{}best_model.pt'.format(hp.save_model, best_loss, best_epoch))
    os.remove('{}model_checkpoint'.format(hp.save_model))
    os.remove('{}autoencoder.pt'.format(hp.save_model))
    os.remove('{}female.pt'.format(hp.save_model))
    os.remove('{}male.pt'.format(hp.save_model))

def remove_words_not_in_word2emb(emb, words):
    return [word for word in words if word in emb]

def remove_pairs_not_in_word2emb(emb, pairs):
    return [word1 for word1, word2 in pairs if word1 in emb and word2 in emb], \
           [word2 for word1, word2 in pairs if word1 in emb and word2 in emb]


def make_no_gender_words(f, emb):
    return remove_words_not_in_word2emb(emb,
                                        [l.strip() for l in f])

def make_pair_words(f, emb):
    df = pd.read_csv(f, sep='\t')
    female_words = df.iloc[:,0].values.tolist()
    male_words = df.iloc[:,1].values.tolist()

    if len(female_words) == len(male_words):
        return remove_pairs_not_in_word2emb(emb, [[word1, word2] for word1, word2 in zip(female_words, male_words)])
    elif len(female_words) != len(male_words):
        return remove_words_not_in_word2emb(emb, female_words), remove_words_not_in_word2emb(emb, male_words)

def make_optim(model, optimizer, learning_rate, lr_decay, max_grad_norm):
    model_optim = optim.Optim(optimizer, learning_rate, lr_decay, max_grad_norm)
    model_optim.set_parameters(model.parameters())

    return model_optim

def main():
    print('Loading word embedding')
    emb = KeyedVectors.load_word2vec_format(hp.word_embedding,
                                        binary=hp.emb_binary)

    print("Loading data")
    stereotype_words = {}
    gender_words = {}
    no_gender_words = make_no_gender_words(open(hp.no_gender_words), emb)
    stereotype_words['female'], stereotype_words['male'] = \
              make_pair_words(hp.stereotype_words, emb)
    gender_words['female'], gender_words['male'] = \
              make_pair_words(hp.gender_words, emb)
    all_words = no_gender_words \
              + stereotype_words['female'] \
              + stereotype_words['male'] \
              + gender_words['female'] \
              + gender_words['male']

    train_words, dev_words = create_train_dev(gender_words, no_gender_words, stereotype_words)

    word2emb = {}
    for word in all_words:
        word2emb[word] = emb[word]

    if hp.pre_train_autoencoder:
        print('Pre-training autoencoder')
        encoder = model.Encoder(hp.emb_size, hp.hidden_size, hp.pta_dropout_rate)
        decoder = model.Decoder(hp.hidden_size, hp.emb_size, hp.pta_dropout_rate)
        if hp.gpu >= 0:
            encoder.cuda()
            decoder.cuda()
        encoder_optim = make_optim(encoder,
                                     hp.pta_optimizer,
                                     hp.pta_learning_rate,
                                     hp.pta_lr_decay,
                                     hp.pta_max_grad_norm)
        decoder_optim = make_optim(decoder,
                                            hp.pta_optimizer,
                                            hp.pta_learning_rate,
                                            hp.pta_lr_decay,
                                            hp.pta_max_grad_norm)
        if hp.pre_data == 'random':
            checkpoint = pre_train_autoencoder(hp,
                                          encoder,
                                          encoder_optim,
                                          decoder,
                                          decoder_optim,
                                          emb)
        elif hp.pre_data == 'common':
            checkpoint = pre_train_autoencoder(hp,
                                          encoder,
                                          encoder_optim,
                                          decoder,
                                          decoder_optim,
                                          emb,
                                          dev_words=dev_words)

    encoder = model.Encoder(hp.emb_size, hp.hidden_size, hp.dropout_rate)
    decoder = model.Decoder(hp.hidden_size, hp.emb_size, hp.dropout_rate)
    if hp.gpu >= 0:
        encoder.cuda()
        decoder.cuda()
    if hp.pre_train_autoencoder:
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])

    if hp.pre_train_classifier:
        female_classifier = model.Classifier(hp.hidden_size)
        male_classifier = model.Classifier(hp.hidden_size)
        if hp.gpu >= 0:
            female_classifier.cuda()
            male_classifier.cuda()
        female_classifier_optim = make_optim(female_classifier,
                                     hp.cls_optimizer,
                                     hp.cls_learning_rate,
                                     hp.cls_lr_decay,
                                     hp.cls_max_grad_norm)
        male_classifier_optim = make_optim(male_classifier,
                                            hp.cls_optimizer,
                                            hp.cls_learning_rate,
                                            hp.cls_lr_decay,
                                            hp.cls_max_grad_norm)

        encoder.eval()
        encoder.zero_grad()

        train_females = []
        train_males = []
        dev_females = []
        dev_males = []

        train_female_embs = [encoder(torch.FloatTensor(emb[word[0]]).cuda()).data if hp.gpu >= 0 else encoder(torch.FloatTensor(emb[word[0]])).data for word in train_words['female & male']]
        encoder.zero_grad()
        train_male_embs = [encoder(torch.FloatTensor(emb[word[1]]).cuda()).data if hp.gpu >= 0 else encoder(torch.FloatTensor(emb[word[1]])).data for word in train_words['female & male']]
        encoder.zero_grad()
        train_stereotype_embs = [encoder(torch.FloatTensor(emb[word]).cuda()).data if hp.gpu >= 0 else encoder(torch.FloatTensor(emb[word])).data for word in train_words['no gender']]
        encoder.zero_grad()

        dev_female_embs = [encoder(torch.FloatTensor(emb[word[0]]).cuda()).data if hp.gpu >= 0 else encoder(torch.FloatTensor(emb[word[0]])).data for word in dev_words['female & male']]
        encoder.zero_grad()
        dev_male_embs = [encoder(torch.FloatTensor(emb[word[1]]).cuda()).data if hp.gpu >= 0 else encoder(torch.FloatTensor(emb[word[1]])).data for word in dev_words['female & male']]
        encoder.zero_grad()
        dev_stereotype_embs = [encoder(torch.FloatTensor(emb[word]).cuda()).data if hp.gpu >= 0 else encoder(torch.FloatTensor(emb[word])).data for word in dev_words['no gender']]
        encoder.zero_grad()

        print('Pre-training classifier')
        female_checkpoint, male_checkpoint = pre_train_classifier(hp,
                                            female_classifier,
                                            male_classifier,
                                            female_classifier_optim,
                                            male_classifier_optim,
                                            train_female_embs,
                                            train_male_embs,
                                            train_stereotype_embs,
                                            dev_female_embs,
                                            dev_male_embs,
                                            dev_stereotype_embs)

    print('Building female & male classifiers')
    female_classifier = model.Classifier(hp.hidden_size)
    male_classifier = model.Classifier(hp.hidden_size)
    if hp.gpu >= 0:
        female_classifier.cuda()
        male_classifier.cuda()
    if hp.pre_train_classifier:
        female_classifier.load_state_dict(female_checkpoint['female'])
        male_classifier.load_state_dict(male_checkpoint['male'])

    print('Setting optimizer')
    encoder_optim = make_optim(encoder,
                                 hp.optimizer,
                                 hp.learning_rate,
                                 hp.lr_decay,
                                 hp.max_grad_norm)
    female_classifier_optim = make_optim(female_classifier,
                                        hp.optimizer,
                                        hp.learning_rate,
                                        hp.lr_decay,
                                        hp.max_grad_norm)
    male_classifier_optim = make_optim(male_classifier,
                                        hp.optimizer,
                                        hp.learning_rate,
                                        hp.lr_decay,
                                        hp.max_grad_norm)
    decoder_optim = make_optim(decoder,
                                        hp.optimizer,
                                        hp.learning_rate,
                                        hp.lr_decay,
                                        hp.max_grad_norm)

    trainModel(encoder, encoder_optim,
               female_classifier, female_classifier_optim,
               male_classifier, male_classifier_optim,
               decoder, decoder_optim,
               train_words, dev_words, word2emb)


if __name__ == "__main__":
    main()
