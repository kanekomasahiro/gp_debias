import random
import torch


torch.manual_seed(0)
random.seed(0)

def save_checkpoint(hp, mode, model):
    model_state_dict = model.state_dict()
    checkpoint = {
        mode: model_state_dict,
        'hp': hp
    }
    torch.save(checkpoint,
        '{}{}_checkpoint'.format(hp.save_model, mode))

def pre_train_classifier(hp,
                         female_classifier,
                         male_classifier,
                         female_optim,
                         male_optim,
                         train_female_embs,
                         train_male_embs,
                         train_stereotype_embs,
                         dev_female_embs,
                         dev_male_embs,
                         dev_stereotype_embs):

    criterion = torch.nn.MSELoss()

    def run_model(female_embs, male_embs, no_gender_embs, mode):
        inputs = female_embs + male_embs + no_gender_embs
        tags = [0 for _ in range(len(female_embs))] + [1 for _ in range(len(male_embs))] + [2 for _ in range(len(no_gender_embs))]

        if mode == 'train':
            female_classifier.train()
            male_classifier.train()
            perm = [idx for idx in range(len(inputs))]
            random.shuffle(perm)
            inputs = [inputs[idx] for idx in perm]
            tags = [tags[idx] for idx in perm]
        elif mode == 'eval':
            female_classifier.eval()
            male_classifier.eval()
        total_female_num = 0
        total_male_num = 0
        total_female_loss = 0
        total_male_loss = 0
        for input, tag in zip(inputs, tags):
            zero_tag = torch.zeros(1)
            one_tag = torch.ones(1)
            if hp.gpu >= 0:
                input = input.cuda()
                zero_tag = zero_tag.cuda()
                one_tag = one_tag.cuda()
            female_classifier.zero_grad()
            male_classifier.zero_grad()
            female_pre = female_classifier(input)
            male_pre = male_classifier(input)
            if tag == 0:
                female_loss = criterion(female_pre, one_tag)
                male_loss = criterion(male_pre, zero_tag)
            elif tag == 1:
                female_loss = criterion(female_pre, zero_tag)
                male_loss = criterion(male_pre, one_tag)
            elif tag == 2:
                female_loss = criterion(female_pre, zero_tag)
                male_loss = criterion(male_pre, zero_tag)
                #continue
            if mode == 'train':
                female_loss.backward()
                male_loss.backward()
                female_optim.step()
                male_optim.step()
            total_female_loss += female_loss.item()
            total_male_loss += male_loss.item()
            total_female_num += len(female_embs)
            total_male_num += len(male_embs)

        return total_female_loss / total_female_num, total_male_loss / total_male_num


    min_female_loss = float('inf')
    min_male_loss = float('inf')
    for epoch in range(1, hp.cls_epochs):
        train_female_loss, train_male_loss = run_model(train_female_embs, train_male_embs, train_stereotype_embs, 'train')
        eval_female_loss, eval_male_loss = run_model(dev_female_embs, dev_male_embs, dev_stereotype_embs, 'eval')

        if eval_female_loss < min_female_loss:
            min_female_epoch = epoch
            min_female_loss = eval_female_loss
            female_state_dict = female_classifier.state_dict()
            female_checkpoint = {
                'female': female_state_dict,
                'hp': hp
            }
            torch.save(female_checkpoint,
                '{}female_checkpoint'.format(hp.save_model))
        if eval_male_loss < min_male_loss:
            min_male_epoch = epoch
            min_male_loss = eval_male_loss
            male_state_dict = male_classifier.state_dict()
            male_checkpoint = {
                'male': male_state_dict,
                'hp': hp
            }
            torch.save(male_checkpoint,
                '{}male_checkpoint'.format(hp.save_model))

    female_checkpoint = torch.load('{}female_checkpoint'.format(hp.save_model))
    torch.save(female_checkpoint, '{}female.pt'.format(hp.save_model))
    male_checkpoint = torch.load('{}male_checkpoint'.format(hp.save_model))
    torch.save(male_checkpoint, '{}male.pt'.format(hp.save_model))

    import os
    os.remove('{}female_checkpoint'.format(hp.save_model))
    os.remove('{}male_checkpoint'.format(hp.save_model))

    return female_checkpoint, male_checkpoint
