import random
import torch


torch.manual_seed(0)
random.seed(0)

def pre_train_autoencoder(hp,
                          encoder,
                          encoder_optim,
                          decoder,
                          decoder_optim,
                          embs,
                          dev_words=None):

    if dev_words == None:
        emb_list = [embs[word] for word in embs.wv.vocab]
        random.shuffle(emb_list)
        eval_inputs = torch.split(torch.FloatTensor(emb_list[:hp.pta_dev_num]), hp.pta_batch_size)
        train_inputs = torch.split(torch.FloatTensor(emb_list[hp.pta_dev_num:]), hp.pta_batch_size)
    else:
        dev_words = [w for w in dev_words['no gender']] \
                  + [w[0] for w in dev_words['female & male']] \
                  + [w[1] for w in dev_words['female & male']] \
                  + [w for w in dev_words['stereotype']]
        eval_inputs = torch.split(torch.FloatTensor([embs[word] for word in dev_words]), hp.pta_batch_size)
        train_inputs = torch.split(torch.FloatTensor([embs[word] for word in embs.wv.vocab if word not in dev_words]), hp.pta_batch_size)
    decoder_criterion = torch.nn.MSELoss()

    def run_model(inputs, mode):
        if mode == 'train':
            encoder.train()
            decoder.train()
            perm = torch.randperm(len(inputs))
            inputs = [inputs[idx] for idx in perm]
        elif mode == 'eval':
            encoder.eval()
            decoder.eval()
        total_num = 0
        total_loss = 0
        for input in inputs:
            if hp.gpu >= 0:
                input = input.cuda()
            encoder.zero_grad()
            decoder.zero_grad()
            hidden = encoder(input)
            pre = decoder(hidden)
            loss = decoder_criterion(pre, input)
            if mode == 'train':
                loss.backward()
                encoder_optim.step()
                decoder_optim.step()
            total_loss += loss.item()
            total_num += len(input)

        return total_loss / total_num


    min_loss = float('inf')
    for epoch in range(1, hp.pta_epochs):
        train_loss = run_model(train_inputs, 'train')
        eval_loss = run_model(eval_inputs, 'eval')

        if eval_loss < min_loss:
            min_epoch = epoch
            min_loss = eval_loss
            encoder_state_dict = encoder.state_dict()
            decoder_state_dict = decoder.state_dict()
            checkpoint = {
                'encoder': encoder_state_dict,
                'decoder': decoder_state_dict,
                'hp': hp
            }
            torch.save(checkpoint,
                '{}autoencoder_checkpoint'.format(hp.save_model))

    checkpoint = torch.load('{}autoencoder_checkpoint'.format(hp.save_model))
    torch.save(checkpoint, '{}autoencoder.pt'.format(hp.save_model))

    import os
    os.remove('{}autoencoder_checkpoint'.format(hp.save_model))

    return checkpoint
