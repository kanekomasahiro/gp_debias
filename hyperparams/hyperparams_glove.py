
class Hyperparams:
    gender_words = 'wordlist/gender_pair.tsv'
    stereotype_words = 'wordlist/stereotype_list.tsv'
    no_gender_words = 'wordlist/no_gender_list.tsv'

    save_model = 'src/glove_model/'
    eval_model = 'src/glove_model/best_model.pt'

    word_embedding = './embeddings/glove.txt'
    # pre-train autoencoder
    pre_train_autoencoder = True
    pre_data = 'random' # random or common
    pta_batch_size = 512
    pta_learning_rate = 0.0002
    pta_optimizer = 'adam'
    pta_dev_num = 5000
    pta_lr_decay = 1
    pta_dropout_rate = 0.05
    pta_max_grad_norm = None
    pta_epochs = 315

    # pre-train classifier
    pre_train_classifier = True
    cls_learning_rate = 0.0002
    cls_optimizer = 'adam'
    cls_lr_decay = 1
    cls_max_grad_norm = None
    cls_epochs = 15

    dev_num = 20
    sampling = 'under_sampling' # over_sampling or under_sampling

    emb_binary = False
    batch_size = 1
    learning_rate = 0.1
    optimizer = 'sgd'
    lr_decay = 1
    dropout_rate = 0.01
    autoencoder = True
    gender_no_gender_loss = True
    classifier_loss = True
    gender_vektor_loss = False
    max_grad_norm = None

    decoder_loss_rate = 1.0
    female_loss_rate = 0.0001
    male_loss_rate = 0.0001
    gender_stereotype_loss_rate = 0.0001
    gender_no_gender_loss_rate = 0.0001
    gender_vektor_loss_rate = 0.0001

    emb_size = 300
    hidden_size = 300

    epochs = 5
    seed = 0
    gpu = 1
