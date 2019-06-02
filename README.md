# Gender-preserving Debiasing for Pre-trained Word Embeddings

[Masahiro Kaneko](https://sites.google.com/view/masahirokaneko/english?authuser=0), [Danushka Bollegala](http://danushka.net/)

Code and debiased word embeddings for the paper: "Gender-preserving Debiasing for Pre-trained Word Embeddings" (In ACL 2019). If you use any part of this work, make sure you include the following citation:

```
@inproceedings{Kaneko:ACL:2019,    
    title={Gender-preserving Debiasing for Pre-trained Word Embeddings},    
    author={Masahiro Kaneko and Danushka Bollegala},    
    booktitle={Proc. of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)},    
    year={2019} 
}
```

### Our experiment settings
- python==3.7.2
- torch==1.1.0
- gensim==3.7.1


### How to train yourself

First download the [necessary data](https://github.com/uclanlp/gn_glove) listed below using following command.
- Trained glove and gn-glove
- SemBias dataset
- Female and male word lists
```
./download.sh
```
Perform debiasing for word embeddings and its evaluation. Debiased word embeddings are stored in `debiased_embeddings` in both bin and txt format. Name of word embeddings to be debiased as `glove` or `gn` and give as the first argument. For example,
```
./run.sh gn
```
You can also evaluate your word embaddings without training on SemBias:
```
eval_word_embeddings.py -i path/to/your/embeddings
```

### Our debiased word embeddings

You can directly download our debiased [GP (glove)](https://drive.google.com/file/d/12VK2-BpLAg_-VPVl_wcLBZbzd9wcwyqN/view?usp=sharing) and [GP  (gn-glove)](https://drive.google.com/file/d/1Rn--1pxjBhyp5os7zw75VB-YQUHXcfgF/view?usp=sharing).

### License
See the LICENSE file.
