
# README for CBART
This repository contains the implementation of the EMNLP 2021 paper: 
"[**Parallel Refinements for Lexically Constrained Text Generation with BART**](https://arxiv.org/abs/2109.12487)".
****
##  Abstract
Lexically constrained text generation aims to control the generated text by incorporating
some pre-specified keywords into the output. 
Previous work injects lexical constraints into 
the output by controlling the decoding process or refining the candidate output iteratively, 
which tends to generate generic or ungrammatical sentences, and has high computational 
complexity. To address these challenges, we 
propose Constrained BART (CBART) for lexically constrained text generation. CBART 
leverages the pre-trained model BART and 
transfers part of the generation burden from 
the decoder to the encoder by decomposing 
this task into two sub-tasks, thereby improving 
the sentence quality. Concretely, we extend 
BART by adding a token-level classifier over 
the encoder, aiming at instructing the decoder 
where to replace and insert. Guided by the encoder, the decoder refines multiple tokens of 
the input in one step by inserting tokens before specific positions and re-predicting tokens 
with low confidence. To further reduce the inference latency, the decoder predicts all tokens 
in parallel. Experiment results on One-BillionWord and Yelp show that CBART can generate 
plausible text with high quality and diversity 
while significantly accelerating inference. 
****
## Requirements
python 3.6  
pip install torch==1.4.0  
pip install transformers==3.0.2  
pip install pympler==0.8 
****
## Dataset
All our experiments are conducted on [One-Billion-Word](http://www.statmt.org/lm-benchmark/) and 
[Yelp review](https://www.yelp.com/dataset) corpora. In this paper, we choose 1M, 0.1M
sentences from each dataset as the training and validation sets (The full data used in this paper are available at https://drive.google.com/drive/folders/1Dj7VX2CjSn3-g7FEYuJrT5_JWGdsAHjE?usp=sharing). 
If you want to train the model from scratch, you should download the corresponding data first 
and put them in the corresponding directory, i.e. data/one-billion-words (data/yelp_review).
Note we only put several sentences in the data/one-billion-words/train.txt and data/one-billion-words/dev.txt. 

****
## Try our model with well-trained model checkpoints 
| Model           |  Download link
|----------------------|--------|
| CBART-base for Yelp review| [\[link\]](https://drive.google.com/file/d/1JPPhqdapW_p2AQ9jyx0MuYeD31gHuQAD/view?usp=sharing)  | 
| CBART-large for Yelp review| [\[link\]](https://drive.google.com/file/d/1tbkF2yAEFJ-wE6iG2nd_iWxzCXfH2boU/view?usp=sharing)  | 
| CBART-base for One-Billion-Word| [\[link\]](https://drive.google.com/file/d/1A6BU_hc3O5ppy89im4g3Z9hXVUkgFqnw/view?usp=sharing)  | 
| CBART-large for One-Billion-Word| [\[link\]](https://drive.google.com/file/d/13NOAsdSnO-eLIDxdo0M-_sX2KxyrYndX/view?usp=sharing)  | 

If you want to try our models, you should download these checkpoints, put them into the 'checkpoints' directory, and decompress them with the following command:
Then you can directly go to [Generate sentences with lexical constraints](#generate).
```bash
tar -xzvf checkpoint_name.tar.gz # replace 'checkpoint_name' with the corresponding checkpoint name.
```
If you want to train our model on another dataset, please refer to the following steps.
****
## Train our model from scratch 
Note the default dataset is One-Billion-Word. You can freely change it to another dataset. 
* Step 1: Create synthetic data to train CBART

```bash
cd utils  
sh create_synthetic_data.sh
```


* Step 2: Train CBART
```bash
cd models
```
If you want to train CBART-base on One-Billion-Word:
```bash
python bart.py --batch_size 80 --gpu 5 --dataset one-billion-words
```

If you want to train CBART-large on One-Billion-Word:
```bash
python bart.py --batch_size 25 --gpu 5 --dataset one-billion-words --bart large
```

## <span id="generate"> Generate sentences with lexical constraints </span>

[comment]: <> (You can find the keywords files used in the paper in the following directories: data/one-billion-words and data/one-billion-words.  )

[comment]: <> (Each directory contains 6 keywords files: 1keywords.txt, 2keywords.txt, 3keywords.txt, 4keywords.txt, 5keywords.txt, and 6keywords.txt, )

[comment]: <> (where the number denotes the number of keywords in each line. )

We show some keywords in "data/one-billion-words/4keywords.txt", 
where each line has 4 keywords. 
In the following, we'll generate sentences with 4 keywords. 
If you want to generate sentences with other number of keywords, 
you should prepare keywords and put them in the "data/dataset_name/{k}keywords.txt", 
where '{k}' denotes the number of keywords in each line. 
If so, you need to change the hyperparameter "num_keywords" 
(e.g., --num_keywords 1, if you want to generate sentence with one keyword).


Generate sentences with 4keywords.txt by running **greedy decoding** on CBART-base:
```bash
python main.py --gpu 7 --num_keywords 4 --do_sample 0 --batch_size 10 --bart base --dataset one-billion-words
```

Generate sentences with 4keywords.txt by running **multiple-sequence decoding (p=0.5, c=5 )** decoding on CBART-base:
```bash
python main.py --gpu 7 --num_keywords 4 --do_sample 1 --top_p 0.5 --decoder_chain 5 --batch_size 10 --bart base --dataset one-billion-words
```
Generate sentences with 4keywords.txt by running **multiple-sequence decoding (k=5, c=5)** decoding on CBART-base:
```bash
python main.py --gpu 7 --num_keywords 4 --do_sample 1 --top_k 5 --decoder_chain 5 --batch_size 10 --bart base --dataset one-billion-words
```


## Citation
If you want to use this code in your research, you can cite our [paper](https://arxiv.org/abs/2109.12487):
```bash

@inproceedings{he2021cbart,
  title={Parallel Refinements for Lexically Constrained Text Generation with BART},
  author={He, Xingwei},
  booktitle={Proceedings of EMNLP},
  year={2021}
}

```

