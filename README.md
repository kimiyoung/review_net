# Review Network for Caption Generation

## Image Captioning on MSCOCO

You can use the code in this repo to genearte a MSCOCO evaluation server submission with CIDEr=0.96+ with just a few hours.

No fine-tuning required. No fancy tricks. Just train three end-to-end review networks and do an ensemble.

+ Feature extraction: 2 hours in parallel
+ Single model training: 6 hours
+ Ensemble model training: 30 mins
+ Beam search for caption generation: 3 hours in parallel

Below is a comparison with other state-of-the-art systems (with according published papers) on the MSCOCO evaluation server:

| Model | BLEU-4 | METEOR | ROUGE-L | CIDEr | Fine-tuned | Task specific features |
|----|----|----|----|----|----|----|
| Attention | 0.537 | 0.322 | 0.654 | 0.893 | No | No |
| MS Research | 0.567 | 0.331 | 0.662 | 0.925 | No | Yes |
| Google NIC | 0.587 | 0.346 | 0.682 | 0.946 | Yes | No |
| Semantic Attention | **0.599** | 0.335 | 0.682 | 0.958 | No | Yes |
| Review Net | 0.597 | **0.347** | **0.686** | **0.969** | No | No |

In the diretcory `image_caption_online`, you can use the code therein to reproduce our evaluation server results.

In the directory `image_caption_offline`, you can rerun experiments in our paper using offline evaluation.

## Code Captioning

Predicting comments for a piece of source code is another interesting task.
In the repo we also release a dataset with train/dev/test splits, along with the code of a review network.

Check out the directory `code_caption`.


## References

This repo contains the code and data used in the following paper:

[Review Network for Caption Generation](https://arxiv.org/abs/1605.07912)

Zhilin Yang, Ye Yuan, Yuexin Wu, Ruslan Salakhutdinov, William W. Cohen

NIPS 2016


