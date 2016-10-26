
## Download Data

You can download the data by
```
wget http://kimi.ml.cmu.edu/code_data/train.dat
wget http://kimi.ml.cmu.edu/code_data/dev.dat
wget http://kimi.ml.cmu.edu/code_data/test.dat
```
where `train.dat`, `dev.dat`, and `test.dat` corresponds to training, development, and test sets respectively. Each data file is in the following format:
```
<tokenized_code_separated_by_space>\t<tokenized_comment_separated_by_space>
```

## Training

We can train the model with the following command

```
th main.lua
```

The best model on the dev set will be saved.

## Evaluation

In order to evaluate our model, we need to compute the CS-k metrics. For better performance, we need to do precomputation for string prefixes.
```
th comp_prefix.lua
```

Then we can do evaluation,
```
th eval.lua
```

The performance on the test set will be printed to stdout.
