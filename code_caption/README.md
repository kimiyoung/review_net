
## Download Data

We download the source code with comments from a github repo.
```
git clone https://github.com/habeascorpus/habeascorpus-data-withComments.git
```

## Data Preprocessing

Then we run a script to preprocess the dataset and split it into train, dev, and test sets.

```
python code_proc.py
```

Three files, `train.dat`, `dev.dat`, and `test.dat` will be extracted to the current directory.

## Training

We can train the model with the following command

```
th main.lua
```

The best model on the dev set will be saved.

## Evaluation

```
th eval.lua
```

The performance on the test set will be printed to stdout.
