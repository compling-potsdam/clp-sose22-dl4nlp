# clp-sose22-dl4nlp

Showcase the PyTorch quickstart and project structure

## Setup Conda environment

```
conda env create -n dl4nlp python=3.9
```

Install requirements

```
 pip install -r requirements.txt
```

You can run the lightning scripts directly and have a look at the tensorboard logs.

```
 tensorboard --logdir <path-to-logs>
```

### CLI

Go in the root directory of this project and install using pip

```
pip install . -U
```

Then you can then use the CLI to train:

```
dl4nlp-pytorch-training -d <path-to-data> -l <path-to-logs>
```

and to predict

```
dl4nlp-pytorch-predict -d <path-to-data> -l <path-to-logs>
```

Note: The model checkpoint is saved to the logs directory. You might want to adjust the defaults.
