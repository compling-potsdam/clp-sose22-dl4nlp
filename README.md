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

You can run the lightning scripts directly.

```
 tensorboard --logdir logs
```

### CLI

Go in the root directory of this project and install using pip

```
pip install . -U
```

Then you can then use the CLI to train:

```
dl4nlp-pytorch-training
```

and to predict

```
dl4nlp-pytorch-predict
```

Note: The model checkpoint is simply saved to the directory where you invoke the CLI.