import click

from project.models.pytorch import perform_training


@click.command()
@click.option("-e", '--epochs', type=int, default=3)
@click.option("-b", '--batch_size', type=int, default=64)
@click.option("-v", '--verbose', type=bool, required=False, is_flag=True)
def cli(epochs: int, batch_size: int, verbose: bool = False):
    print("Verbose:", verbose)
    perform_training(epochs, batch_size)
