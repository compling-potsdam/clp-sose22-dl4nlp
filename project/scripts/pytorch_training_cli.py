import click

from project.models.fashion import perform_training


@click.command()
@click.option("-d", '--epochs', type=str, default="/Users/philippsadler/Opts/Git/clp-sose22-dl4nlp/data")
@click.option("-l", '--ckpts_dir', type=str, default="/Users/philippsadler/Opts/Git/clp-sose22-dl4nlp/logs")
@click.option("-e", '--epochs', type=int, default=3)
@click.option("-b", '--batch_size', type=int, default=64)
@click.option("-v", '--verbose', type=bool, required=False, is_flag=True)
def cli(data_dir: str, ckpts_dir: str, epochs: int, batch_size: int, verbose: bool = False):
    print("Data dir:", data_dir)
    print("Ckpts dir:", ckpts_dir)
    print("Verbose:", verbose)
    perform_training(data_dir, ckpts_dir, epochs, batch_size)
