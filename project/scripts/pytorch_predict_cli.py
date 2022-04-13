import click

from project.models.pytorch import perform_prediction


@click.command()
@click.option("-v", '--verbose', type=bool, required=False, is_flag=True)
def cli(verbose: bool = False):
    print("Verbose:", verbose)
    perform_prediction()
