import click

from project.models.fashion import perform_prediction


@click.command()
@click.option("-d", '--data_dir', type=str, default="/Users/philippsadler/Opts/Git/clp-sose22-dl4nlp/data")
@click.option("-l", '--ckpts_dir', type=str, default="/Users/philippsadler/Opts/Git/clp-sose22-dl4nlp/logs")
@click.option("-v", '--verbose', type=bool, required=False, is_flag=True)
def cli(data_dir: str, ckpts_dir: str, verbose: bool = False):
    print("Data dir:", data_dir)
    print("Ckpts dir:", ckpts_dir)
    print("Verbose:", verbose)
    perform_prediction(data_dir, ckpts_dir)
