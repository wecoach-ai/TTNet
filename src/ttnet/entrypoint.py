import click


@click.group()
def cli() -> None:
    pass


@cli.command()
def train() -> None:
    click.echo("Hello, World!!")
