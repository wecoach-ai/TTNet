import click


@click.group()
def cli() -> None:
    click.echo("Hello, World!!")
