import os


def echo_page_break(filepath: str, heading: str):
    sep = '\#' + ' ' + '\#' * 100
    os.system(f"echo >> {filepath}")
    os.system(f"echo {sep} >> {filepath}")
    os.system(f"echo \# {heading} >> {filepath}")
    os.system(f"echo {sep} >> {filepath}")
    os.system(f"echo >> {filepath}")
