from typing import List, Union
import numpy


def format_float(x: Union[int, float, numpy.ndarray]) -> str:
    assert type(x) in [int, float, numpy.ndarray]
    if type(x) == numpy.ndarray:
        assert x.size == 1 and x.dtype in [numpy.float, numpy.int]
        x = x.item()
    return f"{x:.3f}"


def format_percentage(x: Union[int, float, numpy.ndarray]) -> str:
    assert type(x) in [int, float, numpy.ndarray]
    if type(x) == numpy.ndarray:
        assert x.size == 1 and x.dtype in [numpy.float, numpy.int]
        x = x.item()
    return f"{x*100:.1f}\%"


def apply_bold(x: str) -> str:
    return f"\\textbf{{{x}}}"


INDENTATION = ' ' * 4


def compile_tabular(
    data: List[List[str]], row_names: List[str], col_names: List[str],
) -> str:
    # input checks
    assert type(data) == list, f"{type(data)=}"
    assert all([type(row) == list for row in data])
    assert all([all([type(d) == str for d in row]) for row in data])
    assert type(row_names) == list, f"{type(row_names)=}"
    assert all([type(n) == str for n in row_names])
    assert type(col_names) == list, f"{type(col_names)=}"
    assert all([type(n) == str for n in col_names])
    assert len(data) == len(row_names), f"{len(data)=}, {len(row_names)=}"
    assert len(data[0]) == len(col_names), f"{len(data[0])=}, {len(col_names)=}"
    # initialize
    latex: str = ""
    # begin tabular
    latex += f"\\begin{{tabular}}{{c|{'c'*len(col_names)}}}" + '\n'
    # add heading
    latex += INDENTATION + "\\hline" + '\n'
    latex += INDENTATION + " & " + " & ".join(col_names) + " \\\\" + '\n'
    latex += INDENTATION + "\\hline" + '\n'
    # add rows
    lines = [
        INDENTATION + row_names[i] + " & " + " & ".join(data[i]) + " \\\\" + '\n'
        for i in range(len(row_names))
    ]
    latex += "".join(lines)
    latex += INDENTATION + "\\hline" + '\n'
    # end tabular
    latex += "\\end{tabular}" + '\n'
    return latex
