from __future__ import annotations

import csv
import io
import json
import re


def to_tsv(header: list[str], rows: list[list[str]]) -> str:
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(row))
    return "\n".join(lines)


def to_markdown(header: list[str], rows: list[list[str]]) -> str:
    n_cols = len(header)
    widths = [
        max(len(header[i]), max((len(row[i]) for row in rows if i < len(row)), default=1), 3)
        for i in range(n_cols)
    ]

    def fmt(cells: list[str]) -> str:
        return "| " + " | ".join(
            (cells[i] if i < len(cells) else "").ljust(widths[i]) for i in range(n_cols)
        ) + " |"

    sep = "| " + " | ".join("-" * widths[i] for i in range(n_cols)) + " |"
    return "\n".join([fmt(header), sep] + [fmt(row) for row in rows])


def to_prose(header: list[str], rows: list[list[str]]) -> str:
    parts = []
    for i, row in enumerate(rows, 1):
        cells = ", ".join(f"{h} is {v}" for h, v in zip(header, row))
        parts.append(f"Row {i}: {cells}.")
    return " ".join(parts)


def to_csv(header: list[str], rows: list[list[str]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue().rstrip("\r\n")


def to_csv_quoted(header: list[str], rows: list[list[str]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_ALL)
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue().rstrip("\r\n")


def to_row_object(header: list[str], rows: list[list[str]]) -> str:
    lines = []
    for i, row in enumerate(rows, 1):
        parts = [f"Row {i}"] + [f"{h}={v}" for h, v in zip(header, row)]
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def to_col_sep(header: list[str], rows: list[list[str]]) -> str:
    sep = " <COL> "
    lines = [sep.join(header)]
    for row in rows:
        lines.append(sep.join(row))
    return "\n".join(lines)


def to_sec_sep(header: list[str], rows: list[list[str]]) -> str:
    sep = " § "
    lines = [sep.join(header)]
    for row in rows:
        lines.append(sep.join(row))
    return "\n".join(lines)


def to_json(header: list[str], rows: list[list[str]]) -> str:
    data = [dict(zip(header, row)) for row in rows]
    return json.dumps(data, indent=2, ensure_ascii=False)


def to_html(header: list[str], rows: list[list[str]]) -> str:
    lines = ["<table>", "  <tr>"]
    lines += [f"    <th>{value}</th>" for value in header]
    lines += ["  </tr>"]
    for row in rows:
        lines += ["  <tr>"]
        lines += [f"    <td>{value}</td>" for value in row]
        lines += ["  </tr>"]
    lines += ["</table>"]
    return "\n".join(lines)


def to_yaml(header: list[str], rows: list[list[str]]) -> str:
    lines = []
    for row in rows:
        first = True
        for h, v in zip(header, row):
            prefix = "- " if first else "  "
            lines.append(f"{prefix}{h}: {v}")
            first = False
    return "\n".join(lines)


def to_xml(header: list[str], rows: list[list[str]]) -> str:
    def tagify(value: str) -> str:
        tag = re.sub(r"[^\w]", "_", value).strip("_")
        return tag or "col"

    lines = ["<table>"]
    for i, row in enumerate(rows, 1):
        lines.append(f"  <row id=\"{i}\">")
        for h, v in zip(header, row):
            tag = tagify(h)
            lines.append(f"    <{tag}>{v}</{tag}>")
        lines.append("  </row>")
    lines.append("</table>")
    return "\n".join(lines)


FORMATTERS: dict[str, callable] = {
    "tsv": to_tsv,
    "md": to_markdown,
    "prose": to_prose,
    "csv": to_csv,
    "csv_q": to_csv_quoted,
    "row_obj": to_row_object,
    "col_sep": to_col_sep,
    "sec_sep": to_sec_sep,
    "json": to_json,
    "html": to_html,
    "yaml": to_yaml,
    "xml": to_xml,
}
