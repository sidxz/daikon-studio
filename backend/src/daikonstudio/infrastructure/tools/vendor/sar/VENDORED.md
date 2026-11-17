# Vendored: SAR workbench chemistry

- Source: `sar-workbench` repository, commit `bfb0bb3` (the commit its own `VENDORED.md` names)
- Copied from: the `sar-workbench` working tree at `5c12b06`; the four files last changed in `46bb65c`
- Author: Steve Berthel
- Files: `sar_variable.py`, `sar_core.py`, `sar_edit.py`, `sar_cliffs.py`

Do not edit these files. Daikon's tools wrap them in `impl/cores.py` and `impl/pairs.py`.
To update, copy again from a newer commit and change the commit above.

Deliberately not taken:

- `sar_io.py` and `sar_report.py`: file I/O and HTML rendering. Daikon reads and writes
  through its own codecs and blob store.
- `substituent_names.csv`: its licence is unsettled. Substituent naming uses RDKit's
  built-in abbreviations only.
