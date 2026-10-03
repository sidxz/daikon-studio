# Dataset identifier column

**Date:** 2026-10-03
**Status:** awaiting review

## Goal

A scientist can name which column of their upload holds compound IDs, when creating a
dataset or at any time afterwards, and then sees each training compound's own ID wherever
the app shows one: the Compounds tab, the scorecard's largest prediction errors, and the
chemical-space map. On a large dataset they can find a compound by its ID.

## What exists today

- A frozen snapshot already keeps every column of the upload. The 400k dataset's snapshot
  still holds `NATHAN ID` and `RU ID`; BBBP's holds `num` and `name`. Nothing records which
  column is the identifier, and nothing reads these columns.
- Prediction runs already have this: an optional identifier column, guessed by
  `guessIdColumn` in the run wizard, carried into results as `compound_id`.
- A snapshot's structures are canonical and unique (duplicates are merged at upload), and
  the scorecard's and the map's structure strings are the snapshot's own. A structure is
  therefore a key from any of those surfaces back to its snapshot row.

## Decisions

Made by the user on 2026-10-03:

1. **Any editor may set or change the identifier column.** It is display metadata: no
   frozen data, score or content hash changes, and it can be changed back.
2. **The Compounds tab gets a search box** that finds compounds by ID.

Defaults set while writing this spec (not asked):

- **IDs are looked up when a page is read, never copied** into the scorecard inputs or
  the map files. Setting or changing the column takes effect everywhere at once, with no
  retraining and no map rebuild, and works for every existing dataset and protocol.
- **Allowed columns:** any snapshot column except the structure column, the target column
  and `split`. An ID value is shown as text; a blank value shows as no ID.
- **Merged duplicates are out of scope.** A compound merged from several rows keeps the
  first row's ID, as today.
- **Search** is a case-insensitive "contains" match on the ID text, and is offered only
  when the dataset has an identifier column.

## Data

- `datasets.id_column`: nullable `VARCHAR(128)`, migration `012`. `Dataset.id_column:
  str | None`. Not part of `content_hash`: two uploads of the same data and split are still
  the same dataset whatever their ID column.
- Creating a dataset accepts an optional `id_column`, validated against the uploaded file's
  columns with the rule above.

## API

- `POST /api/v1/datasets`: optional `id_column`. A column not in the file, or a reserved
  one, is 422 "Choose an identifier column other than the structure, target or split
  column." or "Column '{name}' is not in the uploaded file."
- `PUT /api/v1/datasets/{id}/id-column`, body `{"id_column": "<name>" | null}`. Editor or
  higher; `null` clears it. Validated against the snapshot's columns, with the same
  messages. Answers the updated `DatasetResponse`. The routes module's "no PATCH" rule
  stays true for everything frozen; this is the one mutable setting, and its docstring
  says so.
- `GET /api/v1/datasets/{id}/columns`: the snapshot columns eligible as an identifier,
  for the picker on the dataset page. Reads only the Parquet schema.
- `DatasetResponse` gains `id_column`.
- `GET /api/v1/datasets/{id}/compounds`: each row gains `compound_id`; a new `q`
  parameter filters to IDs containing it (case-insensitive). `q` on a dataset with no
  identifier column is 422 "This dataset has no identifier column."
- Scorecard `worst_rows[]` gain `compound_id`.
- Protocol map compound lookups (`/protocols/{id}/chemical-space/compounds`) gain
  `compound_id`; the run map's training points use the same lookup.

One application helper reads only the structure and ID columns from the snapshot and
returns a structure → ID map; the scorecard and map lookups use it, off the event loop.

## Interface

- **Dataset wizard, Columns step:** an "Identifier (optional)" select beside Structures
  and Value to predict, preselected by `guessIdColumn` (moved to `shared/lib` so both
  wizards use one copy). "None" is an option.
- **Dataset page, Overview:** an "Identifier column" field showing the current column.
  Editors get a select (columns from `/columns`, plus "None") that saves on change;
  viewers see the value only.
- **Compounds tab:** an ID column in the table, and, when the dataset has an identifier
  column, a search box above it ("Search by ID").
- **Largest prediction errors:** each card shows the compound's ID above its structure.
- **Chemical-space map tooltips:** a training compound's tooltip shows its ID, as a run
  compound's already does.

## Testing

- **API:** create with a valid, a missing and a reserved `id_column`; `PUT` by an editor
  (200), a viewer (403), with an invalid column (422), and with `null` (clears); `/columns`
  excludes structure, target and split; compounds carry `compound_id` and `q` filters
  (case-insensitive, contains) and is 422 without an identifier column; scorecard worst
  rows and map compounds carry `compound_id`, and change when the column is changed.
- **Migration:** `012` applies; existing datasets read back with `id_column` null.
- **Frontend:** the wizard preselects the guessed column; the Overview select saves; the
  search box appears only with an identifier column; cards and tooltips show the ID.
- **Live:** in dev, set `NATHAN ID` on the 400k dataset and search for one ID; set `name`
  on BBBP and see IDs in its Compounds tab and on the errors and map of a protocol trained
  on it (several exist).
