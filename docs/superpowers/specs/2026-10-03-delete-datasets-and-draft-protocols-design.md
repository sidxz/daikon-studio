# Delete datasets and draft protocols

**Date:** 2026-10-03
**Status:** awaiting review

## Goal

A workspace admin, or the person who created it, can permanently delete a dataset or a
draft protocol from its detail page, and the storage it used is freed.

This reverses a rule the code states today: "There is no PATCH and no DELETE. A Dataset
is immutable and cited by id." Immutability still holds while a dataset exists: nothing
edits one. What changes is that an unused dataset can be removed. The citation argument
is why a dataset that something still depends on cannot be.

## Decisions

Made by the user on 2026-10-03:

1. **Permanent.** Delete removes the database rows and the stored files. It cannot be undone.
2. **A dataset is refused while any protocol was trained on it.** The confirm dialog lists
   those protocols; the draft ones are deleted first. A dataset used by a published
   protocol can never be deleted, because every prediction run of that protocol reads the
   dataset's training compounds to measure applicability domain.
3. **Deleting a draft protocol also deletes the training run that produced it.** The run's
   only output is the protocol; keeping it would leave a run page with no scorecard or map.

Defaults set while writing this spec (not asked):

- **Who may delete:** the workspace roles admin and owner, for anything; or the creator,
  while they still hold editor or higher. Viewers never.
- **The creator is recorded from now on**, as a nullable `created_by` on datasets and
  protocols. Existing protocols are backfilled from their training run's `requested_by`.
  Datasets created before this change have no recorded creator, so only admins can delete them.
- **Published protocols cannot be deleted.** Out of scope.
- **A dataset is also refused while a training run on it is pending or running.** Failed or
  cancelled training runs on it have no protocol, and are deleted with it.
- **A draft protocol is refused while its training run is still running.** The protocol row
  exists before the run finishes: the run's last phase, "Mapping chemical space", happens
  after it.
- **Delete buttons appear on the detail pages only**, not in lists.

## What a draft protocol and a dataset own

Checked against the code on 2026-10-03.

| Item | Rows | Files |
|---|---|---|
| Draft protocol | `protocols` row; every `runs` row with its `protocol_id` (all training runs: a prediction needs a published protocol) | `{ws}/protocols/{id}/`: `artifact/model.joblib`, `scorecard-inputs.json`, `chemical-space.parquet`, `chemical-space.json` |
| Dataset | `datasets` row; training runs whose `params.dataset_id` is this dataset and whose status is failed or cancelled | `{ws}/datasets/{id}/`: `snapshot.parquet`, `profile.json` |

Nothing else points at either. Collections derive from prediction runs, which need a
published protocol. A draft is never a version parent (`new_version` requires a published
protocol), so the `parent_protocol_id` foreign key never blocks a draft's deletion. A sweep
is only the runs that share a `sweep_id`, so a deleted run simply leaves its sweep.

## API

`DELETE /api/v1/datasets/{id}` answers 204, or:

- 404 if it is not in this workspace.
- 403 if the caller is neither an admin nor its creator.
- 409 "Protocols trained on this dataset must be deleted first. A dataset used by a published
  protocol cannot be deleted."
- 409 "A training run on this dataset is still in progress."

`DELETE /api/v1/protocols/{id}` answers 204, or 404 or 403 as above, or:

- 409 "Published protocols cannot be deleted."
- 409 "This protocol's training run is still finishing. Try again when it completes."

Both responses gain `can_delete: bool`: whether this viewer may delete the item, by role
and creator, and for a protocol also by being a draft. It never reflects dependents; those
are checked when the delete is requested, and shown in the dialog.

`GET /api/v1/protocols` gains an optional `dataset_id` filter, which the dataset dialog
uses to list the protocols trained on it.

**Order:** rows are deleted first, then the file folder. A failed file delete is logged and
leaves an orphan folder, never a row that points at a missing file.

**Accepted races:**

- A training run started between the check and the delete fails when it reads the missing
  snapshot. The plan checks that this failure reads clearly on the run page.
- A profile computation in progress when its dataset is deleted checks that the snapshot
  still exists before saving, so it does not recreate the folder.

## Interface

Dataset detail: a **Delete** button beside "Train a protocol" when `can_delete`. Its dialog:

- Title: Delete "{name}"?
- Body: This permanently deletes the dataset's frozen snapshot and profile. This cannot be undone.
- When protocols were trained on it: each is listed with its version, status and a link, under
  "Protocols trained on this dataset must be deleted first. A dataset used by a published
  protocol cannot be deleted." The Delete button is disabled.
- On success: a "Dataset deleted" toast, then the datasets list.

Protocol detail, drafts only, when `can_delete`: a **Delete** button. Its dialog:

- Title: Delete draft protocol "{name}"?
- Body: This permanently deletes the trained model, its scorecard and chemical-space map, and
  the training run that produced it. This cannot be undone.
- On success: a "Protocol deleted" toast, then the protocols list.

A 409 from the server is shown inside the dialog, which stays open.

## Migration

`011`: add `created_by UUID NULL` to `datasets` and `protocols`. Backfill
`protocols.created_by` from the earliest training run with that `protocol_id`.

## Testing

- **Permissions:** an editor deletes their own; another editor gets 403; an admin deletes
  anyone's; a viewer gets 403; another workspace gets 404; a pre-migration dataset is
  admin-only. `can_delete` agrees with each case.
- **Dataset:** 409 with a protocol on it; 409 with a pending training run; success removes
  the row, its failed training runs and its folder; the same file can be uploaded again.
- **Protocol:** 409 when published; 409 while its training run is running; success removes
  the row, its runs and its folder, after which its dataset can be deleted.
- **Migration:** the backfill sets each protocol's creator from its training run.
- **Frontend:** the dialog with dependents disables Delete; success navigates to the list; a
  409 is shown in the dialog.
- **Live:** in dev, train a draft protocol on a small throwaway dataset, then delete the
  protocol and the dataset through the UI. Uploading and deleting are asked for at the time.
