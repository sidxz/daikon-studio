"""
sar_variable.py — cores with variable ring atoms.

Lets a set of related heterocycles share one core, with an X at each position
that differs and the SAR table reporting which element each compound has there.
Nothing here knows what an azole, an azine or a chalcogen ring is: the search
says only *where* atoms may vary — aromatic ring, any ring, anywhere — and
which elements actually do is read back off the compounds. That is what lets it
work on a scaffold family nobody anticipated, selenophene and phosphole
included, and on chain variation like C=C / C=N / N=N with the ring restriction
off.

Three steps, and the last two matter as much as the first:

  1. Run the MCS under a permissive comparator to get the core's *skeleton*.
  2. Choose how each molecule sits on that skeleton, minimising the number of
     positions that vary (`optimize_alignment`).
  3. Rewrite the core from the elements actually observed (`specialize_query`).

Step 2 exists because a symmetric core matches a molecule more than one way,
and taking whichever match came back first is a coin toss per compound: two
compounds landing on opposite faces smear one real difference across two
positions, and the table claims two X where the chemistry has one. Step 3
exists because the SMARTS a permissive comparator produces describes the
*class* of atoms the search was allowed to interchange — so a core found over
indoles, benzimidazoles and benzothiazoles would offer oxygen as a possibility
for a set containing none. A core should describe the compounds in hand.

How
---
RDKit's `rdFMCS` accepts a custom atom comparator: subclass
`rdFMCS.MCSAtomCompare`, override `__call__`, and assign an instance to
`MCSParameters.AtomTyper`. `VariableAtomCompare` below says "atoms in this
structural scope may match one another whatever their element".

Two things worth knowing about that API, both learned the hard way from the
RDKit docs rather than from a traceback:

  * You cannot call `super().__call__(...)`. The class is Boost-wrapped C++ and
    the Python signatures do not line up, so the comparison has to be written
    out in full. The `CheckAtom*` helpers on `self` *are* callable.

  * `MCSParameters.StoreAll = True` returns every degenerate MCS in
    `MCSResult.degenerateSmartsQueryMolDict` -- which is what "show the chemist
    the possibilities" wants. But when it is set, `queryMol` is `None` and
    `smartsString` is `''`. Read the dict, not the scalar properties.

Variable positions are worked out from the actual substructure matches, not by
parsing the SMARTS. RDKit does emit element lists like `[#7,#8,#16]` when a
permissive comparator fires, but deriving them from the matches is both more
robust and the only way to get the per-compound assignment the table needs.

Design notes
------------
Ranking is by coverage alone, then size as a tiebreak. Deliberately not by
"biggest": a permissive comparator always finds a core at least as large as a
strict one, so size-first ranking would put the vaguest candidate on top every
time. Coverage is the number that says what a core is worth, and specificity is
something a chemist judges from the drawing.

There is no cap on how many positions may vary, because a cap would be the
wrong kind of control: how many X columns a set needs is a property of the
molecules, not a setting.

It is, however, a property of the molecules *and the alignment*, which is why
step 2 above is not an optimisation but a correctness fix. Indole plus
benzimidazole plus benzothiazole looks like it needs two X -- ring position 1
reading N/N/S and position 3 reading C/N/N -- but the bicycle is symmetric, and
matching the thiazole flipped puts all three nitrogens on one atom and leaves a
single X spanning C/N/S. One X is the truth about those compounds; two was an
artefact of match ordering.

The trade that buys is an extra R column: flipping a molecule carries its whole
benzo ring with it, so a substituent lands on the mirror-image carbon and the
decomposition grows a position. Each candidate reports both counts, and the
alignment is chosen on X first and R second.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS, rdqueries

RDLogger.DisableLog("rdApp.warning")

# Atomic numbers, for readability below.
C, N, O, S, P, SE = 6, 7, 8, 16, 15, 34


SCOPES = ("none", "aromatic-ring", "ring", "any")


class VariableAtomCompare(rdFMCS.MCSAtomCompare):
    """Atom comparator that permits swaps within a structural scope.

    `scope` says *where* a swap is allowed, not *which elements* may swap:

      "none"          — elements must match. Plain strict matching.
      "aromatic-ring" — any aromatic ring atom may match any other.
      "ring"          — any ring atom may match any other, aromatic or not.
      "any"           — any atom may match any other. Covers chain variation
                        such as C=C / C=N / N=N.

    Deliberately no element whitelist by default. An earlier version took
    groups like ("N", "O", "S") and ("C", "N"), which are just the azoles and
    azines written out by hand -- so a set built on phosphole, selenophene or
    borazine silently failed, and a set with no oxygen in it was still told
    oxygen was a possibility. Which elements vary is a fact about the input,
    and the input is right there to be read. `interchange` survives only as an
    optional narrowing for when a chemist wants to forbid a swap the data
    would otherwise allow.

    Valence matching must stay off for heterocycle swaps: pyrrole-type N
    carries an H where thiophene S does not, so requiring equal valence would
    reject the very match this class exists to make.
    """

    def __init__(self, scope: str = "aromatic-ring",
                 interchange: Sequence[Sequence[str]] = ()):
        super().__init__()
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
        self.scope = scope
        self.groups: List[frozenset] = [
            frozenset(Chem.GetPeriodicTable().GetAtomicNumber(s) for s in grp)
            for grp in interchange
        ]

    def _in_scope(self, a) -> bool:
        if self.scope == "any":
            return True
        if self.scope == "ring":
            return a.IsInRing()
        if self.scope == "aromatic-ring":
            return a.GetIsAromatic() and a.IsInRing()
        return False

    # -- the swap rule, kept separate so it can be reasoned about alone ----
    def may_swap(self, a1, a2) -> bool:
        z1, z2 = a1.GetAtomicNum(), a2.GetAtomicNum()
        if z1 == z2:
            return True
        if not (self._in_scope(a1) and self._in_scope(a2)):
            return False
        if self.groups:      # optional narrowing; absent by default
            return any(z1 in g and z2 in g for g in self.groups)
        return True

    def __call__(self, p, mol1, atom1, mol2, atom2) -> bool:
        a1 = mol1.GetAtomWithIdx(atom1)
        a2 = mol2.GetAtomWithIdx(atom2)
        if not self.may_swap(a1, a2):
            return False
        # No super() call is possible here; each check is spelled out.
        if p.MatchValences and a1.GetTotalValence() != a2.GetTotalValence():
            return False
        if p.MatchChiralTag and not self.CheckAtomChirality(
                p, mol1, atom1, mol2, atom2):
            return False
        if p.MatchFormalCharge and not self.CheckAtomCharge(
                p, mol1, atom1, mol2, atom2):
            return False
        if p.RingMatchesRingOnly and not self.CheckAtomRingMatch(
                p, mol1, atom1, mol2, atom2):
            return False
        return True


# Named conventions, in order of how much they let through. `label` is what
# the core page shows, so a chemist knows what a candidate assumed. None of
# them names an element: that is the point. Which elements vary is read back
# off the data afterwards and written into the candidate's own query.
PRESETS: Dict[str, dict] = {
    "strict": {
        "label": "elements must match",
        "scope": "none",
        "interchange": (),
    },
    "ring-aromatic": {
        "label": "any aromatic ring atom may vary",
        "scope": "aromatic-ring",
        "interchange": (),
    },
    "ring-any": {
        "label": "any ring atom may vary, aromatic or not",
        "scope": "ring",
        "interchange": (),
    },
    "any": {
        "label": "any atom may vary (covers chain cases such as C=C, C=N, N=N)",
        "scope": "any",
        "interchange": (),
    },
}

# Strict first so the plain answer is always on the table, then the general
# heterocycle case. "ring-any" and "any" are opt-in: they are more permissive
# than most sets need and will happily unify things a chemist would not.
DEFAULT_PRESETS = ("strict", "ring-aromatic")
DEFAULT_THRESHOLDS = (1.0, 0.9, 0.8)

# The old element-group names, kept working. Renaming them turned a permissive
# search into a silent strict-only one: every unknown name was dropped with a
# note, and a caller holding the old list -- a notebook cell not re-run after a
# reload, a saved command line -- got benzo rings and no explanation. Aliases
# cost nothing; a rename that quietly changes results costs a day.
PRESET_ALIASES: Dict[str, str] = {
    "azole": "ring-aromatic",
    "azine": "ring-aromatic",
    "aromatic-any": "ring-aromatic",
    "chain-cn": "any",
    "heteroatom-any": "any",
}


def resolve_preset(name: str) -> str:
    """Map a preset name, old or current, to a key in PRESETS.

    Raises rather than returning a default: a preset nobody recognises means
    the search the caller asked for is not the search they would get.
    """
    if name in PRESETS:
        return name
    if name in PRESET_ALIASES:
        return PRESET_ALIASES[name]
    raise ValueError(
        f"unknown preset {name!r}. Available: {', '.join(sorted(PRESETS))}"
        + f" (aliases: {', '.join(sorted(PRESET_ALIASES))})")


# --------------------------------------------------------------------------
# alignment  --  which way round does each molecule sit on the core?
# --------------------------------------------------------------------------
#
# A core with any symmetry matches a molecule more than one way, and the
# choice is not cosmetic: it decides how many X columns the table needs.
#
# Indole and benzimidazole share a nitrogen at ring position 1 and differ at
# position 3, so one X should do. But the bicycle's mirror symmetry means
# indole matches the core both ways round, and if one compound is taken
# flipped relative to another, that single difference is smeared across two
# positions and the table reports two X where one was the truth. Taking
# whichever match RDKit returned first is a coin toss, per compound.
#
# So the alignment is chosen rather than accepted: one embedding per molecule,
# picked to minimise the number of positions that vary, and then the number of
# positions bearing substituents. Nothing here knows what a benzazole is. It
# is the same computation for azines, for chalcogen five-rings, for anything
# with a symmetric scaffold, and it is what makes the X count a property of
# the set rather than of RDKit's match ordering.
#
# The optimiser is deliberately pure Python over precomputed profiles: the
# chemistry happens once, when the embeddings are enumerated, and the search
# is then plain data. That keeps it testable without RDKit installed.

def alignment_cost(profiles: Sequence[Sequence[tuple]],
                   choice: Sequence[int]) -> tuple:
    """(variable positions, attachment positions, deterministic tiebreak).

    `profiles[i][c]` is molecule i's c-th embedding as
    (symbols per core position, frozenset of positions bearing substituents).
    Pure; no RDKit.
    """
    if not profiles:
        return (0, 0, ())
    cols = list(zip(*(profiles[i][c][0] for i, c in enumerate(choice))))
    attach = set()
    for i, c in enumerate(choice):
        attach |= profiles[i][c][1]
    return (sum(1 for col in cols if len(set(col)) > 1),
            len(attach),
            tuple(tuple(sorted(set(col))) for col in cols))


class _Tally:
    """Running column tallies, so swapping one embedding costs O(positions).

    Recomputing the whole cost inside the descent loop is O(molecules x
    positions) per trial and turns a 500-compound set into a coffee break.
    """

    def __init__(self, profiles, choice):
        n_pos = len(profiles[0][0][0])
        self.cols = [Counter() for _ in range(n_pos)]
        self.att = [0] * n_pos
        for i, c in enumerate(choice):
            self.add(profiles[i][c])

    def add(self, prof):
        syms, att = prof
        for k, s in enumerate(syms):
            self.cols[k][s] += 1
        for k in att:
            self.att[k] += 1

    def remove(self, prof):
        syms, att = prof
        for k, s in enumerate(syms):
            self.cols[k][s] -= 1
            if self.cols[k][s] <= 0:
                del self.cols[k][s]
        for k in att:
            self.att[k] -= 1

    def cost(self) -> tuple:
        return (sum(1 for c in self.cols if len(c) > 1),
                sum(1 for a in self.att if a > 0),
                tuple(tuple(sorted(c)) for c in self.cols))


EXHAUSTIVE_BUDGET = 20000


def _seeds(profiles: Sequence[Sequence[tuple]],
           max_seeds: int) -> List[List[int]]:
    """Starting alignments for the descent, worst to best guess.

    Coordinate descent moves one molecule at a time, and the X count is a
    property of whole columns: it only falls when *every* molecule that could
    line up has lined up. Flipping any single compound out of a two-X
    alignment leaves it at two, so every single-molecule move looks worthless
    and the descent stops on the spot. A sixteen-compound benzazole set sat in
    exactly that hole -- two X reported, one X available, and no sequence of
    one-at-a-time improvements between them.

    So the seeds do the lining-up that descent cannot. Each takes one
    molecule's embedding as a reference and gives every other molecule the
    embedding that disagrees with it at the fewest positions -- "read them all
    the same way round", which is the move the objective rewards and the
    descent has no way to make. Descent then polishes whatever that leaves.
    """
    n = len(profiles)
    seeds: List[List[int]] = [[0] * n]
    # Molecules with a choice to make; one with a single embedding tells us
    # nothing about how to orient the rest.
    refs = [i for i in range(n) if len(profiles[i]) > 1][:max(1, max_seeds)]
    for r in refs:
        for c in range(len(profiles[r])):
            ref_syms = profiles[r][c][0]
            choice = []
            for i in range(n):
                if i == r:
                    choice.append(c)
                    continue
                choice.append(min(
                    range(len(profiles[i])),
                    key=lambda k: (sum(1 for a, b in zip(profiles[i][k][0],
                                                         ref_syms) if a != b),
                                   k)))
            seeds.append(choice)
    return seeds


def alignment_combinations(profiles: Sequence[Sequence[tuple]],
                           cap: int = 10 ** 9) -> int:
    """How many whole-set alignments exist. Stops counting past `cap`."""
    total = 1
    for p in profiles:
        total *= max(len(p), 1)
        if total > cap:
            return cap + 1
    return total


def optimize_alignment(profiles: Sequence[Sequence[tuple]],
                       rounds: int = 6, max_seeds: int = 6,
                       budget: int = EXHAUSTIVE_BUDGET) -> List[int]:
    """One embedding index per molecule, minimising variance then attachments.

    Exhaustive when the whole search space fits in `budget`, which on real
    sets it usually does -- three azanaphthalenes with four embeddings each
    is 64 combinations, and enumerating 64 things to get a guaranteed answer
    beats a heuristic that is probably right. Beyond the budget it falls back
    to coordinate descent from several seeds.

    The distinction matters enough to report: descent on a core with a large
    automorphism group can settle in a local minimum, and a table quietly
    built on the second-best alignment is exactly the kind of wrong that does
    not announce itself. `alignment_method` on the analysis says which ran.

    Pure; no RDKit.
    """
    if not profiles:
        return []

    if alignment_combinations(profiles, budget) <= budget:
        best, best_key = None, None
        for combo in itertools.product(*[range(len(p)) for p in profiles]):
            key = alignment_cost(profiles, combo)
            if best_key is None or key < best_key:
                best, best_key = list(combo), key
        return best or []

    n = len(profiles)
    best: Optional[Tuple[tuple, List[int]]] = None

    for choice in _seeds(profiles, max_seeds):
        tally = _Tally(profiles, choice)
        for _ in range(max(1, rounds)):
            changed = False
            for i in range(n):
                if len(profiles[i]) < 2:
                    continue
                cur = choice[i]
                tally.remove(profiles[i][cur])
                scored = []
                for c in range(len(profiles[i])):
                    tally.add(profiles[i][c])
                    scored.append((tally.cost(), c))
                    tally.remove(profiles[i][c])
                pick = min(scored)[1]
                tally.add(profiles[i][pick])
                if pick != cur:
                    choice[i] = pick
                    changed = True
            if not changed:
                break
        key = (tally.cost(), tuple(choice))
        if best is None or key < best[0]:
            best = (key, list(choice))
    return best[1] if best else []


def tied_alternatives(profiles: Sequence[Sequence[tuple]],
                      choice: Sequence[int]) -> List[int]:
    """Molecules whose embedding could be swapped at no cost.

    These are the genuinely arbitrary ones. Benzimidazole is the standing
    example: nitrogen sits at both ring positions, so which benzo carbon its
    substituent is assigned to is a coin toss whatever the core says -- and
    honestly so, since 5- and 6-substituted benzimidazoles are tautomers.
    Pure; no RDKit.
    """
    if not profiles:
        return []
    base = alignment_cost(profiles, choice)
    out = []
    for i, c in enumerate(choice):
        for alt in range(len(profiles[i])):
            if alt == c:
                continue
            trial = list(choice)
            trial[i] = alt
            if alignment_cost(profiles, trial) == base:
                out.append(i)
                break
    return out


# --------------------------------------------------------------------------
# variable-position analysis  (derived from matches, not from SMARTS)
# --------------------------------------------------------------------------

def embedding_profiles(mols: Sequence, query_mol, max_embeddings: int = 32,
                       max_raw_matches: int = 512):
    """Enumerate each molecule's embeddings once, as plain data.

    Returns (profiles, matches, matched_idx, unmatched_idx). This is the only
    place the alignment machinery touches RDKit; everything downstream of it
    is pure Python and testable without a chemistry toolkit installed.

    Matches are taken with uniquify=False. RDKit's uniquify collapses matches
    that cover the same *set* of atoms, which is precisely the wrong
    equivalence here: reading a fused bicycle the other way round selects the
    same ten atoms and hands them to different core positions, and choosing
    between those readings is the entire job of the optimiser downstream. With
    uniquify on, the alternative was deleted before anything could weigh it,
    so an indole/benzimidazole/benzothiazole set was stuck reporting two
    variable positions when one would do.

    Deduplication instead happens on the profile itself, which is what the
    optimiser actually consumes: a benzene ring matched twelve equivalent ways
    collapses to one entry, while two readings that differ in where the
    nitrogen lands are both kept.
    """
    profiles, matches, matched, unmatched = [], [], [], []
    truncated: List[int] = []
    for i, mol in enumerate(mols):
        if mol is None:
            unmatched.append(i)
            continue
        embs = mol.GetSubstructMatches(query_mol, uniquify=False,
                                       maxMatches=max_raw_matches)
        if not embs:
            unmatched.append(i)
            continue
        if len(embs) >= max_raw_matches:
            truncated.append(i)
        prof, keep, seen = [], [], set()
        for emb in embs:
            inside = set(emb)
            syms, att = [], set()
            for q_idx, mol_idx in enumerate(emb):
                atom = mol.GetAtomWithIdx(mol_idx)
                syms.append(atom.GetSymbol())
                # Implicit hydrogens are not neighbours, so this counts heavy
                # substituents only -- which is what becomes an R column.
                if any(nb.GetIdx() not in inside for nb in atom.GetNeighbors()):
                    att.add(q_idx)
            key = (tuple(syms), frozenset(att))
            if key in seen:
                continue
            seen.add(key)
            prof.append(key)
            keep.append(emb)
            if len(prof) >= max_embeddings:
                if i not in truncated:
                    truncated.append(i)
                break
        profiles.append(prof)
        matches.append(keep)
        matched.append(i)
    # Truncation is reported rather than swallowed. Hitting either cap means
    # the best alignment may simply not be among the ones the optimiser got
    # to weigh, and a table built on the second-best alignment is the kind of
    # wrong that never announces itself.
    return profiles, matches, matched, unmatched, truncated


def position_elements(mols: Sequence, query_mol,
                      ids: Optional[Sequence[str]] = None, *,
                      optimize: bool = True,
                      max_embeddings: int = 32) -> dict:
    """What element does each core position hold, in each molecule?

    Returns:
        {"per_position": {query_atom_idx: {"A": "N", "B": "S", ...}},
         "elements":     {query_atom_idx: ["N", "S"]},     # sorted, distinct
         "variable":     [query_atom_idx, ...],            # >1 element
         "attachments":  {query_atom_idx: [id, ...]},      # bears a substituent
         "n_rgroups":    int,                              # estimated R columns
         "unmatched":    [id, ...]}

    A symmetric core matches a molecule more than one way, and which way is
    chosen decides how many positions appear to vary. With `optimize` set --
    the default -- one embedding per molecule is chosen to minimise that count
    (see `optimize_alignment`), so the X count reflects the chemistry of the
    set rather than the order RDKit happened to return matches in. Set it
    False to see the raw first-match behaviour.

    `ambiguous` lists compounds whose embedding could be swapped without
    changing the cost: genuinely arbitrary, not merely symmetric.

    `attachments` records which core positions carry something other than
    hydrogen, in which compounds. Its length is how many R columns the
    decomposition will produce -- worth watching, because reducing the X count
    usually buys an R.
    """
    if ids is None:
        ids = [str(i + 1) for i in range(len(mols))]
    # Everything below keys per-compound results by id, so two compounds
    # sharing one do not collide loudly -- the second overwrites the first and
    # every position looks invariant. A sheet whose ID column was blank
    # reported no variable positions at all on a set that had one. Blank and
    # repeated ids are made distinct here rather than trusted, because this is
    # where the damage would be done.
    ids = [i or f"row {n + 1}" for n, i in enumerate(map(str, ids))]
    if len(set(ids)) != len(ids):
        ids = [f"{i}#{n + 1}" if ids.count(i) > 1 else i
               for n, i in enumerate(ids)]

    n_q = query_mol.GetNumAtoms() if query_mol is not None else 0
    profiles, matches, matched, unmatched_idx, trunc_idx = embedding_profiles(
        mols, query_mol, max_embeddings)

    def _id_of(i):
        return ids[i] if i < len(ids) else str(i + 1)

    if not profiles:
        return {"per_position": {}, "elements": {}, "variable": [],
                "attachments": {}, "n_rgroups": 0,
                "unmatched": [_id_of(i) for i in unmatched_idx],
                "ambiguous": [], "alignment": {}, "optimized": optimize,
                "truncated": [], "alignment_method": "none",
                "n_alignments": 0, "n_core_atoms": n_q}

    combos = alignment_combinations(profiles, EXHAUSTIVE_BUDGET)
    if not optimize:
        choice, method = [0] * len(profiles), "first-match"
    else:
        choice = optimize_alignment(profiles)
        method = ("exhaustive" if combos <= EXHAUSTIVE_BUDGET
                  else "descent (search space too large to enumerate)")

    per_pos: Dict[int, Dict[str, str]] = defaultdict(dict)
    attach: Dict[int, set] = defaultdict(set)
    for k, mol_i in enumerate(matched):
        _id = _id_of(mol_i)
        syms, att = profiles[k][choice[k]]
        for q_idx, sym in enumerate(syms):
            per_pos[q_idx][_id] = sym
        for q_idx in att:
            attach[q_idx].add(_id)

    elements = {q: sorted({v for v in d.values()})
                for q, d in per_pos.items()}
    variable = sorted(q for q, els in elements.items() if len(els) > 1)
    return {
        "per_position": dict(per_pos),
        "elements": elements,
        "variable": variable,
        "attachments": {q: sorted(v) for q, v in attach.items()},
        "n_rgroups": len(attach),
        "unmatched": [_id_of(i) for i in unmatched_idx],
        "ambiguous": [_id_of(matched[k])
                      for k in tied_alternatives(profiles, choice)],
        # The chosen embedding per compound, so the decomposition and the
        # drawing can be held to the same alignment the table was built on.
        "alignment": {_id_of(matched[k]): matches[k][choice[k]]
                      for k in range(len(matched))},
        "optimized": optimize,
        # How the alignment was arrived at, and whether anything was left
        # out of the reckoning. Both matter when the answer looks wrong:
        # "descent" can settle in a local minimum, and a truncated compound
        # may have had a better reading that was never enumerated.
        "alignment_method": method,
        "n_alignments": combos,
        "truncated": [_id_of(i) for i in trunc_idx],
        "n_core_atoms": n_q,
    }


def label_variables(variable: Sequence[int]) -> Dict[int, str]:
    """Map core atom indices to X1, X2, ... in a stable order.

    Pure; no RDKit. Ordering is by atom index, which is deterministic for a
    given core but carries no chemical meaning -- X1 is not "ring position 1".
    """
    return {q: f"X{i}" for i, q in enumerate(sorted(variable), start=1)}


def variable_columns(analysis: dict, ids: Sequence[str]) -> Tuple[Dict[str, dict], List[str]]:
    """Per-compound X assignments.

    Returns ({compound_id: {"X1": "N", "X2": "C"}}, ["X1", "X2"]).
    Pure; no RDKit.
    """
    labels = label_variables(analysis.get("variable", []))
    cols = [labels[q] for q in sorted(labels)]
    out: Dict[str, dict] = {i: {} for i in ids}
    for q, label in labels.items():
        for _id, sym in analysis["per_position"].get(q, {}).items():
            out.setdefault(_id, {})[label] = sym
    return out, cols


def attach_variable_columns(rows: List[dict], assignments: Dict[str, dict],
                            cols: Sequence[str], id_col: str = "ID") -> List[dict]:
    """Add X columns to decomposed rows, shaped like the R-group columns.

    Giving them the same companion fields (`_name`, `_key`, `_display`,
    `_sort`, `_source`) means the existing table renderer, the sorter and the
    activity-cliff analysis all handle them with no special-casing -- and an X
    change is a single-point change, so cliffs across indole -> benzimidazole
    come out of the existing matched-pair code for free.

    Pure; no RDKit.
    """
    for r in rows:
        vals = assignments.get(str(r.get(id_col, "")), {})
        for c in cols:
            sym = vals.get(c, "")
            r[c] = sym
            r[f"{c}_name"] = sym
            r[f"{c}_symbol"] = sym
            r[f"{c}_key"] = sym
            r[f"{c}_smiles"] = ""
            r[f"{c}_source"] = "variable-atom" if sym else "empty"
            r[f"{c}_display"] = "name"
            r[f"{c}_heavy"] = 1 if sym else 0
            # Element order, so a column reads C, N, O, S rather than
            # alphabetically by accident.
            r[f"{c}_sort"] = f"{_ELEMENT_RANK.get(sym, 99):02d}|{sym}"
    return rows


_ELEMENT_RANK = {"C": 1, "N": 2, "O": 3, "S": 4, "P": 5, "Se": 6,
                 "F": 10, "Cl": 11, "Br": 12, "I": 13}


# --------------------------------------------------------------------------
# candidate generation
# --------------------------------------------------------------------------

def _mcs_params(preset: dict, threshold: float, timeout: int,
                whole_rings: bool, store_all: bool) -> "rdFMCS.MCSParameters":
    p = rdFMCS.MCSParameters()
    if preset["scope"] == "none":
        p.AtomTyper = rdFMCS.AtomCompare.CompareElements
    else:
        p.AtomTyper = VariableAtomCompare(preset["scope"],
                                          preset["interchange"])
    p.BondTyper = rdFMCS.BondCompare.CompareOrder
    p.AtomCompareParameters.RingMatchesRingOnly = whole_rings
    p.BondCompareParameters.RingMatchesRingOnly = whole_rings
    p.BondCompareParameters.CompleteRingsOnly = whole_rings
    # Valences must not be matched: pyrrole N has an H where thiophene S does
    # not, and requiring equal valence would reject the azole swap outright.
    p.AtomCompareParameters.MatchValences = False
    p.Threshold = threshold
    p.Timeout = timeout
    p.StoreAll = store_all
    return p


def _results_from(mcs) -> List[Tuple[str, object]]:
    """(smarts, queryMol) pairs, handling the StoreAll quirk.

    With StoreAll set, `queryMol` is None and `smartsString` is '' by design;
    the answers live in `degenerateSmartsQueryMolDict`.
    """
    dd = getattr(mcs, "degenerateSmartsQueryMolDict", None)
    if dd:
        return list(dd.items())
    if mcs.smartsString and mcs.queryMol is not None:
        return [(mcs.smartsString, mcs.queryMol)]
    return []


def coverage_of(mols: Sequence, query_mol) -> float:
    ok = [m for m in mols if m is not None]
    if not ok or query_mol is None:
        return 0.0
    return sum(1 for m in ok if m.HasSubstructMatch(query_mol)) / len(ok)


# --------------------------------------------------------------------------
# pinned variants  --  trading an X for an R
# --------------------------------------------------------------------------
#
# A secondary mechanism, and worth being clear about what it is *not* for.
#
# Pinning was written to reduce the X count on a symmetric core, by fixing one
# position and letting the molecules flip onto it. Choosing the alignment does
# that job properly and generally, so pinning no longer earns its keep there:
# on a set where the alignment is already optimal, no coverage-free pin exists
# to find.
#
# What remains is the deliberate trade. Fixing a position to an element that
# only some compounds carry isolates a sub-series -- pin the benzazole position
# to S and you have the thiazoles alone, at a quarter of the coverage. That is
# a chemist's decision about which compounds to look at, not an optimisation,
# so it is off unless `pin_coverage_tolerance` says how much coverage may go.

def _pin_query(query_mol, position: int, element: str):
    """Copy `query_mol` with `position` additionally required to be `element`.

    The element test is ANDed onto whatever the MCS already demanded of that
    atom -- aromaticity, ring membership, connectivity -- rather than replacing
    it. Replacing would quietly relax the query everywhere else, and a core
    that matches *more* than the MCS did is not a variant of the MCS.

    Returns None if the atom carries no query to expand, which happens for
    query mols built by hand rather than by FindMCS.
    """
    try:
        z = Chem.GetPeriodicTable().GetAtomicNumber(element)
    except Exception:
        return None
    rw = Chem.RWMol(query_mol)
    if position >= rw.GetNumAtoms():
        return None
    atom = rw.GetAtomWithIdx(position)
    if not hasattr(atom, "ExpandQuery"):
        return None
    atom.ExpandQuery(rdqueries.AtomNumEqualsQueryAtom(z),
                     Chem.rdchem.CompositeQueryType.COMPOSITE_AND)
    return rw.GetMol()


def _element_query(symbols: Sequence[str]):
    """A query atom matching any one of `symbols`, as an OR chain."""
    pt = Chem.GetPeriodicTable()
    zs = sorted({pt.GetAtomicNumber(s) for s in symbols})
    qa = rdqueries.AtomNumEqualsQueryAtom(zs[0])
    for z in zs[1:]:
        qa.ExpandQuery(rdqueries.AtomNumEqualsQueryAtom(z),
                       Chem.rdchem.CompositeQueryType.COMPOSITE_OR)
    return qa


def specialize_query(query_mol, analysis: dict):
    """Narrow every core position to exactly the elements the data holds.

    This is what stops a core advertising possibilities that are not in the
    set. A permissive comparator asks RDKit to treat a class of atoms as
    interchangeable, and the SMARTS that comes back describes that class --
    so a core found over indoles, benzimidazoles and benzothiazoles could be
    written with oxygen among its options, for a set containing no oxygen at
    all. Rewriting each atom from the observed elements makes the core a
    statement about the compounds in hand.

    It earns its place twice over: an invariant position becomes a fixed
    element, which removes the symmetry that let the molecule match two ways,
    so the R-group decomposition is forced into the same alignment the table
    was built on instead of choosing its own.

    Returns None if the query carries no expandable atom queries.
    """
    elements = analysis.get("elements") or {}
    if not elements:
        return None
    rw = Chem.RWMol(query_mol)
    for q, els in elements.items():
        if q >= rw.GetNumAtoms() or not els:
            continue
        atom = rw.GetAtomWithIdx(q)
        if not hasattr(atom, "ExpandQuery"):
            return None
        atom.ExpandQuery(_element_query(els),
                         Chem.rdchem.CompositeQueryType.COMPOSITE_AND)
    return rw.GetMol()


def same_selection(mols: Sequence, q1, q2, max_matches: int = 64) -> bool:
    """Do two queries pick out exactly the same atoms, every way round?

    Stronger than comparing coverage: two cores can match the same compounds
    while disagreeing about which atom is which, and that disagreement is what
    puts a substituent in the wrong R column.
    """
    for m in mols:
        if m is None:
            continue
        a = set(m.GetSubstructMatches(q1, uniquify=True,
                                      maxMatches=max_matches))
        b = set(m.GetSubstructMatches(q2, uniquify=True,
                                      maxMatches=max_matches))
        if a != b:
            return False
    return True


def same_atoms(mols: Sequence, q1, q2, max_matches: int = 64) -> bool:
    """Do two queries pick out the same atoms, ignoring which position is which?

    Weaker than `same_selection` on purpose: this is true when the only
    difference between the queries is how their own atoms are numbered.
    """
    for m in mols:
        if m is None:
            continue
        a = {frozenset(t) for t in m.GetSubstructMatches(
            q1, uniquify=True, maxMatches=max_matches)}
        b = {frozenset(t) for t in m.GetSubstructMatches(
            q2, uniquify=True, maxMatches=max_matches)}
        if a != b:
            return False
    return True


def canonical_query(smarts: str, query_mol, mols: Sequence):
    """Re-read a query from its own SMARTS, so one index means one position.

    FindMCS numbers the query's atoms in the order it built them, while
    MolToSmarts writes them in traversal order, and the two orders differ. That
    matters because everything downstream is keyed by query atom index --
    `per_position`, the X labels, which R column a substituent lands in -- and
    `sar.py table --core '<smarts>'` and the notebook both re-parse the printed
    string. Analysed one way and applied the other, a label moves quietly onto
    a different atom.

    Reading the query back from the string it will be printed as removes the
    discrepancy at the source rather than checking for it later: the core that
    was analysed is then literally the core that string yields. Re-parsing is a
    fixed point, so the round-trip checks downstream hold by construction.

    Falls back to the original query if the string does not parse or does not
    select the same atoms -- a change of meaning, not merely of numbering.
    """
    if not smarts or query_mol is None:
        return query_mol
    rep = Chem.MolFromSmarts(smarts)
    if rep is None or rep.GetNumAtoms() != query_mol.GetNumAtoms():
        return query_mol
    return rep if same_atoms(mols, query_mol, rep) else query_mol


def reachable_elements(mols: Sequence, query_mol,
                       max_matches: int = 32) -> Dict[int, set]:
    """Every element a core position can hold, over *all* alignments.

    `position_elements` reports what the first match happens to give; this
    reports what any match could give. The difference is exactly the set of
    pins worth trying, and on a symmetric core it is not empty.
    """
    reach: Dict[int, set] = defaultdict(set)
    for mol in mols:
        if mol is None:
            continue
        for match in mol.GetSubstructMatches(query_mol, uniquify=True,
                                             maxMatches=max_matches):
            for q_idx, mol_idx in enumerate(match):
                reach[q_idx].add(mol.GetAtomWithIdx(mol_idx).GetSymbol())
    return dict(reach)


def profile_signature(analysis: dict) -> tuple:
    """Order-independent fingerprint of a candidate's per-compound table.

    Two pins that differ only in which half of a symmetric core was fixed
    produce the same assignments with the position labels swapped -- the same
    table, twice. Sorting the per-position profiles collapses them to one.
    Pure; no RDKit.
    """
    return tuple(sorted(
        tuple(sorted(per.items()))
        for per in analysis.get("per_position", {}).values()))


def pin_variants(cand: dict, mols: Sequence, ids: Sequence[str], *,
                 coverage_tolerance: float = 0.0,
                 max_per_candidate: int = 4) -> List[dict]:
    """Derive cores that fix one variable position, keeping coverage.

    For every variable position and every element reachable there, build the
    pinned query and re-derive coverage and assignments against it. A variant
    is kept only if it needs strictly fewer X columns than its parent and
    loses no more than `coverage_tolerance` coverage -- so the ones that merely
    throw compounds away (pinning the benzazole position to S keeps only the
    thiazoles) drop out on their own, without a special rule.
    """
    parent_q = cand.get("query_mol")
    analysis = cand.get("analysis") or {}
    variable = list(analysis.get("variable", []))
    if parent_q is None or not variable:
        return []

    parent_cov = float(cand.get("coverage", 0.0))
    parent_nvar = len(variable)
    # Filter ids alongside mols, not separately. Dropping unparseable
    # structures from one list and not the other shifts every assignment by
    # one and the table silently reports the wrong compound's elements.
    pairs = [(i, m) for i, m in zip(ids, mols) if m is not None]
    good_ids = [i for i, _ in pairs]
    good = [m for _, m in pairs]
    if not good:
        return []

    reach = reachable_elements(good, parent_q)
    parent_first = _first_matches(good, parent_q)

    out: List[dict] = []
    seen_sigs = {profile_signature(analysis)}
    for q in variable:
        for element in sorted(reach.get(q, set())):
            pinned = _pin_query(parent_q, q, element)
            if pinned is None:
                continue
            cov = coverage_of(good, pinned)
            if cov < parent_cov - coverage_tolerance - 1e-9:
                continue
            new_analysis = position_elements(good, pinned, good_ids)
            if len(new_analysis["variable"]) >= parent_nvar:
                continue
            sig = profile_signature(new_analysis)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)

            assignments, xcols = variable_columns(new_analysis, good_ids)
            realigned = sum(
                1 for k, m in enumerate(good)
                if parent_first[k] is not None
                and _first_match(m, pinned) not in (None, parent_first[k]))
            label = cand.get("preset_label", "")
            # Name the pin by the X label the chemist saw on the parent card.
            # "position 4 pinned to N" is an atom index and means nothing to
            # anyone; "X1 pinned to N" is the column that just disappeared.
            was = cand.get("variable_labels", {}).get(q) or f"position {q}"
            pin_note = f"{was} pinned to {element}"
            out.append({
                **{k: cand[k] for k in ("preset", "n_atoms", "n_bonds",
                                        "n_ring_atoms", "thresholds",
                                        "timed_out")
                   if k in cand},
                "smarts": Chem.MolToSmarts(pinned),
                "query_mol": pinned,
                "presets": set(cand.get("presets", ())),
                "preset_label": f"{label}; {pin_note}" if label else pin_note,
                "coverage": cov,
                "analysis": new_analysis,
                "variable_labels": label_variables(new_analysis["variable"]),
                "variable_columns": xcols,
                "assignments": assignments,
                "n_variable": len(new_analysis["variable"]),
                "derived": "pin",
                "pin": {"position": q, "element": element},
                "parent_smarts": cand.get("smarts"),
                "parent_n_variable": parent_nvar,
                "n_realigned": realigned,
            })

    # Coverage first here too, for the same reason it comes first in the
    # overall ranking. Sorting on n_variable would be actively perverse: a pin
    # that keeps one compound has no variable positions left at all, and would
    # crowd out the full-coverage variant this function exists to find.
    out.sort(key=lambda c: (-c["coverage"], c["n_variable"],
                            -c.get("n_realigned", 0), c["smarts"]))
    return out[:max_per_candidate]


def saturated_ring_heteroatoms(mols: Sequence) -> set:
    """Elements other than carbon sitting in a non-aromatic ring, set-wide.

    More than one of them is the signature of a series whose variable atom no
    aromatic preset can reach -- a thiepine against an azepine against an
    oxepine, say. One of them, or none, means there is nothing there for a
    permissive preset to find and no reason to suggest one.
    """
    out = set()
    for mol in mols:
        if mol is None:
            continue
        for a in mol.GetAtoms():
            if (a.IsInRing() and not a.GetIsAromatic()
                    and a.GetAtomicNum() not in (0, 6)):
                out.add(a.GetSymbol())
    return out


def n_ring_atoms(query_mol) -> int:
    """How many of a core's atoms are in a ring -- its scaffold size.

    Query mols carry no ring perception until it is asked for, so this does
    the asking rather than trusting whatever the caller's mol happens to hold.
    """
    if query_mol is None:
        return 0
    q = Chem.Mol(query_mol)
    Chem.FastFindRings(q)
    return sum(1 for a in q.GetAtoms() if a.IsInRing())


def peripheral_atoms(query_mol) -> List[int]:
    """Core atoms that are a hanging substituent rather than the scaffold.

    One connection inside the core and not in a ring: the chlorine and the
    methyl of a 5-chloro-2-methyl benzazole core. Each is a candidate for
    being freed into an R group.
    """
    if query_mol is None:
        return []
    q = Chem.Mol(query_mol)
    Chem.FastFindRings(q)          # query mols carry no ring info until asked
    return [a.GetIdx() for a in q.GetAtoms()
            if a.GetDegree() == 1 and not a.IsInRing()]


def _realign_variant(cand: dict, good: Sequence, good_ids: Sequence[str], *,
                     drop: Optional[int] = None,
                     coverage_tolerance: float = 0.0,
                     min_atoms: int = 3) -> Optional[dict]:
    """Rebuild a core with its variable positions widened, and keep it if that
    buys an X. Optionally free one peripheral atom on the way.

    The widening is the point. FindMCS writes each variable atom as the narrow
    element list it happened to see there -- [#7,#16] at one position and
    [#6,#7] at the other, for a benzazole set -- and those lists are what
    forbid the better reading, because indole's carbon cannot land on a
    position spelled "N or S". Widening every variable position to their union
    lets the readings compete; whichever alignment wins is then narrowed
    straight back down to the elements it actually puts there, so the core
    still states the set in hand rather than the union it was searched with.

    `drop` additionally removes a peripheral atom. FindMCS maximises, so a
    substituent every compound shares gets pulled into the core, and once
    there it pins the scaffold's orientation: on a 5-chloro benzazole series
    indole's ring nitrogen sits five bonds from the chlorine and
    benzothiazole's sits four, so no alignment can bring them together while
    the chlorine is part of the core. Freeing it makes it an R group that
    lands in one column for some compounds and the other for the rest -- one
    scaffold, one X, two R positions, which is how the series gets drawn in
    the first place.

    Same acceptance rule as `pin_variants`: strictly fewer X than the parent
    and no coverage given up. A rebuild that buys nothing returns None.
    """
    parent_q = cand["query_mol"]
    analysis = cand.get("analysis") or {}
    variable = list(analysis.get("variable", []))
    parent_nvar = len(variable)
    parent_cov = float(cand.get("coverage", 0.0))
    union = sorted({e for p in variable
                    for e in (analysis.get("elements", {}).get(p) or [])})
    if not union:
        return None

    rw = Chem.RWMol(parent_q)
    if drop is not None:
        if parent_q.GetNumAtoms() - 1 < min_atoms:
            return None
        rw.RemoveAtom(drop)
        # RemoveAtom renumbers everything above it down by one.
        moved = [p - 1 if p > drop else p for p in variable if p != drop]
    else:
        moved = list(variable)
    for p in moved:
        rw.ReplaceAtom(p, _element_query(union))
    wide = rw.GetMol()
    if not wide.GetNumAtoms() or len(Chem.GetMolFrags(wide)) != 1:
        return None         # a core in two pieces is not a core

    if coverage_of(good, wide) < parent_cov - coverage_tolerance - 1e-9:
        return None
    wide_analysis = position_elements(good, wide, good_ids)
    if len(wide_analysis["variable"]) >= parent_nvar:
        return None

    # Narrow the widened positions back to what this alignment puts there. The
    # ring and aromaticity requirements live on the bonds, which are untouched.
    rw2 = Chem.RWMol(wide)
    for p in moved:
        els = wide_analysis["elements"].get(p)
        if els:
            rw2.ReplaceAtom(p, _element_query(els))
    smarts = Chem.MolToSmarts(rw2.GetMol())
    rebuilt = canonical_query(smarts, rw2.GetMol(), good)
    if rebuilt is None or not rebuilt.GetNumAtoms():
        return None

    cov = coverage_of(good, rebuilt)
    if cov < parent_cov - coverage_tolerance - 1e-9:
        return None
    new_analysis = position_elements(good, rebuilt, good_ids)
    if len(new_analysis["variable"]) >= parent_nvar:
        return None

    if drop is None:
        note = "read the same way round throughout"
        kind = "realign"
        extra = {}
    else:
        sym = parent_q.GetAtomWithIdx(drop).GetSymbol()
        note = (f"{sym if sym and sym != '*' else 'substituent'} freed to an "
                f"R group")
        kind = "trim"
        extra = {"trimmed": {"position": drop, "element": sym}}
    label = cand.get("preset_label", "")
    assignments, xcols = variable_columns(new_analysis, good_ids)
    return {
        **{k: cand[k] for k in ("preset", "thresholds", "timed_out")
           if k in cand},
        **extra,
        "smarts": smarts,
        "query_mol": rebuilt,
        "n_atoms": rebuilt.GetNumAtoms(),
        "n_bonds": rebuilt.GetNumBonds(),
        "n_ring_atoms": n_ring_atoms(rebuilt),
        "presets": set(cand.get("presets", ())),
        "preset_label": f"{label}; {note}" if label else note,
        "coverage": cov,
        "analysis": new_analysis,
        "variable_labels": label_variables(new_analysis["variable"]),
        "variable_columns": xcols,
        "assignments": assignments,
        "n_variable": len(new_analysis["variable"]),
        "derived": kind,
        "parent_smarts": cand.get("smarts"),
        "parent_n_variable": parent_nvar,
    }


def trim_variants(cand: dict, mols: Sequence, ids: Sequence[str], *,
                  coverage_tolerance: float = 0.0,
                  min_atoms: int = 3,
                  max_per_candidate: int = 4) -> List[dict]:
    """Cores that need fewer X once the alignment is free to be chosen.

    Two moves, both rebuilding through `_realign_variant`: read every compound
    the same way round, and -- where a shared substituent is what pins the
    orientation -- free that substituent into an R group first. See
    `_realign_variant` for why each is needed.
    """
    parent_q = cand.get("query_mol")
    if parent_q is None:
        return []
    if len((cand.get("analysis") or {}).get("variable", [])) < 1:
        return []

    pairs = [(i, m) for i, m in zip(ids, mols) if m is not None]
    good_ids = [i for i, _ in pairs]
    good = [m for _, m in pairs]
    if not good:
        return []

    out: List[dict] = []
    seen_sigs = {profile_signature(cand.get("analysis") or {})}
    # Widening alone first: it is the cheaper answer and keeps every atom.
    for drop in [None] + peripheral_atoms(parent_q):
        v = _realign_variant(cand, good, good_ids, drop=drop,
                             coverage_tolerance=coverage_tolerance,
                             min_atoms=min_atoms)
        if v is None:
            continue
        sig = profile_signature(v["analysis"])
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        out.append(v)

    out.sort(key=lambda c: (-c["coverage"], c["n_variable"], -c["n_atoms"],
                            c["smarts"]))
    return out[:max_per_candidate]


def _first_match(mol, query_mol):
    if mol is None or query_mol is None:
        return None
    m = mol.GetSubstructMatches(query_mol, uniquify=True, maxMatches=1)
    return m[0] if m else None


def _first_matches(mols: Sequence, query_mol) -> List[Optional[tuple]]:
    return [_first_match(m, query_mol) for m in mols]


def _as_candidate(smarts, qmol, preset_name, preset, thr, cov, analysis,
                  good_ids, mcs) -> dict:
    """Package one core into the shape the report and CLI expect."""
    assignments, xcols = variable_columns(analysis, good_ids)
    return {
        "smarts": smarts,
        "query_mol": qmol,
        "preset": preset_name,
        "presets": {preset_name},
        "preset_label": preset["label"],
        "coverage": cov,
        "n_atoms": qmol.GetNumAtoms(),
        "n_bonds": qmol.GetNumBonds(),
        "n_ring_atoms": n_ring_atoms(qmol),
        "thresholds": [thr],
        "timed_out": bool(getattr(mcs, "canceled", False)),
        "analysis": analysis,
        "variable_labels": label_variables(analysis["variable"]),
        "variable_columns": xcols,
        "assignments": assignments,
        "n_variable": len(analysis["variable"]),
    }


def _narrow(base: dict, good: Sequence, good_ids: Sequence[str],
            notes: List[str]) -> Optional[dict]:
    """The specialized twin of a candidate, or None if it fails any check.

    Every check here is a way the narrowing could quietly make things worse,
    and each returns None rather than a damaged core:

      * coverage must be unchanged and non-zero -- an equality that also holds
        when both are zero, hence the second half;
      * the X count must not rise;
      * the SMARTS must survive being written and read back. That string is
        what the CLI and the notebook hand to the decomposition, and SMARTS
        gives "," lower precedence than "&", so an unparenthesised and-of-or
        would read back as a wider query than the one analysed.
    """
    qmol, cov = base["query_mol"], base["coverage"]
    special = specialize_query(qmol, base["analysis"])
    if special is None:
        return None

    # When the comparator's own element list already matched the data exactly,
    # narrowing ANDs each atom with what it says anyway -- "[#7,#16;#7,#16]" --
    # which is the same query spelled twice as long. The MCS string is what a
    # chemist reads and pastes back, so leave it alone.
    if same_selection(good, qmol, special):
        return None

    cov2 = coverage_of(good, special)
    if cov2 <= 0 or abs(cov2 - cov) > 1e-9:
        notes.append(f"core left un-narrowed: coverage would move "
                     f"{cov:.0%} -> {cov2:.0%}")
        return None

    smarts2 = Chem.MolToSmarts(special)
    reparsed = Chem.MolFromSmarts(smarts2)
    if reparsed is None or not same_selection(good, special, reparsed):
        notes.append("core left un-narrowed: its SMARTS did not survive a "
                     "write/read round trip, so the printed core would not "
                     "have matched the one analysed")
        return None

    analysis2 = position_elements(good, special, good_ids)
    if len(analysis2["variable"]) > base["n_variable"]:
        notes.append("core left un-narrowed: it would have added a variable "
                     "position rather than removing one")
        return None

    out = dict(base)
    out.update(_as_candidate(smarts2, special, base["preset"],
                             {"label": base["preset_label"]},
                             base["thresholds"][0], cov2, analysis2, good_ids,
                             None))
    out["timed_out"] = base["timed_out"]
    out["presets"] = set(base["presets"])
    out["derived"] = "narrowed"
    out["parent_smarts"] = base["smarts"]
    return out


def find_core_candidates(
    smiles_list: Sequence[str],
    ids: Optional[Sequence[str]] = None,
    presets: Sequence[str] = DEFAULT_PRESETS,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    timeout: int = 60,
    whole_rings: bool = True,
    store_all: bool = True,
    max_candidates: Optional[int] = None,
    pin: bool = True,
    pin_rounds: int = 2,
    pin_coverage_tolerance: float = 0.0,
) -> dict:
    """Propose cores across several matching conventions, ranked by coverage.

    Returns {"candidates": [...], "notes": [...], "invalid_rows": [...]}.
    Each candidate carries smarts, query_mol, preset, preset_label, coverage,
    n_atoms, thresholds it appeared at, the variable-position analysis, and
    the per-compound X assignments.
    """
    out = {"candidates": [], "notes": [], "invalid_rows": []}

    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    out["invalid_rows"] = [i for i, m in enumerate(mols) if m is None]
    good_ids = [i for i, m in zip(
        ids or [str(k + 1) for k in range(len(mols))], mols) if m is not None]
    good = [m for m in mols if m is not None]

    if len(good) < 2:
        out["notes"].append(
            f"{len(good)} parseable structure(s); need at least two to "
            f"compare")
        return out

    # Resolve every name up front and fail loudly on one that does not exist.
    # Dropping it with a note further down was how a permissive search became
    # a strict-only one without anybody noticing.
    use, seen_names = [], set()
    for name in presets:
        key = resolve_preset(name)          # raises on an unknown name
        if key != name:
            out["notes"].append(f"preset {name!r} is now called {key!r}")
        if key not in seen_names:
            seen_names.add(key)
            use.append(key)

    if all(PRESETS[p]["scope"] == "none" for p in use):
        out["notes"].append(
            "every requested preset matches elements exactly, so no core can "
            "span a ring whose atoms differ between compounds. Add "
            "'ring-aromatic' to unify a heterocycle series")

    seen: Dict[str, dict] = {}
    # What each (preset, threshold) actually produced, reported whatever
    # happens. "Only six-membered rings came back" should be answerable from
    # the output rather than by rerunning with a debugger.
    found: Dict[tuple, list] = {}
    for preset_name, thr in itertools.product(use, thresholds):
        preset = PRESETS[preset_name]
        try:
            mcs = rdFMCS.FindMCS(good, _mcs_params(
                preset, thr, timeout, whole_rings, store_all))
        except RuntimeError as e:
            out["notes"].append(f"{preset_name} @ {thr}: RDKit error: {e}")
            continue

        pairs = _results_from(mcs)
        if not pairs:
            continue
        for smarts, qmol in pairs:
            if qmol is None:
                continue
            key = smarts
            if key in seen:
                seen[key]["thresholds"].append(thr)
                seen[key]["presets"].add(preset_name)
                continue
            # Analyse the query as the printed SMARTS yields it, not as
            # FindMCS numbered it; see canonical_query.
            qmol = canonical_query(smarts, qmol, good)
            cov = coverage_of(good, qmol)
            analysis = position_elements(good, qmol, good_ids)
            base = _as_candidate(smarts, qmol, preset_name, preset, thr, cov,
                                 analysis, good_ids, mcs)
            seen[key] = base
            found.setdefault((preset_name, thr), []).append(
                (qmol.GetNumAtoms(), cov))

            # Narrowing is a *derived* candidate, never a replacement. It
            # rewrites the query from the observed elements, which is right,
            # but it is also the newest and least-exercised step here -- and a
            # step that can silently delete the only core spanning both rings
            # is not one to put in the critical path. If it fails any of its
            # checks the MCS answer stands, unchanged.
            narrowed = _narrow(base, good, good_ids, out["notes"])
            if narrowed is not None:
                if (profile_signature(narrowed["analysis"])
                        == profile_signature(base["analysis"])):
                    seen[key] = narrowed        # same table, tighter query
                elif narrowed["smarts"] not in seen:
                    seen[narrowed["smarts"]] = narrowed

    for (preset_name, thr) in itertools.product(use, thresholds):
        got = found.get((preset_name, thr))
        if not got:
            out["notes"].append(f"MCS {preset_name} @ {thr:g}: nothing")
            continue
        biggest, cov_at = max(got)
        out["notes"].append(
            f"MCS {preset_name} @ {thr:g}: {len(got)} answer(s), largest "
            f"{biggest} atoms at {cov_at:.0%} coverage")

    # Dedup again on the *specialized* SMARTS: two different MCS answers can
    # narrow to the same core once the element lists come from the data.
    pool, final_seen = [], set()
    for c in seen.values():
        if c["smarts"] in final_seen:
            continue
        final_seen.add(c["smarts"])
        pool.append(c)

    # Pinned and trimmed variants are derived from the MCS answers, not found
    # by FindMCS, and can themselves be derived from again -- a three-X core may
    # come down to one in two rounds, and a core may need both its chlorine
    # freed and a position pinned. Each round only sees what the previous round
    # produced.
    if pin:
        sigs = {profile_signature(c["analysis"]) for c in pool}
        frontier = pool
        for _ in range(max(0, pin_rounds)):
            fresh = []
            for parent in frontier:
                for v in itertools.chain(
                        trim_variants(
                            parent, good, good_ids,
                            coverage_tolerance=pin_coverage_tolerance),
                        pin_variants(
                            parent, good, good_ids,
                            coverage_tolerance=pin_coverage_tolerance)):
                    sig = profile_signature(v["analysis"])
                    if sig in sigs:
                        continue
                    sigs.add(sig)
                    fresh.append(v)
            if not fresh:
                break
            pool.extend(fresh)
            frontier = fresh

    cands = rank_candidates(pool)
    if max_candidates is not None:
        cands = cands[:max_candidates]
    out["candidates"] = cands

    if not cands:
        out["notes"].append("no core found under any of the requested presets")
    elif (not any(c["n_variable"] for c in cands
                  if c["coverage"] >= max(x["coverage"] for x in cands) - 1e-9)
          and not any(PRESETS[p]["scope"] in ("ring", "any") for p in use)
          and len(saturated_ring_heteroatoms(good)) > 1):
        # A heteroatom that varies inside a *saturated* ring is invisible to
        # every preset here: "ring-aromatic" is the most permissive of them
        # and it requires the atom to be aromatic. The symptom is not an
        # error but an absence -- a thiepine/azepine/oxepine series comes back
        # as separate strict cores with no X and nothing said. Say it.
        out["notes"].append(
            "no candidate needed a variable position. If a ring heteroatom "
            "varies in a saturated ring, the presets used here cannot see it "
            "-- 'ring-aromatic' requires the atom to be aromatic. Add "
            "'ring-any' to search non-aromatic ring atoms too")
    return out


def rank_candidates(cands: List[dict]) -> List[dict]:
    """Coverage, ring system, fewest X, then size. Pure; no RDKit.

    Not size-first: a permissive comparator always finds a core at least as
    large as a strict one, so ranking by size would always float the vaguest
    candidate to the top. Coverage says what a core is actually worth; how
    specific it is, the chemist judges from the drawing.

    The scaffold is the ring system, so that is what size means here. Ranking
    on total atoms instead gets two things wrong. A bare benzene with no X
    would outrank the whole bicyclic core with one X, and the chemist loses
    the ring they care about -- which is why atom count outranked the X count
    before. But a trimmed variant is one atom *smaller* than the core it came
    from, having moved a substituent into an R column, so counting atoms
    buries exactly the representation a chemist would draw: one scaffold with
    an X and an R, rather than two variable positions held apart by a
    chlorine that was never part of the scaffold.

    Ring atoms settle both. Benzene's six lose to the bicycle's nine whatever
    the X count, while a trim ties its parent's ring system and then wins on
    needing one X instead of two. A pinned variant keeps every atom and wins
    on the same key, as it did before.
    """
    ordered = sorted(
        cands,
        key=lambda c: (-round(c.get("coverage", 0.0), 6),
                       -c.get("n_ring_atoms", c.get("n_atoms", 0)),
                       c.get("n_variable", 0),
                       -c.get("n_atoms", 0),
                       c.get("smarts", "")))
    for i, c in enumerate(ordered, start=1):
        c["rank"] = i
    return ordered


def describe(cand: dict) -> List[str]:
    """Readable summary lines for one candidate."""
    a = cand["analysis"]
    tag = {"pin": "  [pinned]", "narrowed": "  [narrowed to the data]",
           "trim": "  [substituent freed to an R group]"}.get(
        cand.get("derived"), "")
    lines = [
        f"#{cand.get('rank', '?')}  {cand['preset']}{tag} — "
        f"{cand['preset_label']}",
        f"    coverage {cand['coverage']:.0%}   {cand['n_atoms']} atoms   "
        f"{cand['n_variable']} variable position(s)   "
        f"~{a.get('n_rgroups', 0)} R column(s)",
    ]
    if cand.get("derived") == "pin":
        n = cand.get("n_realigned", 0)
        parent_n = cand.get("parent_n_variable", cand["n_variable"] + 1)
        lines.append(
            f"    derived from a {parent_n}-X core by fixing one position; "
            f"{n} compound(s) realigned"
            + (", so expect an extra R position" if n else ""))
    labels = cand["variable_labels"]
    for q in sorted(labels):
        els = a["elements"].get(q, [])
        counts = Counter(a["per_position"].get(q, {}).values())
        detail = ", ".join(f"{e}×{counts[e]}" for e in els)
        lines.append(f"    {labels[q]} = {'/'.join(els)}   ({detail})")
    if a.get("truncated"):
        lines.append(
            f"    WARNING: {len(a['truncated'])} compound(s) hit the "
            f"embedding cap, so the best alignment may not have been "
            f"considered: {', '.join(map(str, a['truncated'][:6]))}")
    if a.get("alignment_method", "").startswith("descent"):
        lines.append(f"    alignment by {a['alignment_method']} — "
                     f"{a.get('n_alignments', 0):,}+ possibilities, so this "
                     f"is a good answer rather than a proven best")
    if a["unmatched"]:
        lines.append(f"    unmatched: {', '.join(map(str, a['unmatched'][:8]))}")
    if a["ambiguous"]:
        lines.append(
            f"    {len(a['ambiguous'])} compound(s) sit on the core equally "
            f"well either way round, so their R positions are arbitrary: "
            + ", ".join(map(str, a["ambiguous"][:6])))
    if cand.get("timed_out"):
        lines.append("    TIMED OUT — not maximal")
    lines.append(f"    {cand['smarts']}")
    return lines


def summarize(result: dict) -> List[str]:
    lines = [f"{len(result['candidates'])} candidate core(s)"]
    if result.get("invalid_rows"):
        lines.append(f"{len(result['invalid_rows'])} structure(s) did not parse")
    for c in result["candidates"]:
        lines.extend(describe(c))
    for n in result.get("notes", []):
        lines.append(f"note: {n}")
    return lines
