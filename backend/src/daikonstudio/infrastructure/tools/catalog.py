"""The tools this server offers, as a module-level tuple.

A handful of curated tools is not a discovery problem, so there is no plugin scan
and no entry-point lookup -- same call as `infrastructure/engines/registry.py`.
A planner has to be able to trust that the catalogue is the same on every replica,
and a loader would make it depend on what happens to be installed.

Order is presentation order: roughly the order a chemist performs them, so the
catalogue page reads as a workflow rather than an alphabetical list.

Several descriptions below name the thing a naive version gets wrong. That is
deliberate and load-bearing: the planner picks tools from these sentences, so a
description that reads well but omits the trap produces plans that look right.
Most of those traps were paid for already in the sibling SAR workbench, whose
hand-off notes are the source for the ones marked as such.
"""

from __future__ import annotations

from daikonstudio.application.tools.manifest import (
    Slot,
    SlotKind,
    ToolArea,
    ToolManifest,
    ToolSurface,
)
from daikonstudio.application.tools.registry import ToolRegistry
from daikonstudio.infrastructure.tools.params import (
    ActivityParams,
    AssayParams,
    CliffParams,
    ClusterParams,
    ConfoundParams,
    CoreParams,
    DiverseParams,
    DockParams,
    FetchAlphafoldParams,
    FetchPdbParams,
    FetchUniprotParams,
    InteractionParams,
    PrepareParams,
    SimilarityParams,
    SubstructureParams,
    ValidityParams,
)


def _in(key: str, kind: SlotKind, label: str, *, optional: bool = False) -> Slot:
    return Slot(key=key, kind=kind, label=label, optional=optional)


_COMPOUNDS = _in("compounds", SlotKind.COMPOUND_TABLE, "Compounds")
_CORE = _in("core", SlotKind.CORE, "Core")
_TABLE = _in("table", SlotKind.RGROUP_TABLE, "SAR table")
_PAIRS = _in("pairs", SlotKind.PAIRS, "Matched pairs")


_TOOLS: tuple[ToolManifest, ...] = (
    ToolManifest(
        id="standardize",
        version="1",
        area=ToolArea.DATA,
        name="Standardize structures",
        description=(
            "Strips salts and solvents, normalizes charges and tautomers, and reports "
            "every structure it changed and every one that would not parse. Runs before "
            "anything else touches a table: "
            "the same compound drawn two ways reads as two compounds, which quietly "
            "inflates a series and breaks the pairing every later step depends on."
        ),
        inputs=(_COMPOUNDS,),
        outputs=(_in("compounds", SlotKind.COMPOUND_TABLE, "Standardized compounds"),),
        library="rdkit.Chem.MolStandardize",
        automatic=True,
        surface=ToolSurface.NONE,
    ),
    ToolManifest(
        id="cluster_series",
        version="1",
        area=ToolArea.SAR,
        name="Find series",
        description=(
            "Groups compounds that share chemistry, and says whether splitting them is "
            "worth it. Compares ring scaffolds rather than whole molecules, because on "
            "whole molecules the substituents dominate the similarity and one scaffold "
            "wearing two different groups reads as two series. Recommends a split only "
            "when the typical group's core is substantially bigger than the core the "
            "whole set already gives, so a set that clusters cleanly but analyses fine "
            "as one series is left alone."
        ),
        inputs=(_COMPOUNDS,),
        outputs=(_in("series", SlotKind.SERIES, "Series"),),
        library="rdkit (Butina over Morgan fingerprints of Murcko scaffolds)",
        judgment=True,
        surface=ToolSurface.EDITOR,
    ),
    ToolManifest(
        id="find_cores",
        version="1",
        area=ToolArea.SAR,
        name="Propose cores",
        description=(
            "Searches for the largest substructure the series shares and returns ranked "
            "candidates, each drawn, with what fraction it covers, which compounds it "
            "misses, and which ring positions it had to let vary. Allows a position to "
            "match several elements, so an indole, a benzimidazole and a benzothiazole "
            "collapse into one core with one variable position instead of three "
            "separate scaffolds. Rank one is the first option, never the answer: on a "
            "benzamide series the top core reaches full coverage only by dropping the "
            "amide carbonyl so a thioamide fits, and a chemist takes the narrower one."
        ),
        inputs=(_COMPOUNDS, _in("series", SlotKind.SERIES, "Series", optional=True)),
        outputs=(_in("candidates", SlotKind.CORE_CANDIDATES, "Candidate cores"),),
        library="sar-workbench (RDKit maximum common substructure)",
        judgment=True,
        resolves_to=SlotKind.CORE,
        surface=ToolSurface.EDITOR,
        external=True,
    ),
    ToolManifest(
        id="score_core",
        version="1",
        area=ToolArea.SAR,
        name="Score a core",
        description=(
            "Takes a core a chemist drew or typed, given as a setting, and reports "
            "what it covers, plus a sentence for each compound it misses saying why it "
            "missed. The route by which a scientist overrules the search: nothing is "
            "built from a hand-drawn core until it has been scored. Emits the same "
            "shape as a proposed candidate, so a drawn core flows into the rest of a "
            "study untouched."
        ),
        inputs=(_COMPOUNDS,),
        outputs=(_in("candidates", SlotKind.CORE_CANDIDATES, "Scored core"),),
        library="sar-workbench (RDKit substructure matching)",
        judgment=True,
        resolves_to=SlotKind.CORE,
        surface=ToolSurface.EDITOR,
        external=True,
        params=CoreParams,
    ),
    ToolManifest(
        id="decompose_rgroups",
        version="1",
        area=ToolArea.SAR,
        name="Decompose R groups",
        description=(
            "Splits every compound into the approved core plus its substituents, one "
            "column per attachment point. This is the SAR table. Reports the fraction "
            "it could not decompose rather than dropping those compounds, because a "
            "table built on two thirds of a series looks exactly like a table built on "
            "all of it. Keeps hydrogen positions as values: an unsubstituted position "
            "is a result, not a blank."
        ),
        inputs=(_COMPOUNDS, _CORE),
        outputs=(_in("table", SlotKind.RGROUP_TABLE, "SAR table"),),
        library="rdkit.Chem.rdRGroupDecomposition",
        surface=ToolSurface.VIEW,
    ),
    ToolManifest(
        id="name_substituents",
        version="1",
        area=ToolArea.SAR,
        name="Name substituents",
        description=(
            "Resolves each R group to the word a chemist would use (Me, OMe, CF3), "
            "falling back to a drawing where no name is clearer than a picture. The "
            "decision is per group: a group with a conventional short name is written as "
            "that word, and anything else is drawn, because a SMILES string in a table "
            "cell is not something a chemist reads."
        ),
        inputs=(_TABLE,),
        outputs=(_in("table", SlotKind.RGROUP_TABLE, "Named SAR table"),),
        library="rdkit.Chem.rdAbbreviations (built-in abbreviations only)",
        surface=ToolSurface.NONE,
    ),
    ToolManifest(
        id="matched_pairs",
        version="1",
        area=ToolArea.SAR,
        name="Matched pairs",
        description=(
            "Finds every pair of compounds identical at every position but one, with "
            "the change made and the difference in activity. Free once the series is "
            "decomposed, because the table has already done the pairing. Keeps censored "
            "readings as bounds rather than turning them into numbers, so a pair of two "
            "unbounded values yields nothing instead of a fictitious zero."
        ),
        inputs=(_TABLE,),
        outputs=(_in("pairs", SlotKind.PAIRS, "Matched pairs"),),
        library="daikonstudio (over the R-group table)",
        surface=ToolSurface.VIEW,
        params=AssayParams,
    ),
    ToolManifest(
        id="effect_sizes",
        version="1",
        area=ToolArea.SAR,
        name="Effect per change",
        description=(
            "For each substituent change, the average effect on activity across every "
            "pair that made it, with a bootstrap interval. Answers which parts of a "
            "structure-activity story the data can actually carry, as opposed to which "
            "ones rest on a single pair."
        ),
        inputs=(_PAIRS,),
        outputs=(_in("effects", SlotKind.EFFECTS, "Effect sizes"),),
        library="numpy",
        surface=ToolSurface.VIEW,
    ),
    ToolManifest(
        id="activity_cliffs",
        version="1",
        area=ToolArea.SAR,
        name="Activity cliffs",
        description=(
            "Picks out pairs whose activity differs far more than a one-position change "
            "should explain. Returns every pair with the cliffs flagged and ranked, so "
            "a step reading the pairs afterwards still sees all of them. Each is either "
            "the most informative compound in the set or an assay error, and how many "
            "times each compound was measured usually says which, so the result is "
            "labelled rather than celebrated. Gated on an "
            "absolute difference, not a z-score: a bimodal series inflates the pooled "
            "spread past the largest real jump, and a single pair has no spread at all."
        ),
        inputs=(_PAIRS,),
        outputs=(_in("cliffs", SlotKind.PAIRS, "Cliffs, adjudicated"),),
        library="daikonstudio (over the matched pairs)",
        judgment=True,
        surface=ToolSurface.EDITOR,
        params=CliffParams,
    ),
    ToolManifest(
        id="property_profile",
        version="1",
        area=ToolArea.DATA,
        name="Physicochemical properties",
        description=(
            "Molecular weight, calculated logP, polar surface area, donors, acceptors, "
            "fraction sp3 and rotatable bonds per compound, plus ligand efficiency and "
            "lipophilic efficiency where the activity is a log potency and those mean "
            "something."
        ),
        inputs=(_COMPOUNDS,),
        outputs=(_in("properties", SlotKind.PROPERTY_TABLE, "Properties"),),
        library="rdkit.Chem.Descriptors",
        surface=ToolSurface.NONE,
        params=ActivityParams,
    ),
    ToolManifest(
        id="substructure_search",
        version="1",
        area=ToolArea.DATA,
        name="Substructure search",
        description=(
            "Flags which compounds contain a given substructure, written as SMARTS. "
            "Annotates the table with a column rather than returning only the hits, so a "
            "query that matched nothing reads as a count of zero instead of an empty "
            "table. Answers 'which of these have the piece I care about', which is the "
            "question that usually precedes defining a series."
        ),
        inputs=(_COMPOUNDS,),
        outputs=(_in("compounds", SlotKind.COMPOUND_TABLE, "Compounds"),),
        library="rdkit.Chem.HasSubstructMatch",
        surface=ToolSurface.NONE,
        params=SubstructureParams,
    ),
    ToolManifest(
        id="similarity_search",
        version="1",
        area=ToolArea.DATA,
        name="Similarity search",
        description=(
            "Ranks every compound by Tanimoto similarity to a query structure on Morgan "
            "fingerprints, most similar first, and says how many reach the cutoff. Use it "
            "to find the near neighbours of a hit, or to ask whether a proposed compound "
            "is actually new. Similarity is not activity: close neighbours can differ "
            "sharply in potency, which is what the activity-cliff tool is for."
        ),
        inputs=(_COMPOUNDS,),
        outputs=(_in("compounds", SlotKind.COMPOUND_TABLE, "Compounds"),),
        library="rdkit.DataStructs.TanimotoSimilarity",
        surface=ToolSurface.VIEW,
        params=SimilarityParams,
    ),
    ToolManifest(
        id="cluster_compounds",
        version="1",
        area=ToolArea.DATA,
        name="Cluster compounds",
        description=(
            "Groups a set of compounds into chemotypes by Taylor-Butina clustering, "
            "numbering clusters from largest and marking each cluster's centroid. Use it "
            "to see how many distinct chemistries a set really holds before treating it as "
            "one series, and to pick one representative per group. The cutoff decides the "
            "answer, so a cluster count is only meaningful next to the cutoff that "
            "produced it."
        ),
        inputs=(_COMPOUNDS,),
        outputs=(_in("compounds", SlotKind.COMPOUND_TABLE, "Compounds"),),
        library="rdkit.ML.Cluster.Butina",
        surface=ToolSurface.VIEW,
        params=ClusterParams,
    ),
    ToolManifest(
        id="pick_diverse_subset",
        version="1",
        area=ToolArea.DATA,
        name="Pick a diverse subset",
        description=(
            "Chooses a maximally dissimilar subset of a given size by MaxMin, for a plate, "
            "a first screen, or a spread of examples to look at by hand. This is the one "
            "tool here that returns fewer rows than it was given, and it reports what it "
            "picked from. Diversity is measured on fingerprints alone, so it knows nothing "
            "about which compounds are available or affordable."
        ),
        inputs=(_COMPOUNDS,),
        outputs=(_in("compounds", SlotKind.COMPOUND_TABLE, "Compounds"),),
        library="rdkit.SimDivFilters.MaxMinPicker",
        surface=ToolSurface.VIEW,
        params=DiverseParams,
    ),
    ToolManifest(
        id="murcko_scaffolds",
        version="1",
        area=ToolArea.SAR,
        name="Scaffolds per compound",
        description=(
            "Reduces each compound to its Bemis-Murcko scaffold, and to the generic ring "
            "skeleton beside it, so a set can be counted and grouped by chemotype. The "
            "generic form ignores which atoms the rings are made of, which is what groups "
            "two series that differ only by a ring heteroatom. Unlike finding series, this "
            "assigns a scaffold to every compound without deciding what the core is."
        ),
        inputs=(_COMPOUNDS,),
        outputs=(_in("compounds", SlotKind.COMPOUND_TABLE, "Compounds"),),
        library="rdkit.Chem.Scaffolds.MurckoScaffold",
        surface=ToolSurface.NONE,
    ),
    ToolManifest(
        id="synthetic_accessibility",
        version="1",
        area=ToolArea.DATA,
        name="Synthetic accessibility",
        description=(
            "Scores how hard each compound looks to make, from 1 for easy to 10 for hard, "
            "from the frequency of its fragments in known chemistry plus size and "
            "complexity penalties. Worth running over anything a model or an agent "
            "proposed, before it reaches a chemist. It is a prior and not a route: compare "
            "within one table rather than against a fixed cutoff, and nothing here says "
            "whether a given compound can actually be made."
        ),
        inputs=(_COMPOUNDS,),
        outputs=(_in("properties", SlotKind.PROPERTY_TABLE, "Properties"),),
        library="rdkit Contrib SA_Score (Ertl-Schuffenhauer)",
        surface=ToolSurface.NONE,
    ),
    ToolManifest(
        id="property_confound",
        version="1",
        area=ToolArea.DATA,
        name="Property confound check",
        description=(
            "Tests whether activity is really tracking a bulk property such as "
            "lipophilicity rather than anything structural. Common enough, and "
            "deflating enough, that a structure-activity claim made without it is worth "
            "less than it looks."
        ),
        inputs=(
            _COMPOUNDS,
            _in("properties", SlotKind.PROPERTY_TABLE, "Properties", optional=True),
        ),
        outputs=(_in("correlation", SlotKind.CORRELATION, "Correlation"),),
        library="numpy",
        surface=ToolSurface.VIEW,
        params=ConfoundParams,
    ),
    ToolManifest(
        id="structural_alerts",
        version="1",
        area=ToolArea.DATA,
        name="Structural alerts",
        description=(
            "Flags known problem substructures: assay interference, reactive groups, "
            "frequent hitters. Given a decomposed series it flags the substituent rather "
            "than the whole molecule, which is what tells you whether the liability sits "
            "in the part you are varying or in the scaffold you are keeping."
        ),
        inputs=(_COMPOUNDS, _in("table", SlotKind.RGROUP_TABLE, "SAR table", optional=True)),
        outputs=(_in("alerts", SlotKind.ALERTS, "Alerts"),),
        library="rdkit.Chem.FilterCatalog (PAINS, BRENK, NIH)",
        surface=ToolSurface.VIEW,
    ),
    ToolManifest(
        id="free_wilson",
        version="1",
        area=ToolArea.SAR,
        name="Additive substituent model",
        description=(
            "Fits activity as a sum of per-substituent contributions and reports how "
            "well that holds, which is itself the finding: where it fits, positions act "
            "independently and can be optimized separately; where it fails, they "
            "interact. Refuses rather than fits when the series has too few compounds "
            "for the number of distinct groups in it."
        ),
        inputs=(_TABLE,),
        outputs=(_in("fit", SlotKind.MODEL_FIT, "Model fit"),),
        library="numpy",
        surface=ToolSurface.VIEW,
        params=AssayParams,
    ),
    ToolManifest(
        id="coverage_gaps",
        version="1",
        area=ToolArea.SAR,
        name="Coverage gaps",
        description=(
            "Which combinations of substituents the series has never made. Turns the "
            "SAR table from a record of what is known into a map of what is untested, "
            "which is the question a chemist is usually actually asking."
        ),
        inputs=(_TABLE,),
        outputs=(_in("gaps", SlotKind.GAPS, "Gaps"),),
        library="polars",
        surface=ToolSurface.VIEW,
    ),
    ToolManifest(
        id="chembl_actives",
        version="1",
        area=ToolArea.CONTEXT,
        name="Public actives for a target",
        description=(
            "Known active compounds for a target from ChEMBL, for context on whether "
            "this chemotype or this structure-activity story is already published."
        ),
        inputs=(),
        outputs=(_in("compounds", SlotKind.COMPOUND_TABLE, "Public actives"),),
        library="ChEMBL web services",
        surface=ToolSurface.VIEW,
        external=True,
    ),
    ToolManifest(
        id="literature_search",
        version="1",
        area=ToolArea.CONTEXT,
        name="Literature",
        description=(
            "Searches the published literature and returns citations whose links "
            "resolve. Returns references, never claims: a sentence summarising a paper "
            "cannot be checked, and a citation can."
        ),
        inputs=(),
        outputs=(_in("citations", SlotKind.CITATIONS, "Citations"),),
        library="PubMed E-utilities",
        surface=ToolSurface.VIEW,
        external=True,
    ),
    ToolManifest(
        id="fetch_uniprot",
        version="1",
        area=ToolArea.CONTEXT,
        name="Protein entry from UniProt",
        description=(
            "Looks a protein up by accession, gene or name and returns the curated "
            "entry: canonical sequence, organism, function, the residues annotated "
            "as binding, active or metal sites, known variants, and the deposited "
            "structures and AlphaFold model that cover it. The annotations are "
            "numbered on the canonical sequence, which the author numbering in a "
            "structure file rarely matches, so a step that compares the two has to "
            "map them first. Curated knowledge, not a result: it says what has been "
            "reported for the protein, which is how a detected pocket is told apart "
            "from the site the literature already knows."
        ),
        inputs=(_in("structure", SlotKind.STRUCTURE, "Structure", optional=True),),
        outputs=(_in("annotations", SlotKind.SITE_ANNOTATIONS, "Annotated sites"),),
        library="UniProt REST API",
        surface=ToolSurface.VIEW,
        external=True,
        params=FetchUniprotParams,
    ),
    # -- Structure ---------------------------------------------------------------
    # A protein structure arrives one of two ways: the scientist attaches a file,
    # or a step fetches one by id. These are the first tools that need more than
    # RDKit, so they are also where the lanes start to matter.
    ToolManifest(
        id="fetch_pdb",
        version="1",
        name="Fetch a deposited structure",
        area=ToolArea.STRUCTURE,
        description=(
            "Downloads an entry from the Protein Data Bank by its four-character id, "
            "as deposited: every chain, ligand, cofactor and water, with the "
            "resolution and the method. Records which residues sit within 4 Å of "
            "each ligand before anything strips them, so the site a bound ligand "
            "marks can still be found once the pocket search runs on the emptied "
            "protein. An id names one entry, not the protein: the same enzyme has "
            "dozens, differing in ligand, mutation and resolution, and which one to "
            "use is a choice worth stating."
        ),
        inputs=(),
        outputs=(_in("structure", SlotKind.STRUCTURE, "Deposited structure"),),
        library="RCSB PDB (files.rcsb.org, data.rcsb.org)",
        surface=ToolSurface.VIEW,
        external=True,
        params=FetchPdbParams,
    ),
    ToolManifest(
        id="fetch_alphafold",
        version="1",
        name="Fetch a predicted structure",
        area=ToolArea.STRUCTURE,
        description=(
            "Downloads the AlphaFold model for a protein by its UniProt accession, "
            "for a protein with no deposited structure or a region no crystal "
            "covers. A prediction, not an observation: it has no ligands, cofactors "
            "or waters, so every site reads as empty, and its per-residue "
            "confidence is written where a crystal's B-factors would be. A cavity "
            "lined by low-confidence residues is disorder drawn as a surface rather "
            "than a pocket, so the confidence has to be read beside anything found "
            "on the model."
        ),
        inputs=(),
        outputs=(_in("structure", SlotKind.STRUCTURE, "Predicted structure"),),
        library="AlphaFold Protein Structure Database (EMBL-EBI)",
        surface=ToolSurface.VIEW,
        external=True,
        params=FetchAlphafoldParams,
    ),
    ToolManifest(
        id="prepare_structure",
        version="1",
        name="Prepare the structure",
        area=ToolArea.STRUCTURE,
        description=(
            "Repairs missing residues and side chains, adds hydrogens at the chosen "
            "pH and estimates each titratable residue's pKa. Removes waters and "
            "crystallisation additives. By default every other heterogen stays "
            "in place as deposited, so a cavity a cofactor fills is not reported "
            "as somewhere to aim a new molecule. Set keep_ligands to choose, and "
            "to an empty list to strip them all, which is what you want when the "
            "bound ligand is an inhibitor and its site is the question. Runs "
            "before anything else touches a "
            "structure: a pocket found on an unrepaired model can be an artefact of "
            "the gap rather than a feature of the protein, and a charge assigned "
            "wrongly changes which residues line the site."
        ),
        inputs=(_in("structure", SlotKind.STRUCTURE, "Structure"),),
        outputs=(_in("structure", SlotKind.STRUCTURE, "Prepared structure"),),
        params=PrepareParams,
        library="PDBFixer and PROPKA",
        automatic=True,
        lane="chem-tools",
        surface=ToolSurface.NONE,
        external=True,
    ),
    ToolManifest(
        id="find_pockets",
        version="1",
        name="Find binding pockets",
        area=ToolArea.STRUCTURE,
        description=(
            "Detects cavities on the protein surface with fpocket and ranks them, "
            "with the volume, "
            "the residues that line each one, and a druggability score. Returns "
            "candidates: the top-ranked pocket is frequently a crystallisation "
            "artefact or a surface groove, and whether a site is the one worth "
            "aiming at is a question about the biology rather than the geometry. "
            "Cofactors kept by preparation are part of the receptor here, so the "
            "candidates are the cavities beside them rather than the cavity one "
            "fills. Names any ligand of the fetched entry whose contact residues "
            "line a candidate, marking the ones that came from the crystallisation "
            "drop. Given the protein's curated site annotations and sequence, "
            "checks the file's residue numbering against that sequence and labels "
            "each candidate with the annotated sites it contains; where the "
            "numbering disagrees it reports the offset and labels nothing. Volumes "
            "are Monte Carlo estimates and move a few percent between runs."
        ),
        inputs=(
            _in("structure", SlotKind.STRUCTURE, "Structure"),
            _in("annotations", SlotKind.SITE_ANNOTATIONS, "Annotated sites", optional=True),
        ),
        outputs=(_in("pockets", SlotKind.POCKET_CANDIDATES, "Candidate pockets"),),
        library="fpocket",
        judgment=True,
        resolves_to=SlotKind.POCKET,
        lane="chem-tools",
        surface=ToolSurface.EDITOR,
        external=True,
    ),
    ToolManifest(
        id="predict_pockets",
        version="1",
        name="Predict binding sites",
        area=ToolArea.STRUCTURE,
        description=(
            "Scores points on the protein surface with a model trained on known "
            "protein-ligand complexes, clusters them into candidate sites and ranks "
            "those by a calibrated probability of binding a ligand, with the residues "
            "that line each one. No volume: the candidate is a patch of surface, not "
            "a cavity. Works from the protein atoms alone, so a cofactor kept by "
            "preparation is not part of the receptor here and the cleft it sits in "
            "can be reported inside a candidate; the result names the ligands it "
            "did not see. Its ranking has been right more often than a geometric "
            "score on the entries checked, which is still not a recommendation: a "
            "candidate is judged on the residues that line it, the ligands it "
            "touches and the annotated sites it contains, exactly as for a cavity "
            "search. Same numbering check and site labels as the cavity search."
        ),
        inputs=(
            _in("structure", SlotKind.STRUCTURE, "Structure"),
            _in("annotations", SlotKind.SITE_ANNOTATIONS, "Annotated sites", optional=True),
        ),
        outputs=(_in("pockets", SlotKind.POCKET_CANDIDATES, "Candidate sites"),),
        library="p2rank",
        judgment=True,
        resolves_to=SlotKind.POCKET,
        lane="chem-tools",
        surface=ToolSurface.EDITOR,
        external=True,
    ),
    # -- Predicted properties ------------------------------------------------------
    ToolManifest(
        id="dock_compounds",
        version="1",
        name="Dock compounds into the pocket",
        area=ToolArea.STRUCTURE,
        description=(
            "Places each compound in the approved pocket with AutoDock Vina and ranks the "
            "poses it finds. The search box is derived from the pocket's own residues, not "
            "typed in, because a box aimed at the wrong site is the commonest way a docking "
            "run answers a question nobody asked. Read the result as an ordering: a Vina "
            "affinity is a score, and in blind assessment molecular weight tracked the truth "
            "about as well as the best method did, so it ranks compounds in this box and is "
            "not a binding energy. Each compound takes seconds to minutes, so the number "
            "docked is capped and whatever is left over is reported rather than dropped."
        ),
        inputs=(
            _in("structure", SlotKind.STRUCTURE, "Structure"),
            _in("pocket", SlotKind.POCKET, "Pocket"),
            _COMPOUNDS,
        ),
        outputs=(
            _in("poses", SlotKind.POSES, "Poses"),
            _in("receptor", SlotKind.STRUCTURE, "Receptor docked into"),
        ),
        library="AutoDock Vina 1.2.7, Meeko",
        lane="chem-tools",
        surface=ToolSurface.VIEW,
        params=DockParams,
    ),
    ToolManifest(
        id="pose_interactions",
        version="1",
        name="Contacts a pose makes",
        area=ToolArea.STRUCTURE,
        description=(
            "Reads which residue each docked pose touches and how: hydrogen bonds, "
            "hydrophobic packing, van der Waals contact, salt bridges. This is the half "
            "of a docking result that can be checked against what is already known about "
            "a site, because a score can only be compared with another score while a "
            "contact can be held against the literature. A pose that scores well and "
            "never reaches the residue the site is named for is a pose to disbelieve. "
            "Contacts are geometric, so they say a pose is placed to make an interaction "
            "and not that it makes one. Needs a receptor with hydrogens, which is what "
            "preparing the structure produces."
        ),
        inputs=(
            _in("poses", SlotKind.POSES, "Poses"),
            _in("receptor", SlotKind.STRUCTURE, "Receptor"),
        ),
        outputs=(_in("interactions", SlotKind.INTERACTIONS, "Interactions"),),
        library="ProLIF",
        lane="chem-tools",
        surface=ToolSurface.VIEW,
        params=InteractionParams,
    ),
    ToolManifest(
        id="check_poses",
        version="1",
        name="Check poses are physically possible",
        area=ToolArea.STRUCTURE,
        description=(
            "Tests each docked pose for things no real complex does: bond lengths and "
            "angles off their bounds, aromatic rings bent, a conformation far more "
            "strained than the molecule's own low-energy shapes, atoms inside the "
            "protein, or the ligand drifted out of the site. A pose that fails is not "
            "a weak answer but a wrong one, however good its score, and this is the "
            "step that refuses it. Run it on every set of poses before reading their "
            "scores or contacts, since a scoring function cannot tell a clash from a "
            "tight fit. It says a pose is possible, not that it is right."
        ),
        inputs=(
            _in("poses", SlotKind.POSES, "Poses"),
            _in("receptor", SlotKind.STRUCTURE, "Receptor"),
        ),
        outputs=(
            _in("poses", SlotKind.POSES, "Poses that passed"),
            _in("receptor", SlotKind.STRUCTURE, "Receptor docked into"),
        ),
        library="PoseBusters",
        lane="chem-tools",
        surface=ToolSurface.VIEW,
        params=ValidityParams,
    ),
    ToolManifest(
        id="predict_admet",
        version="1",
        name="Predict ADMET",
        area=ToolArea.DATA,
        description=(
            "Predicts absorption, distribution, metabolism, excretion and toxicity "
            "endpoints per compound from structure alone, with the training "
            "distribution each one came from. Predictions, not measurements: they "
            "are useful for ranking what to make and for flagging a liability worth "
            "measuring, and they do not substitute for an assay."
        ),
        inputs=(_COMPOUNDS,),
        outputs=(_in("admet", SlotKind.ADMET_TABLE, "Predicted ADMET"),),
        library="admet-ai (Chemprop models trained on the Therapeutics Data Commons)",
        lane="gpu",
        surface=ToolSurface.VIEW,
    ),
    # -- Composites ------------------------------------------------------------
    # Sequences everyone runs together, offered as one choice so the planner has
    # fewer chances to get the order wrong. Each expands back into its steps before
    # anything executes, so the gates and the provenance inside survive.
    ToolManifest(
        id="build_sar_table",
        version="1",
        area=ToolArea.SAR,
        name="Build the SAR table",
        description=(
            "Decomposes the series against the approved core and resolves each "
            "substituent to the word a chemist would use. One step because nobody "
            "stops between splitting the molecules and naming the pieces: the names "
            "are how the table is read, not a separate result."
        ),
        inputs=(_COMPOUNDS, _CORE),
        outputs=(_in("table", SlotKind.RGROUP_TABLE, "SAR table"),),
        library="rdkit",
        expands_to=("decompose_rgroups", "name_substituents"),
        surface=ToolSurface.VIEW,
    ),
    ToolManifest(
        id="analyse_pairs",
        version="1",
        area=ToolArea.SAR,
        name="Analyse matched pairs",
        description=(
            "Finds the pairs differing in one position, summarises what each change "
            "is worth across every pair that made it, and picks out the jumps too "
            "large for one change to explain. The three are one piece of work, and "
            "the cliffs still pause for you, because some of them are assay errors."
        ),
        inputs=(_TABLE,),
        outputs=(
            _in("effects", SlotKind.EFFECTS, "Effect sizes"),
            _in("cliffs", SlotKind.PAIRS, "Cliffs, adjudicated"),
        ),
        library="rdkit, numpy",
        judgment=True,
        expands_to=("matched_pairs", "effect_sizes", "activity_cliffs"),
        surface=ToolSurface.EDITOR,
    ),
    ToolManifest(
        id="check_properties",
        version="1",
        area=ToolArea.DATA,
        name="Check the properties",
        description=(
            "Computes the physicochemical properties and tests whether activity is "
            "really tracking one of them rather than anything structural. Computing "
            "them and not checking is the common mistake, so they are one step."
        ),
        inputs=(_COMPOUNDS,),
        outputs=(
            _in("properties", SlotKind.PROPERTY_TABLE, "Properties"),
            _in("correlation", SlotKind.CORRELATION, "Correlation"),
        ),
        library="rdkit, numpy",
        expands_to=("property_profile", "property_confound"),
        surface=ToolSurface.VIEW,
    ),
)


def default_registry() -> ToolRegistry:
    return ToolRegistry(_TOOLS)
