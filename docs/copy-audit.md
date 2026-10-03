# UI copy audit — 2026-10-03

Target voice: **academic and friendly**. Precise, plain, calm and scientifically exact, without chatty narration.
Scope: every string a user can see, in both the frontend and the backend. That covers labels, help text, empty states, errors, toasts,
chart labels, engine manifests, validation reasons, undefined-metric reasons, progress phases and stored run failures.

About 670 strings reviewed and **271 flagged**. Nothing has been applied yet.

---

## 1. Conventions (applied across every row below)

| Topic | Convention | Today |
|---|---|---|
| Spelling | **American** everywhere (color, catalog, normalize, generalize, labeled, centered, canceled, analog) | UI is mostly British today ("labelled", "normalised", "Centred", "Cancelled", "neighbour"); backend is American |
| Product nouns | Lowercase in running text ("this protocol", "the scorecard"). Capitalise only in titles and nav | Mixed: "this Protocol", "the Dataset", "Scorecard" |
| Dashes | No em-dash clause chains (`—`, `--`). Use two sentences, a colon or a semicolon. En dash only for ranges and "Bemis–Murcko" | About 40 em-dash chains. That is the single biggest "AI-written" tell |
| Headings | Noun phrases, not questions ("Train–test similarity", not "Is this split a real test?") | About 10 question or clause headings |
| Case | Sentence case for titles, labels, buttons, chart axes, table headers, chips | Chart axes, `<dt>` labels, the "recommended" chip and the partition filters are lowercase |
| Frontend load errors | `Could not load {thing}` with no period | 4 outliers |
| Backend messages | One sentence-case sentence ending in a period. It says what is wrong and, where possible, the next step. **No UUIDs, enum names, field names, env vars or class names** | UUIDs appear in every 404 via `NotFoundError`. `SplitStrategy.RANDOM`, `row_ids`, `upload_ref`, `ValueError:` and `make`/`uv` commands all reach users |
| Transport errors | `{Action} failed ({status})[: detail]` | "API error: N — detail", "Download failed: N", "Failed to load config" |
| Engine naming | `manifest.name` ("Chemprop D-MPNN") in phases and errors. Tasks are written "regression" and "binary classification" | `manifest.id` and `binary_classification` reach users |
| "Score" | **Predict** for applying a protocol ("Predict 120 compounds"). **Primary metric** for ranking. "Scorecard" stays as a product noun | "Score" is used for both meanings |
| Settings vs conditions | **Settings** in the UI. "conditions" stays the API term | Both appear in the UI |
| Similarity | "Tanimoto similarity to nearest training compound". "NN similarity" where space is tight | Five different phrasings |
| Scaffold | "Bemis–Murcko scaffold" on first mention, then "scaffold". Never "ring system" or "family" (only the empty-scaffold label says "no ring system") | Three phrasings, and "ring system" is factually wrong |
| Target | "target" / "target column". Types are "Continuous value" / "Active or inactive" | "Value to predict", "measured value", "measurements" |
| Dataset creation | Buttons and toasts say "Create". Descriptions say "immutable snapshot" | "freeze", "frozen" and "Immutable once created" |
| Baseline | The flagged engine is the **default baseline**, since the baseline is now user-selectable | Copy still describes a fixed "fingerprint baseline" |
| Partitions | "training / validation / test set" in prose | partition, split and set are mixed |
| Sweep flow verbs | "New sweep" (button and heading), then "Start sweep" (submit) | New, then Start a sweep, then Submit sweep |
| Abbreviations | "configuration", not "config" | — |
| Condition help | Do not restate the default ("500 is a good default"), because the form already shows it | — |

---

## 2. Factual errors (fix regardless of tone)

All of these were checked against the code.

| Where | Now says | Code does |
|---|---|---|
| `features/datasets/types/index.ts:107` | Binary target: "exactly two distinct values" | `prepare_frame.py:67` accepts only 0 or 1 |
| `validation-report-view.tsx:53` | "Compounds to train on" | Counts `valid − duplicates`, which includes the test set. Should say "Unique compounds" |
| `validation-report-view.tsx:94` | "Rejected rather than dropped quietly" | Invalid rows are filtered out and the file is still accepted. They are **excluded** |
| `dataset-profile-view.tsx:176` | Flags partitions "under 10% active" | Also flags those **above 90%** (line 160) |
| `dataset-profile-view.tsx:459` | "Scanned on a sample of N compounds" | `cliffs_sampled_from` is the **full** count (`build_profile.py:124`). The sample is smaller than N |
| `dataset-profile-view.tsx:226,235` | "within {t} Tanimoto" | That reads as a distance, but the threshold is a similarity of ≥ t |
| `dataset-profile-view.tsx:252`, `scorecard-view.tsx:130` | "share no ring system" | Bemis–Murcko scaffolds can share ring systems |
| `scorecard-diagnostics.tsx:272` | "The same model, scored on a random re-split" | It is a **separately trained** model, using the same engine and settings |
| `scorecard-view.tsx:135`, `protocol-list.tsx:25`, `engine-catalogue.tsx:48` | Baseline described as fixed ("fingerprint baseline", "what every other model is measured against") | The baseline is selectable |
| `scorecard-diagnostics.tsx:159` | "measurably higher", "The applicability domain is real" | No test is run. The code compares only the two end bins |
| `scorecard-view.tsx:82` | "No model … can honestly do better" | The noise floor is the mean replicate range, not a hard bound |
| `dataset-profile-view.tsx:136` | "not measuring the same population" | The check is a heuristic (> 1 SD) |
| `tanimoto_gp.py:134` | "the only engine whose uncertainty is a true posterior spread" | True for regression only |

## 3. Display bugs found along the way

| Where | Bug |
|---|---|
| `application/engines/manifest.py:69,80,81` | Renders "must be **a integer**" |
| `application/execution/retry_run.py:82` | `NotFoundError("Engine", str(error))` renders "Engine 'no baseline engine registered' not found" |
| `application/execution/train_protocol.py:350` | Renders "Engine 'baseline' not found" |
| `features/protocols/lib/verdict.ts:231` | Shows the raw metric key ("auroc") instead of `metricLabel()` |
| `features/runners/components/runner-list.tsx:166` | A literal `--` appears on screen |
| `dataset-detail.tsx:107`, `protocol-detail.tsx:107` | "high is better" should be "higher is better" |
| `manifest.py:98,107,117–121` | Shows condition **keys** and Python tuple/list reprs (`['a', 'b']`) instead of labels |
| `failure_message.py:22` | Prefixes stored run failures with the Python class name (`ValueError: …`) |
| `predict_with_protocol.py:543` | "has no results **yet**" is wrong when the run has failed |

---

## 4. Proposed rewrites

### 4a. Datasets and collections (`frontend/src/features/datasets`, `features/collections`)

| file:line | current | proposed |
|---|---|---|
| collections/collection-detail.tsx:60,70 | "CSV" / "SDF" | "Export CSV" / "Export SDF" |
| collections/collection-detail.tsx:77 | "Where these came from" | "Provenance" |
| collections/collection-detail.tsx:87 | "These values were predicted by a model, not measured. Anything that flows back into another app carries that mark until a human replaces it with a measurement." | "These values are model predictions, not measurements. They stay marked as predicted in other apps until replaced with measured data." |
| collections/collection-detail.tsx:96 | "The run these were triaged from" | "Source run" |
| collections/collection-list.tsx:22 | "A saved triage decision — the compounds you were willing to order, frozen with the run they came from and marked as AI-predicted." | "Compounds selected from a run's predictions, saved with their source run and marked as AI-predicted." |
| collections/collection-list.tsx:44 | "Run a published protocol, then pick the compounds worth pursuing from its results." | "Run a published protocol, then select compounds from its results." |
| datasets/compound-browser.tsx:54 | "Could not read this dataset's snapshot." | "Could not load compounds" |
| datasets/compound-browser.tsx:80 | "train" / "validation" / "test" | "Train" / "Validation" / "Test" |
| datasets/dataset-detail.tsx:85 | "What this predicts" | "Columns and target" |
| datasets/dataset-detail.tsx:90 | "Structures" | "Structure column" |
| datasets/dataset-detail.tsx:99 | "Measured value" / "Active / inactive" | "Continuous value" / "Active or inactive" |
| datasets/dataset-detail.tsx:107 | "· high is better" | "· higher is better" / "· lower is better" |
| datasets/dataset-detail.tsx:155 | "What the file contained" | "Validation report" |
| datasets/dataset-detail.tsx:165 | "Could not profile this dataset" | "Could not load the dataset profile" |
| datasets/dataset-list.tsx:47 | "A frozen snapshot of compounds and one measured value, with its split fixed. Immutable once created, so a protocol trained on it stays citable." | "An immutable snapshot of structures and one target, with a fixed split, so protocols trained on it are reproducible." |
| datasets/dataset-list.tsx:79 | "Upload a CSV of structures and measurements to get started." | "Upload a CSV of structures and target values." |
| datasets/validation-report-view.tsx:53 | "Compounds to train on" | "Unique compounds" |
| datasets/validation-report-view.tsx:74 | "Repeat measurements of the same compound disagreed by {x} on average." | "Mean range between replicate measurements of the same compound: {x}." |
| datasets/validation-report-view.tsx:79 | "That spread is free information: no model trained on this data can honestly beat it, so it becomes the floor your scorecard is measured against." | "Model error below this level is within experimental error. The scorecard reports it as the noise floor." |
| datasets/validation-report-view.tsx:94 | "Structures that did not parse, and target values that are empty, not a number, or not 0/1. Rejected rather than dropped quietly. Row numbers are positions in the file you uploaded." | "Unparseable structures, and target values that are empty, non-numeric, or not 0/1. These rows are excluded. Row numbers refer to the uploaded file." |
| datasets/validation-report-view.tsx:105 | "Why" | "Reason" |
| datasets/validation-report-view.tsx:127 | "{n} compound(s) labelled both ways" | "{n} compound(s) with conflicting labels" |
| datasets/validation-report-view.tsx:130 | "The same structure appears as active and inactive. That is a decision about your data, not one a majority vote should make silently — so the file is refused until you resolve it." | "Each structure is labeled both active and inactive. Conflicts are not resolved automatically; correct them and upload again." |
| datasets/types/index.ts:91 | "Test compounds have chemical scaffolds the model never trained on. This simulates asking the model about a new series — the situation you are usually in, and the harder score to earn." | "Test compounds have Bemis–Murcko scaffolds absent from training. This approximates prediction on a new chemical series and gives a conservative estimate." |
| datasets/types/index.ts:96 | "Test compounds are drawn at random, so close analogues of training compounds end up on both sides. Scores come out higher than the model will achieve prospectively." | "Compounds are assigned at random, so close analogs of training compounds appear in the test set. Scores typically overestimate prospective performance." |
| datasets/types/index.ts:102 | "A measured value" | "Continuous value" |
| datasets/types/index.ts:103 | "IC50, solubility, permeability — anything on a continuous scale." | "For example pIC50, log solubility or permeability." |
| datasets/types/index.ts:107 | "A two-class call. The column must hold exactly two distinct values." | "Values must be 0 (inactive) or 1 (active)." |
| datasets/lib/parse-csv.ts:29 (also runs/predict-wizard.tsx:118) | "No columns found. Is this a CSV with a header row?" | "No columns found. The file must be a CSV with a header row." |
| datasets/dataset-wizard.tsx:157 | "This file was not accepted" | "Validation failed" |
| datasets/dataset-wizard.tsx:159 | "Nothing was frozen. Fix the rows below and upload again — the report tells you exactly which ones and why." | "No dataset was created. Correct the rows listed below and upload the file again." |
| datasets/dataset-wizard.tsx:168 | "Back to the wizard" | "Return to setup" |
| datasets/dataset-wizard.tsx:197 | "One column of structures as SMILES, one column of the value you want to predict. Everything else is ignored." | "A SMILES column and a target column are required. Other columns are not used for training." |
| datasets/dataset-wizard.tsx:202 | "Not sure of the format?" | "Expected format" |
| datasets/dataset-wizard.tsx:230 | "Structures" | "Structure column (SMILES)" |
| datasets/dataset-wizard.tsx:248 | "Value to predict" | "Target column" |
| datasets/dataset-wizard.tsx:308 | "What kind of value is {column}?" | "Target type for {column}" |
| datasets/dataset-wizard.tsx:342 | "Carried onto every prediction, so a predicted value reads the same way a measured one does." | "Shown with every predicted value." |
| datasets/dataset-wizard.tsx:347 | "Better means" | "Preferred direction" |
| datasets/dataset-wizard.tsx:373 | "How should the test set be held out?" | "Split strategy" |
| datasets/dataset-wizard.tsx:390 | "recommended" | "Recommended" |
| datasets/dataset-profile-view.tsx:105 | "What you are predicting" | "Target distribution" |
| datasets/dataset-profile-view.tsx:108 | "The measured values, train against test. A narrow range makes an impressive-looking error meaningless, and a test partition sitting somewhere else in the range is a shift the model will pay for." | "Target values by partition. Read errors against the range; a shift between training and test sets lowers test performance." |
| datasets/dataset-profile-view.tsx:109 | "How the two classes fall across the partitions. A split that leaves the test set badly unbalanced makes MCC unstable, whatever the model does." | "Class balance in each partition. A strongly imbalanced test set makes MCC unstable." |
| datasets/dataset-profile-view.tsx:117 | "Range" | "Training-set range" |
| datasets/dataset-profile-view.tsx:124 | "Any error has to be read against this span — the same RMSE is excellent across six log units and meaningless across half of one." | "Interpret RMSE relative to this range: the same value is small across six log units but large across half of one." |
| datasets/dataset-profile-view.tsx:136 | "More than one standard deviation from the train median. The partitions are not measuring the same population." | "More than one training-set standard deviation from the training median, indicating a shift between partitions." |
| datasets/dataset-profile-view.tsx:137 | "In line with the train partition." | "Within one standard deviation of the training median." |
| datasets/dataset-profile-view.tsx:176 | "Bars are active (coloured) against inactive. A partition under 10% active is flagged: MCC and AUPRC both get unstable there, and the number that comes back will move a lot between seeds." | "Colored segment: active. Partitions below 10% or above 90% active are flagged; MCC and AUPRC vary substantially between seeds at that imbalance." |
| datasets/dataset-profile-view.tsx:207 | "Is this split a real test?" | "Train–test similarity" |
| datasets/dataset-profile-view.tsx:209 | "How far each test compound sits from the nearest thing the model will train on. This is the applicability question asked before training rather than after it — a test set that is close to the training set will produce a flattering score no matter which engine runs." | "Tanimoto similarity of each test compound to its nearest training compound: an applicability-domain check before training. A test set close to the training set inflates scores for any engine." |
| datasets/dataset-profile-view.tsx:219 | "Median similarity" | "Median NN similarity" |
| datasets/dataset-profile-view.tsx:226 | "Test compounds within {t} Tanimoto of the training set. The rest is extrapolation." | "Test compounds with NN similarity ≥ {t}. The rest lie outside the applicability domain." |
| datasets/dataset-profile-view.tsx:234 | "Test compounds at or above {t} Tanimoto to something in training. The model has effectively already seen these, and every metric is flattered by them." | "Test compounds with Tanimoto similarity ≥ {t} to a training compound. These are effectively seen in training and inflate every metric." |
| datasets/dataset-profile-view.tsx:235 | "No test compound is within {t} Tanimoto of a training compound." | "No test compound has Tanimoto similarity ≥ {t} to a training compound." |
| datasets/dataset-profile-view.tsx:245 | "Scaffolds on both sides" | "Scaffolds shared by training and test sets" |
| datasets/dataset-profile-view.tsx:251 | "{n} compounds share a scaffold across train and test, which a scaffold split is supposed to prevent." | "{n} compounds have a scaffold present in both training and test sets. A scaffold split should prevent this." |
| datasets/dataset-profile-view.tsx:252 | "Zero, which is what a scaffold split promises. Train and test share no ring system." | "As expected for a scaffold split: the training and test sets share no Bemis–Murcko scaffold." |
| datasets/dataset-profile-view.tsx:253 | "{n} compounds share a scaffold across train and test. Expected for a random split — and exactly what makes its scores optimistic." | "{n} compounds have a scaffold present in both training and test sets. Expected for a random split, and one reason its scores are optimistic." |
| datasets/dataset-profile-view.tsx:261 | "nearest-neighbour Tanimoto to the training set" | "Tanimoto similarity to nearest training compound" |
| datasets/dataset-profile-view.tsx:265 | "domain edge" | "Domain threshold" |
| datasets/dataset-profile-view.tsx:266 | "Mass piled to the right means the test set looks like the training set, and the benchmark is easier than it appears. Mass to the left means genuine extrapolation — a lower score there is worth more than a higher one on the right." | "Mass to the right indicates a test set similar to the training set, so the benchmark is easier than it appears. Mass to the left indicates extrapolation, where a lower score is more informative." |
| datasets/dataset-profile-view.tsx:286 | "Bemis-Murcko scaffolds. Hundreds of analogues of one core is a congeneric series: a model will interpolate across it beautifully and generalize nowhere. A long tail of singletons is a diverse deck, which is harder to fit and worth more when fitted." | "Bemis–Murcko scaffold distribution. A congeneric series is easy to interpolate within but says little about generalization. A set with many singleton scaffolds is harder to fit and more informative." |
| datasets/dataset-profile-view.tsx:299 | "Largest family" | "Most common scaffold" |
| datasets/dataset-profile-view.tsx:301 | "Share of the dataset sitting on its single most common scaffold." | "Fraction of compounds with the most common scaffold." |
| datasets/dataset-profile-view.tsx:304 | "One-off scaffolds" | "Singleton scaffolds" |
| datasets/dataset-profile-view.tsx:306 | "Scaffolds represented by exactly one compound. A high share means little for a model to generalize from within any family." | "Scaffolds with exactly one compound. A high fraction leaves little within-scaffold SAR to learn from." |
| datasets/dataset-profile-view.tsx:312 | "A curve that jumps to the top-left is a congeneric series; one that climbs gradually is a diverse deck." | "A steep initial rise indicates a congeneric series; a gradual rise, a diverse set." |
| datasets/dataset-profile-view.tsx:330 | "no ring system" | "Acyclic (no scaffold)" |
| datasets/dataset-profile-view.tsx:367 | "Where this dataset sits in property space, and whether one ordinary descriptor already explains the target." | "Descriptor distributions by partition, and how strongly each descriptor alone correlates with the target." |
| datasets/dataset-profile-view.tsx:382 | "That is most of the signal. A model that beats the baseline here has not yet shown it learned any chemistry beyond this one property — check it against the parity plot on the trained protocol before believing the headline." | "A strong monotonic relationship. Part of any model's apparent performance may come from this property alone; interpret scores with this in mind." |
| datasets/dataset-profile-view.tsx:384 | "A real but partial trend. Some of any model's score on this dataset is this property rather than chemistry." | "A moderate correlation. Part of any model's performance may reflect this property alone." |
| datasets/dataset-profile-view.tsx:385 | "Weak, which is good news: no single ordinary property explains this target, so a model has something genuine to learn." | "A weak correlation: no single descriptor explains the target." |
| datasets/dataset-profile-view.tsx:396 | "Sign is direction, length is strength. Descriptors that correlate negatively are exactly as informative as ones that correlate positively." | "Bar length shows strength; sign shows direction. Negative and positive correlations are equally informative." |
| datasets/dataset-profile-view.tsx:413 | "Train and test are drawn as shares of their own partitions, so the shapes stay comparable despite the partitions being very different sizes. Two distributions that barely overlap are a covariate shift the split introduced." | "Histograms are normalized within each partition, so training and test sets are comparable despite different sizes. Little overlap indicates covariate shift." |
| datasets/dataset-profile-view.tsx:443 | "No near-identical pair disagrees about the target among the sample scanned of {N} compounds. Nothing here puts an extra ceiling on what a model can achieve." | "No activity cliffs found (similar pairs with differing target values){, in a subsample of the N compounds}." |
| datasets/dataset-profile-view.tsx:459 | "Near-identical structures the assay disagrees about. Any featurization that maps these two molecules to nearly the same point cannot predict both — collectively they are a second ceiling on accuracy, alongside the assay noise floor. Scanned on a sample of {N} compounds." | "Close structural analogs with different measured values. A model that represents both almost identically cannot predict both, which limits accuracy alongside assay noise. Scanned on a subsample of the {N} compounds." |
| datasets/dataset-profile-view.tsx:494,498,505 | "similarity" / "differ by" / "assay noise" | "Similarity" / "Difference" / "Assay noise" |
| datasets/dataset-profile-view.tsx:516 | "A gap far larger than the assay noise floor is a real structure-activity relationship. A gap close to it is two measurements that disagree, which is a data question rather than a chemistry one." | "A difference well above the assay noise floor likely reflects a genuine SAR effect; one close to it may be measurement error." |

### 4b. Protocols, scorecard and engines (`frontend/src/features/protocols`, `features/engines`)

| file:line | current | proposed |
|---|---|---|
| engines/engine-catalogue.tsx:18 | "What a Protocol can be trained with. Every engine describes its own settings, so this list stays true without anyone updating it." | "Engines available for training protocols, with their settings." |
| engines/engine-catalogue.tsx:48 | "Baseline" | "Default baseline" |
| engines/condition-summary.tsx:24, protocols/condition-fields.tsx:46 | "No settings to configure." / "This engine has nothing to configure." | "This engine has no configurable settings." |
| protocols/protocol-detail.tsx:96 | "What it predicts" | "Predicted readouts" |
| protocols/protocol-detail.tsx:107 | "(low is better)" | "(lower is better)" / "(higher is better)" |
| protocols/protocol-detail.tsx:118 | "Could not load the Scorecard." | "Could not load scorecard" |
| protocols/protocol-detail.tsx:129 | "Publishing locks this protocol permanently. Its weights, dataset, split and metrics can never change again, which is what makes it citable — and it means this cannot be undone. Anyone in this workspace will be able to run it." | "Publishing permanently locks this protocol's weights, dataset, split and metrics so that it can be cited. Anyone in this workspace will be able to run it. This cannot be undone." |
| protocols/protocol-list.tsx:25 | "A trained model, scored against a fingerprint baseline. Drafts are yours alone; publishing locks one and makes it runnable by the whole workspace." | "Trained models, each scored against a baseline. Drafts are private; publishing locks a protocol and makes it available to the workspace." |
| protocols/protocol-runs.tsx:38 | "The training run behind this Protocol, and every prediction made with it." | "The training run for this protocol and all prediction runs that used it." |
| protocols/scorecard-diagnostics.tsx:54 | "Predicted probability against truth" | "Predicted probability by true class" |
| protocols/scorecard-diagnostics.tsx:58 | "Every test compound, its true class against the probability the model gave it. Well-separated clouds mean the model can rank; overlap in the middle is where the threshold decision actually costs something." | "Predicted probability of every test compound, split by true class. Clear separation means actives are ranked above inactives; overlap marks where any threshold will misclassify." |
| protocols/scorecard-diagnostics.tsx:59 | "Every test compound in the test set, not just the twenty worst. A cloud that hugs the diagonal is a working model; a cloud that flattens toward the middle is a model predicting the dataset average and scoring respectably for it." | "All test compounds. Points near the diagonal are accurate predictions; a cloud flattened toward the middle means predictions regress to the dataset mean." |
| protocols/scorecard-diagnostics.tsx:69 | "predicted probability of being active" | "Predicted P(active)" |
| protocols/scorecard-diagnostics.tsx:70 | "truly active" / "truly inactive" | "Active" / "Inactive" |
| protocols/scorecard-diagnostics.tsx:71 | "share of class" | "Fraction of class" |
| protocols/scorecard-diagnostics.tsx:73 | "Each class normalised to its own size, so the shapes stay comparable on an unbalanced test set. Two humps pushed to opposite ends is a model that separates the classes; overlap in the middle is the region where whatever threshold you pick will be wrong about something." | "Each class is normalized to its own size, so the shapes stay comparable when classes are imbalanced." |
| protocols/scorecard-diagnostics.tsx:82 | "The dashed line is a perfect prediction; the shaded band is the assay noise floor. A point inside the band is as accurate as this data can prove anything is." | "Dashed line: y = x. Shaded band: ± assay noise floor; points inside it are within experimental error. Color shows NN similarity to the training set." |
| protocols/scorecard-diagnostics.tsx:83 | "The dashed line is a perfect prediction. Points are shaded by how similar the compound is to the training set." | "Dashed line: y = x. Color shows NN similarity to the training set." |
| protocols/scorecard-diagnostics.tsx:98 | "Centred on zero means the model is wrong in both directions equally. Centred to one side is bias — a model that is systematically optimistic or pessimistic, which no error metric on this page reports." | "Centered on zero: no systematic bias. Shifted to one side: systematic over- or under-prediction, which RMSE and MAE do not reveal." |
| protocols/scorecard-diagnostics.tsx:114 | "On the diagonal, a predicted 0.8 means 80% of those compounds really were active — the probability can be read as one. Off it, the model may still rank compounds correctly, but its numbers are scores rather than probabilities. Dot size is how many compounds fell in each band." | "On the diagonal, probabilities are calibrated: of compounds predicted at 0.8, 80% are active. Off it, outputs are scores rather than probabilities, though ranking may still be correct. Dot size shows compounds per bin." |
| protocols/scorecard-diagnostics.tsx:142 | "Does distance from training predict error?" | "Error by similarity to training set" |
| protocols/scorecard-diagnostics.tsx:144 | "Mean absolute error against how similar each test compound is to its nearest training compound, in equal-sized groups. This is the applicability number on the verdict band, shown as evidence rather than asserted." | "MAE of test compounds, binned by NN similarity (equal-count bins). Shows whether the applicability-domain coverage above tracks error." |
| protocols/scorecard-diagnostics.tsx:152 | "nearest-neighbour Tanimoto to the training set" | "Tanimoto similarity to nearest training compound" |
| protocols/scorecard-diagnostics.tsx:154 | "Dot size is how many compounds are in each group; every group holds the same number, so a wobble at one end is not a small-sample artefact." | "Bins hold equal numbers of compounds, so no bin rests on fewer data than another." |
| protocols/scorecard-diagnostics.tsx:159 | "Error is {ratio}× / measurably higher on the least familiar compounds (a) than on the most familiar (b). The applicability domain is real for this model — treat predictions on unfamiliar chemistry as less trustworthy, in that proportion." | "MAE is {ratio}× (fallback: "higher") in the lowest-similarity bin (a) than in the highest (b). Predictions on compounds dissimilar to the training set are less reliable for this model." |
| protocols/scorecard-diagnostics.tsx:171 | "Error does not rise as compounds get less like the training set. Either the model generalizes past its training chemistry, or the similarity measure is not capturing what makes a compound hard here — in both cases the applicability percentage above is not the caveat it appears to be." | "MAE is not higher in the lowest-similarity bin than in the highest. Either the model generalizes beyond its training chemistry, or Tanimoto similarity does not capture what makes these compounds difficult. Either way, applicability-domain coverage is a weak guide to error here." |
| protocols/scorecard-diagnostics.tsx:199 | "Median absolute error per Murcko scaffold across the whole test set, worst first. The worst-predictions grid above shows twenty individual misses; this says which families they come from, which is the version you can act on." | "Median absolute error per Bemis–Murcko scaffold across the full test set, highest first. The individual largest errors are listed above." |
| protocols/scorecard-diagnostics.tsx:210 | "Only families with at least three test compounds appear — a median over one or two is not a median." | "Only scaffolds with at least three test compounds are shown." |
| protocols/scorecard-diagnostics.tsx:270 | "What an easier split would have said" | "Scaffold split versus random split" |
| protocols/scorecard-diagnostics.tsx:272 | "The same model, scored on a random re-split of the same rows. Every metric, not just the primary one — the gap is only convincing if the metrics agree about it." | "The same engine and settings, trained and scored on a random split of the same compounds. Consistent differences across metrics indicate split-induced optimism." |
| protocols/scorecard-view.tsx:61 | "{metric} was X on a random split and Y on the scaffold split it was actually scored on — the difference an easier split would have flattered it by." | "{metric} was X on a random split and Y on the scaffold split used for scoring. The difference is split-induced optimism." |
| protocols/scorecard-view.tsx:82 | "Repeat measurements of the same compound disagreed by this much. No model trained on this data can honestly do better." | "Mean range of replicate measurements of the same compound. Errors below this are within experimental error." |
| protocols/scorecard-view.tsx:88 | "Applicability" | "Applicability domain" |
| protocols/scorecard-view.tsx:98 | "of test compounds sit close enough to the training set for the model to have seen anything like them. The rest is extrapolation." | "of test compounds have NN similarity ≥ 0.3 to the training set. Predictions on the rest are extrapolations." |
| protocols/scorecard-view.tsx:127 | "scored on a random split — close analogues of training compounds are in the test set, so this reads high" | "scored on a random split: the test set contains close analogs of training compounds, so scores are likely optimistic" |
| protocols/scorecard-view.tsx:130 | "scored on a scaffold split — test compounds have ring systems the model never trained on" | "scored on a scaffold split: no test scaffold appears in the training set" |
| protocols/scorecard-view.tsx:135 | "You trained {engine_id}, which is what every other model here is measured against. There is nothing to compare it to — a comparison against itself would be a number that means nothing." | "The model and baseline are the same engine ({engine name}) with the same settings, so there is no comparison to report." |
| protocols/scorecard-view.tsx:141 | "The {metric} could not be computed on one side of the comparison, so no honest verdict is available. The reasons are in the metric table below." | "{metric} could not be computed for the model or the baseline, so no comparison is possible. See All metrics below for the reason." |
| protocols/scorecard-view.tsx:164 | "95% interval for this {metric}: [a, b] (bootstrap over the test set, unpaired)" | "95% bootstrap CI for {metric}: [a, b] (test set, unpaired)" |
| protocols/scorecard-view.tsx:171 | "The baseline's number sits inside that interval, so this test set cannot tell the two models apart." | "The baseline's {metric} lies within this interval, so the two models are not distinguishable on this test set." |
| protocols/scorecard-view.tsx:177 | "The margin is X, and repeat measurements of the same compound in this dataset disagree by Y. You cannot tell these two models apart with this data — treat them as equivalent and prefer the simpler one." | "The margin (X) is smaller than the assay noise floor (Y). The two models are indistinguishable on this data; prefer the simpler one." |
| protocols/scorecard-view.tsx:192 | "In published benchmarks a fingerprint baseline places mid-field against purpose-built models — a model that cannot beat one has not earned its complexity." | "Fingerprint baselines are competitive on many published benchmarks; a more complex model should outperform one to justify its complexity." |
| protocols/scorecard-view.tsx:222 | "Tune conditions against the validation column. The test column is the verdict: every time you retrain and read it, it becomes a little less of a held-out set, and the number it reports drifts upward for reasons that have nothing to do with the model." | "Tune settings against the validation column. Reserve the test column for the final comparison: each time it informs a choice, it becomes less of a held-out set and its estimate more optimistic." |
| protocols/scorecard-view.tsx:229 | "This run has no validation score — its split declared no validation partition, or it was trained before validation was measured. There is nothing here to tune against except the test column, which is the situation to avoid." | "No validation metrics: the split has no validation set, or the run predates validation scoring. Avoid tuning settings against the test column." |
| protocols/scorecard-view.tsx:243 | "tune here" | "For tuning" |
| protocols/scorecard-view.tsx:248 | "the verdict" | "Held out" |
| protocols/scorecard-view.tsx:284 | "is the baseline" | "Same as model" |
| protocols/scorecard-view.tsx:315 | "Where it fails" | "Largest prediction errors" |
| protocols/scorecard-view.tsx:317 | "The {n} worst predictions in the test set, grouped by Murcko scaffold. A cluster here is worth more than any aggregate score: it tells you which chemistry the model has not learned." | "The {n} test compounds with the largest absolute error, grouped by Bemis–Murcko scaffold. Clusters point to chemical series the model predicts poorly." |
| protocols/scorecard-view.tsx:354 | "off by" | "Abs. error" |
| protocols/scorecard-view.tsx:366 | "nearest train" | "NN similarity" |
| protocols/scorecard-view.tsx:426 | "How it was trained" | "Training settings" |
| protocols/scorecard-view.tsx:428 | "The resolved settings behind these numbers. Reproducing this Protocol means this engine, these conditions, and the Dataset it cites." | "Resolved settings for the model and baseline. Reproducing this protocol requires this engine, these settings and the cited dataset." |
| protocols/train-protocol-form.tsx:169 | "Lost track of this training run. It may still finish: look for it under Protocols." | "Could not retrieve the status of this training run. It may still complete; check Protocols." |
| protocols/train-protocol-form.tsx:234 | "Your chosen engine, the mandatory baseline, and — on a scaffold split — the same engine on a random split, so the optimism gap is measured rather than guessed." | "Training the selected engine and the baseline. On a scaffold split, the engine is also trained on a random split to measure the optimism gap." |
| protocols/train-protocol-form.tsx:248 | "A protocol is a trained model someone else can run. It is always scored against a baseline — one is chosen for you, and you can change it, but you cannot skip the comparison." | "A protocol is a trained model that others can run once it is published. Every protocol is scored against a baseline; a default is selected, and you can change it." |
| protocols/train-protocol-form.tsx:282 (also sweeps/sweep-form.tsx:116) | "Predicting {column} in {unit}, held out by {strategy} split." | "Target: {column} ({unit}); {strategy} split." |
| protocols/train-protocol-form.tsx:301,328 (also sweep-form.tsx:151,207) | "Pick a dataset first" | "Choose a dataset first" |
| protocols/train-protocol-form.tsx:307,334 (also sweep-form.tsx:157,214) | " · the baseline" | " · default baseline" |
| protocols/train-protocol-form.tsx:315 | "Compare against" | "Baseline" |
| protocols/train-protocol-form.tsx:341 | "This is the same engine with the same settings on both sides, so there is nothing to compare. Change a setting on one side, or pick a different engine to measure against." | "The model and baseline use the same engine and settings. Change a setting or choose a different baseline engine." |
| protocols/hooks/use-protocols.ts:99 | "Protocol published — anyone in this workspace can run it now" | "Protocol published. Anyone in this workspace can now run it." |
| protocols/lib/verdict.ts:78 | "Identical to the baseline" | "Matches the baseline score" |
| protocols/lib/verdict.ts:96 | "Ahead of the baseline, but by less than the assay noise" | "Better than the baseline, but within assay noise" |
| protocols/lib/verdict.ts:114 | "Ahead of the baseline, but within this test set's sampling noise" | "Better than the baseline, but within the 95% bootstrap CI" |
| protocols/lib/verdict.ts:123 | "Beats the baseline" | "Outperforms the baseline" |
| protocols/lib/verdict.ts:221 | "This model was trained on a random split, so there is no more optimistic split to compare it against." | "Not applicable: the model was scored on a random split." |
| protocols/lib/verdict.ts:231 | "The {metric} was undefined on one of the two splits." | "{metricLabel(metric)} is undefined on one of the two splits." |

### 4c. Runs, sweeps, runners and app shell

| file:line | current | proposed |
|---|---|---|
| app/(dashboard)/page.tsx:17 | "Upload data, train a protocol, read its scorecard, then run it across a compound set and triage what comes back." | "Upload a dataset, train a protocol, review its scorecard, then apply it to new compounds and triage the predictions." |
| app/auth/callback/workspace-selector.tsx:42 | "Entering {workspace}…" | "Opening {workspace}…" |
| app/auth/callback/workspace-selector.tsx:54 | "Select workspace to continue" | "Select a workspace to continue" |
| runs/predict-wizard.tsx:61 | "Trained with" | "Training settings" |
| runs/predict-wizard.tsx:131 | "Could not read that file" | "Could not read the file" |
| runs/predict-wizard.tsx:175 | "Score your own compounds with a published protocol. You do not need to know anything about the model underneath — it carries its own units." | "Predict properties for your compounds with a published protocol. Results are reported in the protocol's units." |
| runs/predict-wizard.tsx:198 | "Nothing is published yet. Train a protocol and publish it first." | "No published protocols. Train and publish a protocol first." |
| runs/predict-wizard.tsx:219 | "One column of SMILES. No measured values needed — that is what you are asking for." | "Requires one SMILES column. Measured values are not needed." |
| runs/predict-wizard.tsx:272 | "Carried beside every prediction and into the export, so results join back to your file." | "Included with each prediction and in the export, so results can be matched to your file." |
| runs/predict-wizard.tsx:287 | "Score {n} compound(s)" | "Predict {n} compound(s)" |
| runs/run-detail.tsx:43 | "Waiting for a runner that serves the "{lane}" lane. None is online right now." | "Waiting for a runner on the "{lane}" lane. None is currently online." |
| runs/run-detail.tsx:144 | "Scored {x} of {y} uploaded rows · {z} did not parse as structures" | "Predicted {x} of {y} uploaded rows · {z} could not be parsed as structures" |
| runs/run-detail.tsx:194 | "These compounds had already been scored by this protocol — these results came from cache, not a new run." | "These compounds were previously predicted with this protocol. Results were loaded from the cache; no new run was started." |
| runs/run-detail.tsx:208 | "Its Scorecard is on the Protocol page." | "Its scorecard is on the protocol page." |
| runs/run-detail.tsx:218 | "It produced no Protocol." | "No protocol was produced." |
| runs/run-detail.tsx:245 | "A collection is a frozen copy of these rows, marked as AI-predicted. It keeps its own snapshot, so it survives whatever happens to this run." | "A collection stores a fixed copy of these rows, labeled AI-predicted. It is stored independently of this run." |
| runs/run-list.tsx:31 | "A published protocol scoring a set of compounds. Training runs live on their protocol instead." | "Prediction runs apply a published protocol to a set of compounds. Training runs are listed on their protocol." |
| runs/triage-grid.tsx:114 | "Row in your uploaded file, not counting the header" | "Row number in the uploaded file, excluding the header" |
| runs/triage-grid.tsx:227 | "In domain only" | "Within applicability domain" |
| runs/triage-grid.tsx:237 | "{x} of your {n} are outside the domain of applicability" | "{x} of {n} selected compounds are outside the applicability domain" |
| runs/hooks/use-runs.ts:94 | "Run queued again" | "Run requeued" |
| sweeps/sweep-detail.tsx:95 | "A workspace runs 10 at a time, so this sweep finishes in waves." | "Each workspace runs at most 10 runs concurrently; the rest are queued." |
| sweeps/sweep-detail.tsx:111 | "Config" | "Configuration" |
| sweeps/sweep-detail.tsx:114 | "Score" | "Primary metric" |
| sweeps/sweep-detail.tsx:115 | "vs baseline" | "Improvement over baseline" |
| sweeps/sweep-form.tsx:88 | "Start a sweep" | "New sweep" |
| sweeps/sweep-form.tsx:90 | "Compare engines and conditions on the same dataset in one submission, ranked by score." | "Train several engine configurations on one dataset and rank them by primary metric." |
| sweeps/sweep-form.tsx:181 | "Config {n}" | "Configuration {n}" |
| sweeps/sweep-form.tsx:188 | aria "Remove config" | aria "Remove configuration {n}" |
| sweeps/sweep-form.tsx:240 | "Add config" | "Add configuration" |
| sweeps/sweep-form.tsx:253 | "Submit sweep" | "Start sweep" |
| sweeps/sweep-list.tsx:52 | "Compare engines and conditions on the same dataset, ranked by score." | "Compare engine configurations on one dataset, ranked by primary metric." |
| runners/new-runner-dialog.tsx:118 | "Register a machine to run training on your own hardware. You'll get a one-time token to start it with." | "Register a machine to run training on your own hardware. A one-time token is issued to start it." |
| runners/new-runner-dialog.tsx:148 | "Which queues this runner should pick up work from." | "The queues this runner accepts jobs from." |
| runners/new-runner-dialog.tsx:174 | "This token is shown once. Treat it like a password — if you lose it, revoke this runner and register a new one." | "This token is shown only once. Store it securely. If it is lost, revoke this runner and register a new one." |
| runners/new-runner-dialog.tsx:187 | "STUDIO_URL above is the API's address as the runner machine must reach it. It is filled in with the address this browser uses; if the runner machine can't reach that (a local address, a firewall), replace it with one it can." | "STUDIO_URL is set to the API address this browser uses. If the runner machine cannot reach it (for example, a local address or a firewall), replace it with one that it can." |
| runners/new-runner-dialog.tsx:193 | "CI does not publish the GPU image: build it on that machine with make image-runner-gpu." | "The GPU image is not published by CI. Build it on the runner machine with make image-runner-gpu." |
| runners/new-runner-dialog.tsx:206 | "I've copied it" | "I have copied the command" |
| runners/runner-list.tsx:67 | "Machines that run training on your own hardware, outside the hosted queue." | "Self-hosted machines that run training outside the hosted queue." |
| runners/runner-list.tsx:166 | "This runner's token stops working immediately. It will not pick up new work, and cannot be reconnected -- register a new runner if you need this hardware again." | "The runner's token is invalidated immediately, and the runner will not accept new jobs. A revoked runner cannot be reconnected. To use this machine again, register a new runner." |
| shared/components/chemistry/structure-thumbnail.tsx:107 | "no structure" | "Cannot render" |
| shared/components/charts/charts.tsx:243 | "similarity to training set" | "Tanimoto similarity to nearest training compound" |
| shared/lib/api/custom-instance.ts:158 | "Your session expired; signing you back in" | "Session expired. Signing in again…" |
| shared/lib/api/custom-instance.ts:162 | "API error: {status} — {detail}" | "Request failed ({status}): {detail}" |
| shared/lib/api/download.ts:57 | "Download failed: {status}" | "Download failed ({status})" |
| shared/lib/navigation.ts:54 | "Catalog" | "Catalog" |
| shared/providers/auth-provider.tsx:33 | "Failed to load config" | "Could not load configuration" |

### 4d. Backend text that reaches the UI (`backend/src/daikonstudio`)

| file:line | current | proposed |
|---|---|---|
| domain/shared/errors.py:47 | "{Entity} '{id}' not found" | "{Entity} not found." (keep the id as an attribute, not in the message) |
| domain/shared/errors.py:70 | "Concurrency conflict on {Entity} '{id}': entity was modified by another transaction" | "This {entity} was changed by another request. Reload and try again." |
| application/auth.py:45 | "Requires {minimum} role or higher" | "This action requires the {minimum} role or higher." |
| application/data/assign_split.py:178 | "(acyclic/invalid -- no shared scaffold)" | "(acyclic or unparseable: no ring system)" |
| application/data/assign_split.py:182 | "Scaffold split cannot honor the requested fractions on this dataset: … The largest scaffold family … Use SplitStrategy.RANDOM instead, or add more chemically diverse compounds to this dataset." | "A scaffold split cannot meet the requested fractions for this dataset: the {sets} set would be empty. The largest Bemis–Murcko scaffold ('{s}') covers {share} of {n} compounds and cannot be divided between sets. Use a random split, or add structurally diverse compounds." |
| application/data/create_collection.py:99 | "Run '{id}' is not ready (status: '{status}'); only a ready run's results can be saved into a Collection" | "Only a completed run's results can be saved to a collection; this run is {status}." |
| application/data/create_collection.py:106 | "row_ids must not be empty" | "Select at least one compound." |
| application/data/create_collection.py:108 | "row_ids must not be negative" | "Row indices must be non-negative." |
| application/data/create_collection.py:110 | "row_ids must not contain duplicates" | "Each compound can be selected only once." |
| application/data/create_collection.py:127 | "row_ids out of range for this run's results: [..]" | "Some selected rows are outside this run's results: {a, b}." |
| application/data/create_dataset.py:104 | "This name is reserved for a column the pipeline itself writes downstream … -- using it as a target would let that column silently overwrite …" | "The application writes a column with this name to prediction results and exports. Rename the column in your file. Reserved names: …" |
| application/data/create_dataset.py:116 (also predict_with_protocol.py:183) | "upload_ref is not a valid upload reference" | "The upload reference is invalid. Upload the file again." |
| application/data/create_dataset.py:135 | "Column(s) not present in the uploaded file: …" | "Columns not found in the uploaded file: …." |
| application/data/create_dataset.py:158 | "No usable rows in the uploaded file: every row failed structure or target validation -- see the report" | "The uploaded file has no usable rows. Every row failed structure or target validation; see the validation report for reasons." |
| application/data/create_dataset.py:276 | "The '{p}' partition has only one distinct target {kind} after splitting, which would train or score a maximally confident but meaningless model" | "After splitting, every compound in the {p} set has the same target {kind}. A model trained or evaluated on it would not be meaningful." |
| application/data/create_dataset.py:280 | "… Use a different split seed, SplitStrategy.RANDOM instead of a scaffold split, or add more diverse compounds/measurements …" | "All rows in the {p} set share the same '{col}' value. Use a different split seed or a random split, or add compounds with more varied measurements." |
| application/data/get_dataset_profile.py:104 (also get_dataset_compounds.py:80) | NotFoundError("Dataset snapshot") | NotFoundError("Stored dataset file") |
| application/data/export_collection.py:147 | "Cannot export as {fmt}: rendering these readouts would produce duplicate column/tag name(s): [..]" | "Cannot export as {FMT}: these readouts would produce duplicate column names: a, b." |
| application/data/export_collection.py:159 | NotFoundError("Collection snapshot") | NotFoundError("Stored collection file") |
| application/data/prepare_frame.py:68 | "binary target must be 0 or 1, got '{raw}'" | "Binary target must be 0 or 1 (found '{raw}')" |
| application/data/prepare_frame.py:75 | "target is not a number: '{raw}'" | "Target value is not numeric: '{raw}'" |
| application/data/prepare_frame.py:81 | "empty target value" | "Missing target value" |
| application/data/prepare_frame.py:109 | "invalid structure" | "SMILES could not be parsed" |
| application/engines/manifest.py:69,80,81 | "{key} must be a {type}, got {value!r}" | "{label} must be a/an {type} (received {value})." |
| application/engines/manifest.py:98 | "unknown conditions for {id}: [..]" | "{name} does not accept these settings: a, b." |
| application/engines/manifest.py:107 | "{key} is required for {id}" | "{label} is required for {name}." |
| application/engines/manifest.py:117,119 | "{key} below minimum {m}" / "{key} above maximum {m}" | "{label} must be at least {m}." / "{label} must be at most {m}." |
| application/engines/manifest.py:121 | "{key} must be one of ('a', 'b')" | "{label} must be one of: a, b." |
| application/engines/registry.py:48 | UnknownEngineError(engine_id) | "Engine '{id}' is not available on this server." |
| application/engines/registry.py:56 | "no baseline engine registered" | "No baseline engine is configured on this server." |
| application/engines/registry.py:59 | "multiple baseline engines registered: [..]" | "More than one baseline engine is configured: a, b." |
| application/execution/failure_message.py:22 | "{ExcClass}: {message}" | "{message}" (needs a check: some engine ValueErrors may rely on the class name for context) |
| application/execution/failure_message.py:24 | "Unexpected {Exc}; the traceback is in the log of the runner that ran it" | "The run failed with an unexpected internal error ({Exc}). Details are in the runner log." |
| application/execution/predict_with_protocol.py:175 | "Protocol '{id}' is not published; only a published Protocol can be run" | "Only a published protocol can be used for prediction. Publish this protocol first." |
| application/execution/predict_with_protocol.py:283,305 | "Column '{c}' not present in the uploaded file: available columns: …" | "Column '{c}' is not in the uploaded file. Available columns: …." |
| application/execution/predict_with_protocol.py:295 | "No valid structures in the uploaded file" | "No SMILES in the uploaded file could be parsed." |
| application/execution/predict_with_protocol.py:543 | "Run '{id}' has no results yet (status: '{s}')" | "Results are available only for completed runs; this run is {s}." |
| application/execution/result_view.py:106 | "Cannot {verb} '{column}'" | "Cannot {verb} '{column}': this column is not in the results." |
| application/execution/retry_run.py:82 | NotFoundError("Engine", str(error)) | NotFoundError("Engine") |
| application/execution/sweeps.py:127 | "A sweep needs at least one config" | "A sweep requires at least one configuration." |
| application/execution/sweeps.py:131 | "A sweep is limited to {MAX} configs; got {n}" | "A sweep can contain at most {MAX} configurations ({n} submitted)." |
| application/execution/train_protocol.py:350 | NotFoundError("Engine", id or "baseline") | NotFoundError("Baseline engine", id) |
| application/execution/train_protocol.py:446,461 | "Engine '{id}' cannot train a {binary_classification} model; it supports …" | "{name} does not support binary classification. Supported tasks: regression." (the baseline variant starts "The baseline engine …") |
| application/execution/train_protocol.py:495 | "training baseline" | "Training baseline model" |
| application/execution/train_protocol.py:649 | "training random-split comparison" | "Training on a random split for comparison" |
| application/execution/train_protocol.py:713 (also chemprop_dmpnn.py:309, molformer_xl.py:410) | "training {id}" | "Training {name}" |
| application/execution/train_protocol.py:797 | "exceeded the {n}s job deadline; raise STUDIO_WORKER_JOB_TIMEOUT, or this lane's entry in STUDIO_WORKER_JOB_TIMEOUT_BY_LANE, if the work is legitimate" | "The run exceeded its {n} s time limit and was stopped. An administrator can raise the limit (STUDIO_WORKER_JOB_TIMEOUT)." |
| application/execution/train_protocol.py:839 | "every row in the test split has the same '{c}' value, so this metric has no defined value -- add positives (or negatives) …" | "Undefined: all test-set compounds have the same '{c}' value. Add compounds of the missing class, or use a different split." |
| application/execution/train_protocol.py:845 | "every row in the training split has the same '{c}' value, so the model only ever learned one class …" | "Undefined: all training-set compounds have the same '{c}' value, so the model learned only one class." |
| application/execution/train_protocol.py:851 | "the engine reported this metric as undefined" | "Undefined: the engine returned no value for this metric." |
| application/execution/train_protocol.py:877 | "this Dataset predates the structure_column migration … -- re-upload it to train on it" | "This dataset was created before its structure column was recorded and cannot be trained on. Upload it again." |
| application/execution/train_protocol.py:884 | "Dataset '{id}' records its structures in column '{c}', which cannot be used: {cause}" | "The structure column '{c}' is missing from this dataset's stored data. Available columns: …." |
| domain/catalog/protocol.py:92 | "Protocol '{id}' is already published" | "This protocol is already published." |
| domain/data/dataset.py:78 | "This data, split this way, is already stored in this workspace" | "A dataset with identical data and split already exists in this workspace." |
| domain/data/dataset.py:80 | "Datasets are content-addressed over the split snapshot … Use dataset {id}, or change the data or the split." | "Open the existing dataset, or change the data or the split settings." |
| domain/data/split.py:26,28 | "split fractions must sum to 1.0" / "… must be non-negative" | "Split fractions must sum to 1." / "Split fractions must be non-negative." |
| domain/execution/run.py:241 | "Cannot retry run '{id}' in status '{s}'" | "Only a failed or canceled run can be retried; this run is {s}." |
| domain/execution/run.py:267 | "Cannot cancel run '{id}' in terminal status '{s}'" | "This run has already ended ({s}) and cannot be canceled." |
| infrastructure/engines/_scoring.py:210 | "This model was fitted against a different set of molecular descriptors than this worker computes (…) -- most likely RDKit was upgraded … predicting through the mismatch would silently read every descriptor as the wrong one." | "This model was trained on a different RDKit descriptor set ({a} descriptors) from the one this runner computes ({b}), probably because RDKit was upgraded. Retrain the protocol before predicting." |
| infrastructure/engines/chemprop_dmpnn.py:62 | "How many passes over the training set. More epochs fit the training data more closely, at growing risk of memorising it. 50 is a good default." | "Number of passes over the training set. More epochs fit the training data more closely, with increasing risk of overfitting." |
| infrastructure/engines/chemprop_dmpnn.py:67 | "Message passing steps" | "Message-passing steps" |
| infrastructure/engines/chemprop_dmpnn.py:72 | "How far information travels across the molecule. Each step lets an atom see one bond further. 3 covers most local chemistry." | "Number of message-passing iterations. Each extends an atom's receptive field by one bond; 3 captures most local chemical environments." |
| infrastructure/engines/chemprop_dmpnn.py:77,82 | "Hidden size" / "How much the network can represent about each atom. Larger needs more data to be worth it." | "Hidden dimension" / "Dimension of the learned message vectors. Larger values need more training data to be beneficial." |
| infrastructure/engines/chemprop_dmpnn.py:92 (also molformer_xl.py:105) | "How many molecules are scored before the weights update. Lower it if training runs out of GPU memory." | "Number of molecules per gradient update. Reduce it if training runs out of GPU memory." |
| infrastructure/engines/chemprop_dmpnn.py:102 | "…pretrained on ~1M PubChem molecules against classical descriptors…" | "…pretrained on ~1M PubChem molecules to predict Mordred descriptors…" (check against the CheMeleon paper) |
| infrastructure/engines/chemprop_dmpnn.py:128 (also molformer_xl.py:148) | "The chemprop-dmpnn engine needs the 'gpu' extra … `make seed-runners` … `uv sync --extra gpu`." | "This runner does not have the GPU dependencies that Chemprop D-MPNN requires. An administrator can register a runner for the 'gpu' lane on the Runners page." |
| infrastructure/engines/descriptors_xgboost.py:45 | "… -- size, lipophilicity … -- … This is the configuration behind most published ADMET leaderboard results, and it is usually the strongest option …" | "Gradient-boosted trees on RDKit 2D descriptors (size, lipophilicity, polarity, topology and charge) instead of a hashed fingerprint. A strong, widely used configuration for ADMET endpoints, particularly on datasets of a few hundred to a few thousand compounds." |
| infrastructure/engines/ecfp4_lightgbm.py:47 | "… Grows trees towards whichever split helps most … natively -- usually the fastest engine here on large datasets." | "Morgan fingerprints with leaf-wise gradient boosting. Trees grow toward the split with the largest gain rather than to a fixed depth, and sparse fingerprint input is handled natively. Usually the fastest engine on large datasets." |
| infrastructure/engines/ecfp4_randomforest.py:50 | "How many decision trees to average over. More trees give steadier predictions but take longer to train. 500 is a good default." | "Number of decision trees in the ensemble. More trees give more stable predictions but take longer to train." |
| infrastructure/engines/ecfp4_xgboost.py:29 | "Morgan fingerprints with gradient-boosted trees. Often sharper than the random forest baseline, at the cost of being more sensitive to its settings." | "Morgan fingerprints with gradient-boosted trees. Often more accurate than the random-forest baseline, but more sensitive to hyperparameters." |
| infrastructure/engines/molformer_xl.py:79 | "…Fine-tunes on your data in minutes to hours on a GPU; freeze it to train just the output layer in a fraction of the time." | "…Fine-tuning takes minutes to hours on a GPU; freezing the encoder trains only the output layer, in a fraction of the time." |
| infrastructure/engines/molformer_xl.py:94 | "… 10 is usually enough and more risks overwriting what the model already knows." | "Number of passes over the training set. A pretrained transformer adapts quickly; longer training risks degrading the pretrained representation." |
| infrastructure/engines/molformer_xl.py:115 | "How far the weights move per update. Fine-tuning … wants a small value (around 0.00003); raise it towards 0.001 when the encoder is frozen, since only the output layer is learning." | "Step size of each weight update. Fine-tuning a pretrained transformer needs a small value (about 3 × 10⁻⁵); increase it toward 10⁻³ when the encoder is frozen, because only the output layer is trained." |
| infrastructure/engines/molformer_xl.py:126 | "…fine-tuning the whole model usually wins once there are thousands of measurements." | "…fine-tuning the whole model usually performs better with thousands of measurements." |
| infrastructure/engines/tanimoto_gp.py:134 | "A Gaussian process over structural similarity … -- where most in-house assays sit -- and the only engine here whose uncertainty is a true posterior spread … comfortable to about 5,000 and will refuse a training set above 10,000." | "A Gaussian process with a Tanimoto kernel on ECFP4 fingerprints, suited to small datasets (a few hundred to a few thousand compounds). For regression, it is the only engine here whose uncertainty is a posterior standard deviation in the target's units rather than an ensemble proxy. Cost scales cubically with training-set size: practical up to about 5,000 compounds, with a hard limit of 10,000." |
| infrastructure/engines/tanimoto_gp.py:151 | "How many extra times to re-tune the model's signal and noise scales from a fresh starting point …" | "Number of additional hyperparameter optimizations from random starting points. More restarts take proportionally longer but are less likely to end in a poor local optimum. 0 uses the initial values only." |
| infrastructure/engines/tanimoto_gp.py:179 | "A Gaussian process cannot be fitted to {n} training compounds -- the method holds a similarity matrix … this one is for the small-data regime." | "The Tanimoto Gaussian process accepts at most {max} training compounds; this training set has {n}. Use a tree-based or graph engine for datasets of this size." |
| infrastructure/persistence/sqlalchemy/data/repository.py:79 | "This data is already stored in this workspace" / "A concurrent upload of identical data won the race." | "A dataset with identical data and split already exists in this workspace." / "It was created by a concurrent upload. Refresh the dataset list." |
| infrastructure/persistence/sqlalchemy/execution/queue.py:24 | "runner lease expired after N attempts" | "No runner completed this run after N attempts; the runner stopped responding." |
| infrastructure/runner/agent.py:115 | "the runner gave up after {n}s: the fit never returned" | "The job did not finish within {n} s and was stopped by the runner." |
| interface/dependencies/_core.py:24 | "Duar auth is not configured" | "Authentication is not configured on this server." |
| interface/error_handlers.py:97 | "Something went wrong on the server" | "An unexpected server error occurred. Include the request ID if you report it." |

---

## 5. Knock-on work when applied

- **Tests that match on copy** need updating: `sweep-form.test.tsx` ("Submit sweep"), `custom-instance.test.ts:64` and `use-protocols.test.tsx:31–38` ("API error"), plus backend tests asserting on `NotFoundError` text, `manifest.py` messages and `prepare_frame` reasons.
- `manifest.py` messages need each spec's `label`, and errors need `manifest.name`. That is a small code change, not just a string swap.
- `scorecard-view.tsx:98` would hard-code 0.3. Either expose `applicability_threshold` on the scorecard response or accept the duplication, which already exists between `build_scorecard.py` and `build_profile.py`.
- Removing ids from `NotFoundError` messages loses some debuggability. The request ID in logs covers it.
