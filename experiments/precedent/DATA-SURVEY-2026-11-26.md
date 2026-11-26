# Data for a second registration: survey of 2026-11-26

After `prereg-precedent-v1` ended void, the owner asked whether better data exist. This
records what was checked. The survey of public datasets was made by a research agent that
read the papers, dataset cards and release files; items it could not open are listed as
unverified at the end. Nothing here was run as an experiment.

## What the void run and its addendum say the data must have

The addendum to `RESULTS-prereg-precedent-v1.md` found that vote noise was not the cause
(about 86% of the variance of the Scruples judgement is not noise at five or more votes)
and that sentence embeddings explain 2.5% to 4% of it. So the need is not more votes on the
same kind of text. It is material where the judgement depends on factors stated in the
text, with many raters per item and, ideally, pairs that differ in one factor.

## Finding

No public dataset has all of: English text that the raters saw, 20 or more raters per item,
minimal pairs on consent, promise, coercion, emergency and impossibility, hundreds of items,
and a clear licence. The usable options are partial.

| Use | Dataset | Items | Raters per item | Gate factors | Licence | Catch |
|---|---|---|---|---|---|---|
| Gate-flip test | MoralExceptQA (Jin, Levine et al. 2022) | 148 vignettes in three families | about 50 to 100 | duress or threat, emergency or need, a prior promise, the purpose of the rule | CC BY-NC-SA 4.0 on the dataset card | Three template families, so the unit of resampling is the family or grid row. 42 of 66 line vignettes were shown with pictures. Well known, so a language model reading gates may have seen it |
| Gate-flip test, small | MoCa moral stories (Nie et al. 2023) | 62 stories on 34 base dilemmas | 25, votes released | causal role, personal force, evitability, beneficiary (annotated by two experts) | none on the repository | Famous trolley-type dilemmas. No consent, promise or coercion factor |
| Gate-flip test, small | CNI dilemma batteries (Gawronski et al. 2017; Körner et al. 2020) | 24 and 48 dilemmas on 18 base stories | about 160 to 200 | norm prohibits or prescribes, crossed with benefits against costs | none set | Few base stories |
| Consent pairs | Sommers 2020; Demaree-Cotton and Sommers 2022 | about two dozen scenarios; 24 cells | about 40 to 100 | refusal against deceived agreement, coercion, intoxication, capacity | article means; one OSF project with no licence | Means only for the first |
| Main test, no gates | Scruples Anecdotes, heavily voted posts | 849 posts with 100 or more votes, 1,738 with 50 or more | by threshold | none | repository Apache-2.0 | Same corpus as the void run. Votes are forum comments and not independent. The embedding result above applies to it |
| Main test, no gates | Dillion et al. 2023 compilation; Knutson et al. 2010 | 464 one-sentence acts; 312 vignettes | 30 to 100; 30 | none | mixed, some unstated | No factor structure |

On Atlas, `/archive/ethics-corpora` holds Scruples (anecdotes and dilemmas), the ETHICS
commonsense set, Moral Stories, Social Chemistry 101, the Moral Machine comparison data and
several labelled sets. None has many raters on minimal pairs: ETHICS has five relabellers
and binary labels, Scruples Dilemmas five annotators on ten-word titles, Social Chemistry
one worker for almost every rule, Moral Stories labels by construction.

## What follows

1. **The gate claim can be tested on existing data, at small scale.** MoralExceptQA gives
   human shares for vignettes that differ in duress, need and prior commitment, which are
   three of the six gates of `gates.json`. Pooled with the consent pairs and the
   impossibility vignettes of Chituc et al. 2016, it covers five. The test would be H1 of
   the registration (an explicit gate check against similarity alone, where one gate
   changes), resampled by template family. This is the test the first registration could
   not run.
2. **The metric claim has no adequate public test set.** A learned metric needs hundreds of
   independent items with low-noise graded judgements and stated factors. The closest is to
   collect ratings on generated minimal pairs, which needs ethics approval. That decision is
   the owner's.
3. **On Scruples, a second attempt would need a representation that reads the judgement**
   (for example features read from the text by a language model, registered as part of the
   instrument), and would still have no gate annotation to validate against.

## Unverified

Exact raters per vignette for all 148 MoralExceptQA items (only shares are released); the
licence tag on a MoCa mirror; rater counts for parts of the Dillion compilation; the
item-level format of the CNI files; whether Sommers 2020 has a data repository; several
vignette studies that were not opened (Cushman, Kneer and colleagues, Engelmann and
Waldmann). Most OSF projects named have no licence set, which means reuse terms are
unstated. A dataset announced in November 2026 (RULES, Levine et al.) has many raters and
justification tags but shows a photo with each scenario and had no data link.
