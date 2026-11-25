# Stage 0 record: prereg-precedent-v1

**Nothing is sealed.** The owner said on 2026-11-25 to seal as drafted and proceed as recommended.
Stage 0 then showed that the gate instrument was not fit to seal, so it was not. No held-out judgement
has been read at any point.

## What Stage 0 did (2026-11-25, on Atlas)

| Step | Result | File |
|---|---|---|
| Count units, with the split fixed by a hash of the situation id | AITA titles: 23,385 judged situations on the case-base side, 5,821 held out. Dear Abby titles: 9,250 and 2,321 | `stage0/counts_*.json` |
| Check the judgement scale on the case-base side | 261 distinct values. About two thirds of situations round to 0, variance 0.36 | same |
| Choose cases by hash and embed them with LaBSE at the pinned snapshot | 6,000 training, 1,500 ordinary held-out, 1,500 Dear Abby. Texts only | `stage0/cases.sha256.json` |
| Look for human-rated single-gate flips | None found in seven published datasets. H1 cannot be run without newly rated items | `prereg-precedent-v1.md` Section 0 |
| Pilot the gate reader on 300 case-base titles with two hosted models | **Failed.** See below | `stage0/pilot_agreement.json` |

## The pilot that failed (gate registry v1, one-line AITA titles)

Readers: `openai/gpt-oss-120b` and `google/gemma-4-31B-it-qat-w4a16-ct`, 300 titles, all read by both.
The rule written before the second reader's results were seen: a gate enters the strata when the
readers agree at kappa of at least 0.6 and the primary reader finds it in at least 2% of texts.

| Gate | Found by gpt-oss | Found by gemma | Kappa | Passes the rule |
|---|---|---|---|---|
| no consent | 26.0% | 15.7% | 0.47 | no |
| commitment | 3.3% | 4.3% | 0.68 | yes |
| coercion | 2.7% | 3.0% | 0.46 | no |
| emergency | 1.3% | 0.7% | 0.66 | no (too rare) |
| abuse | 1.0% | 1.3% | 0.57 | no |
| impossibility | 0.0% | 0.7% | 0.00 | no |

One gate passes. The registration voids a run with fewer than two (V2). Sealing it would have sealed
a run that voids itself.

**Why it failed.** The disagreements on "no consent" were titles such as "quitting my job
indefinitely" and "leaving my boyfriend who has mental health issues". The v1 wording ("materially
affects another person who has not agreed") fits almost any act. A one-line title also rarely states
a promise, an emergency or abuse. With 300 texts and gates this rare, each kappa rests on 2 to 36
jointly positive cases, so the figures are rough.

The full reading of 9,000 cases had started and was stopped after 18 cases.

## What changed, and what is running

1. **Gate registry v2** (`gates.json`): "no consent" and "coercion" are narrowed, each with a
   sentence saying what does not count.
2. **Corpus.** One-line titles do not carry the gates. Two alternatives were checked.
   - Dear Abby full letters (`sqnd-probe/dear_abby_data/raw_da_qs.csv`, median 717 characters)
     join to Social-Chem by title for 4,977 situations. The Social-Chem judgements were made on the
     title alone, so the judges never saw what the gates would be read from. Not used.
   - **Scruples Anecdotes** (27,766 training posts, median about 1,500 characters in the pilot
     sample) carry community votes on the full post. The judgement would be the share of votes
     finding the author in the wrong. The dataset's own test split would be the held-out set.
3. **Pilot 2** is running on Atlas (screen `precedent-pilot2`): registry v2, 300 full training posts,
   the same two readers. Its result decides whether the registration is rewritten for Scruples
   Anecdotes and sealed, or whether the gate reader needs more work.

## What the registration needs before it can be sealed

- A pilot in which at least two gates pass the rule.
- Section 2 (data) and the case-building scripts rewritten for the corpus that passes.
- A rehearsal of the final code.
- The seal file, a signed commit and a signed tag, pushed before `run_precedent.py` is run.
