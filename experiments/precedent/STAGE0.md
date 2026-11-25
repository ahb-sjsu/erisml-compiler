# Stage 0 record: prereg-precedent-v1

**Sealed on 2026-11-25 after a second pilot.** The owner said on 2026-11-25 to seal as drafted and
proceed as recommended. The first gate pilot showed the instrument was not fit to seal, so it was
revised and piloted again before sealing. No held-out judgement was read at any point before the seal.

## What Stage 0 did (2026-11-25, on Atlas)

| Step | Result | File |
|---|---|---|
| Count units, with the split fixed by a hash of the situation id | AITA titles: 23,385 judged situations on the case-base side, 5,821 held out. Dear Abby titles: 9,250 and 2,321 | `stage0/counts_*.json` |
| Check the judgement scale on the case-base side | 261 distinct values. About two thirds of situations round to 0, variance 0.36 | same |
| Choose cases by hash and embed them with LaBSE at the pinned snapshot | 6,000 training, 1,500 ordinary held-out, 1,500 Dear Abby. Texts only | `stage0/cases_aita_titles.sha256.json` |
| Look for human-rated single-gate flips | None found in seven published datasets. H1 cannot be run without newly rated items | `prereg-precedent-v1.md` Section 0 |
| Pilot the gate reader on 300 case-base titles with two hosted models | **Failed.** See below | `stage0/pilot1_agreement.json` |

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

## What changed

1. **Gate registry v2** (`gates.json`): "no consent" and "coercion" were narrowed, each with a
   sentence saying what does not count.
2. **Corpus.** One-line titles do not carry the gates. Two alternatives were checked.
   - Dear Abby full letters (`sqnd-probe/dear_abby_data/raw_da_qs.csv`, median 717 characters)
     join to Social-Chem by title for 4,977 situations. The Social-Chem judgements were made on the
     title alone, so the judges never saw what the gates would be read from. Not used.
   - **Scruples Anecdotes** (27,766 training posts, median about 1,500 characters in the pilot
     sample) carry community votes on the full post. Used.

## The second pilot (gate registry v2, 300 full Scruples training posts)

Same two readers, all 300 read by both (`stage0/pilot2_agreement.json`).

| Gate | Found by gpt-oss | Found by gemma | Kappa | Passes the rule |
|---|---|---|---|---|
| commitment | 27.3% | 24.7% | 0.64 | yes |
| abuse | 5.0% | 3.3% | 0.63 | yes |
| coercion | 7.0% | 6.0% | 0.59 | no |
| emergency | 3.0% | 1.7% | 0.56 | no |
| no consent | 23.3% | 16.7% | 0.55 | no |
| impossibility | 5.0% | 3.0% | 0.48 | no |

Two gates pass, both narrowly. The registration was rewritten for this corpus and these two gates and
sealed. Agreement is measured again on all 5,500 cases at the run, and the run is void if either gate
falls below 0.6 there.

The gate the book leans on most, consent, did not reach the bar in either pilot.

## After the seal

- Embedding and gate reading of all 5,500 cases (texts only). The gate reading takes many hours at
  the hosted models' speed.
- `run_precedent.py` from the sealed files.
- The owner's spot-check of 200 gate readings.
