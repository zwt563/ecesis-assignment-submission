# Assignment 1 Summary Report

## Objective

The objective is to map PJMISO constraints across Market, Dayzer, and Panorama. The three files describe the same type of grid constraint, but they use different naming conventions and different table structures. I treat each constraint as a monitored facility plus a contingency and build a matched table that shows the closest Dayzer and Panorama expressions for each Market row.

## Data Used

- Market rows: 5,230
- Dayzer rows: 13,813
- Panorama rows: 21,963
- Output anchor: Market rows. The output has one row per Market constraint.

## Method

1. **Use Market as the anchor list.** Each Market row is treated as one facility-contingency pair. This makes the output directly satisfy the required fields: `market_constraint`, `dayzer_constraint`, and `pano_constraint`.
2. **Parse source-specific structures.** **Market already provides separate `CONSTRAINT` and `CONTINGENCY` columns. Dayzer stores most information in `NAME`, so I split the string at the first colon into a facility part and a contingency part. Panorama provides `Monitored Facility` and `Contingency Name`, and I also extract the short alias inside parentheses when available.** This step converts the three files into a more comparable structure. Although all three sources describe constraints, they do not use the same schema. Market is relatively direct because the monitored facility and the contingency are already stored in separate columns. Dayzer is less structured because a single `NAME` field may contain both pieces of information, such as `FALCONER_115 KV_FAL-WAR:ACTUAL`, where the text before the colon is the facility and the text after the colon is the contingency. Panorama is more explicit, but its `Monitored Facility` field often contains both a long facility description and a shorter alias in parentheses. The alias is useful because it may be closer to how the same facility is written in Market or Dayzer. After this parsing step, each source has comparable facility and contingency fields, which makes the later normalization and matching steps more reliable.
3. **Normalize the text.** I normalize case, separators, punctuation, spacing, voltage formatting, and common no-contingency labels. `BASE` and `ACTUAL` are treated as base-case or no-contingency labels. This step is needed because the same physical constraint can be written in many slightly different ways across the three sources. For example, one source may write `345 KV`, another may write `345KV`, and another may use underscores or hyphens between station names. These formatting differences do not change the physical meaning of the constraint, but they can reduce the text similarity score if they are not standardized first. I also group `BASE` and `ACTUAL` together because both are used to describe a non-contingency or base-case condition in these files. After normalization, the matching algorithm focuses more on meaningful differences such as facility name, voltage level, and contingency identity, rather than superficial formatting differences.
4. **Build weighted search strings.** Each source row is represented by normalized facility text plus repeated contingency text. The contingency is repeated because a constraint is defined by both the monitored facility and the contingency, and a different contingency can mean a different constraint. In this task, the facility name alone is not enough to identify a constraint. For example, the same monitored line may appear under several different outage conditions. These rows may look very similar because they share the same facility name, but they should not be treated as the same constraint if the contingency is different. Repeating the normalized contingency text gives it more influence in the TF-IDF representation, so the matching algorithm is less likely to match two rows only because their facility names are similar. This is a simple way to reflect the physical definition of a constraint as a `(facility, contingency)` pair.
5. **Apply character n-gram TF-IDF (Term Frequency–Inverse Document Frequency) matching.** I convert the weighted strings into character n-gram TF-IDF vectors and use cosine similarity to find the nearest Dayzer and Panorama candidate for each Market row. (Character n-gram TF-IDF represents each constraint name using weighted short character sequences, so names with similar abbreviations, spellings, or formatting still have similar vectors. For example, `NOTTINGH 230 KV NOTTINGHM 2-3 SER DEV` and `NOTTINGH_230 KV_2-3` are not exact string matches, but they share many character n-grams such as `NOT`, `OTT`, `230`, `KV`, and `2-3`, so character n-gram TF-IDF can still assign them a high similarity score.)
6. **Check contingency agreement.** After the nearest-neighbor match, I separately check whether the normalized contingency is an exact match. I also add a partial contingency check for complex Dayzer names, such as double-contingency strings where the main outage tokens appear inside a longer description. This check is important because the nearest-neighbor score is based on the full search string, so a high text similarity score does not always guarantee that the contingency is the same. Two rows may share the same facility name but refer to different outage conditions, and those should not be treated as the same constraint. The exact contingency check captures clean cases where the normalized contingency names are the same. The partial contingency check is used for cases where one source writes a more detailed or combined contingency name, for example a Dayzer name containing `DBL:` or multiple outage elements. In those cases, I do not want to mark the row as a clean exact match, but I also do not want to ignore useful overlap in the main outage tokens.
7. **Create source-side status fields.** Since the nearest-neighbor algorithm always returns one candidate, I add `dayzer_match_status` and `pano_match_status` to distinguish accepted matches, partial matches, reviewable candidates, and low-confidence candidates. This is needed because a nearest-neighbor method always gives the closest candidate even when there is no truly reliable match in that source. Without a separate status field, the output table may look like every Market row has a valid Dayzer and Panorama match, even though some candidates are only weak nearest matches. The status fields make this distinction explicit. For example, a row can have a strong Panorama match but a weak Dayzer candidate. In that case, the Panorama side can be marked as `matched`, while the Dayzer side can be marked as `unmatched_low_confidence` or `needs_review`. This makes the output easier to interpret and avoids overstating the reliability of every source-side match.
8. **Create a canonical ID.** I build `canonical_constraint_id` from Market `CONSTRAINTID` and `CONTINGENCYID`, because the assignment defines a constraint as a facility-contingency pair. Using the facility ID alone would merge multiple contingencies for the same facility. The purpose of this field is to give each mapped row a stable internal identifier. Since the Market file is used as the anchor table, the canonical ID is based on the Market-side facility and contingency identifiers. This is more appropriate than using only `CONSTRAINTID`, because the same monitored facility can appear under several different contingencies. For example, the same line may be constrained under outage A, outage B, or base-case conditions, and these should be kept as separate constraint records. The canonical ID therefore represents the full `(facility, contingency)` pair rather than only the monitored facility.
9. **Add audit checks.** I flag voltage mismatch, duplicate Dayzer/Panorama targets, interface-style constraints, Panorama validity windows, partial contingencies, and low-confidence source-side candidates. These checks are used to identify rows where the text match may need more caution. A voltage mismatch can indicate that two names refer to different equipment even if the station names look similar. Duplicate targets show cases where multiple Market rows map to the same Dayzer or Panorama row; this may be valid because the sources can use different levels of detail, but it can also indicate an ambiguous many-to-one match. Interface-style constraints are flagged because they may describe transfer paths or aggregate limits rather than a simple monitored facility. Panorama validity windows are kept as review information because Panorama includes date ranges while the other two sources do not provide directly comparable validity dates. These checks do not automatically mean the match is wrong; they provide reasons for lowering confidence or asking for manual review.
10. **Assign final confidence.** `match_quality` is the initial text-based confidence. `audited_match_quality` is the final recommended confidence after the audit checks. `review_reason` explains why a row is downgraded or requires manual review. I keep both confidence fields because they answer different questions. `match_quality` describes how strong the first-pass text match looks based on similarity scores and contingency agreement. `audited_match_quality` is more conservative because it also considers the warning flags from the audit checks. For example, a row may have a strong text match but still be downgraded if it shares the same target with many other rows or if the voltage information does not agree. The `review_reason` field makes this process transparent by listing the specific reasons behind the final label. This way, the output is not just a black-box match table; it also shows which rows can be used with higher confidence and which rows should be checked more carefully.

## Match Results

The output contains 5,230 matched rows, using the Market constraint list as the anchor table. Each row maps one Market constraint to the closest Dayzer and Panorama candidates.

### First-pass text matching

The first-pass match is based on normalized text similarity and contingency agreement.

- Output rows: 5,230
- Unique canonical IDs: 5,230
- Initial high / medium / review rows: 4,415 / 634 / 181
- Dayzer high / medium / review: 4,535 / 545 / 150
- Panorama high / medium / review: 5,011 / 184 / 35
- Dayzer exact contingency matches: 4,500 of 5,230
- Panorama exact contingency matches: 5,013 of 5,230
- Average Dayzer score: 0.8992
- Average Panorama score: 0.9479

### Final results after audit checks

The audit checks make the confidence labels more conservative by flagging partial matches, low-confidence candidates, duplicate targets, voltage mismatches, interface-style constraints, and Panorama validity-window issues.

- Final audited high / medium / review rows: 2,727 / 2,205 / 298
- Dayzer matched / partial / review / low-confidence candidates: 4,424 / 295 / 441 / 70
- Panorama matched / partial / review / low-confidence candidates: 4,972 / 90 / 149 / 19
- Rows sharing a Dayzer target with another Market row: 1,697
- Rows sharing a Panorama target with another Market row: 1,069
- Interface-style rows: 131
- Dayzer / Panorama voltage mismatches: 23 / 11

The first-pass results show that most constraints can be matched well by normalized text similarity. The final audited results are intentionally more cautious. I use `audited_match_quality` as the final confidence field, and `review_reason` to explain why a row needs more review.

## Insights

- Panorama is generally easier to match than Dayzer because Panorama separates monitored facility and contingency fields, while Dayzer often compresses them into one `NAME` field.
- Exact contingency agreement is very helpful. Many high-confidence rows have strong facility similarity and exact contingency agreement across sources.
- Partial contingency matching improves the treatment of complex Dayzer strings. It prevents some related double-contingency rows from being marked as complete mismatches, but it is still kept below exact matching.
- Duplicate target matches are common. They may reflect real source granularity differences, but they are also important ambiguity warnings.
- Interface and transfer constraints are harder than ordinary line constraints because they may describe aggregate transfer limits instead of a single monitored branch.

## Conclusions

- The notebook produces a complete Market-anchored matched table with 5,230 output rows.
- The required output columns are included: `market_constraint`, `dayzer_constraint`, and `pano_constraint`.
- The final recommended confidence field is `audited_match_quality`; `match_quality` is kept only as the initial text-based confidence.
- The `review_reason`, `dayzer_match_status`, and `pano_match_status` fields make ambiguous or partial matches explicit instead of hiding them.

## Limitations

- The method uses Market as the anchor list. It covers all Market constraints but does not build a full union table for Dayzer-only or Panorama-only constraints.
- The method is primarily text-based. It does not use topology, terminal buses, equipment IDs, or power-flow model information.
- Interface and transfer constraints may not follow the ordinary single-facility plus contingency structure, so those rows should be reviewed carefully.
- Duplicate target matches are not automatically wrong, but they may indicate many-to-one ambiguity.
- Panorama validity windows are used only as audit flags because Market and Dayzer do not provide directly comparable date fields.
- Some low-score rows may still be valid when one source uses an abbreviated name and another uses a longer descriptive name.

## Deliverables

- Matched CSV: `assignment1_matched_constraints_submission.csv`
- Notebook: `assignment1_constraint_mapping_submission.ipynb`
- Summary report: `assignment1_summary_report_submission.md`
