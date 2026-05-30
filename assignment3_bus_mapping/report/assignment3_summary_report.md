# Assignment 3 Summary Report

## Objective

The objective is to map each Dayzer bus to a corresponding Panorama bus. I treat this as a network-alignment problem instead of only a fuzzy-name matching problem.

## My idea

The assignment asks for a bus-to-bus mapping between Dayzer and Panorama. At first glance, this may look like a string matching task, because both datasets have bus names. However, I do not want to treat it as only a name matching problem.

In power grid models, two buses can have different names but still represent the same or a very similar electrical location. On the other hand, two buses can have similar names but refer to different voltage levels, different bus sections, or different sides of a transformer. Also, Dayzer and Panorama may not model the grid at exactly the same level of detail. One source may split a station into several bus sections, while the other source may merge some of them into fewer modeled buses.

Because of this, I treat this assignment as a network alignment problem rather than a pure string matching problem.

### Step:

I use several steps because each signal captures a different part of the matching problem.

1. **Build raw bus-branch graphs for Dayzer and Panorama.**In this step, each bus is treated as a graph node and each branch is treated as an edge. This allows me to use the network structure instead of only bus names. If two buses are truly corresponding nodes, their local neighborhoods should often look similar.
2. **Clean and normalize bus names.**Bus names may use different abbreviations, suffixes, voltage tokens, or formatting conventions. Normalization makes names more comparable before calculating string similarity.
3. **Create station-kV group proxies.**This step handles possible model granularity differences. Instead of forcing each raw Dayzer bus to immediately match one raw Panorama bus, I first compare higher-level station-voltage groups. This is useful when one source has several bus sections and the other source has fewer modeled buses for the same station and voltage.
4. **Generate candidate matches using broad but reasonable filters.**I do not compare every possible pair blindly. I first keep candidates with compatible voltage levels and reasonable evidence from name similarity, station similarity, geography, or topology. This improves both accuracy and interpretability.
5. **Score candidate group matches using multiple signals.**I combine several features:

   - station/name similarity,
   - voltage consistency,
   - geographic proximity,
   - graph degree similarity,
   - neighbor voltage profile similarity,
   - branch type profile similarity,
   - matched-neighbor topology consistency.

   The final group score is based on both attribute evidence and topology evidence:

   $$
   Score = 0.6 \times AttributeScore + 0.4 \times TopologyScore
   $$
6. **Use topology as validation, not only as a feature.**After initial high-confidence matches are found, I check whether already-matched neighbors also line up in the other graph. If a Dayzer bus has neighbors that map to neighbors of a Panorama bus, that is strong evidence that the match is structurally reasonable.
7. **Refine group-level matches back to the required bus-level output.**The assignment requires columns like `dayzer_bus` and `pano_bus`, so the final CSV still reports a bus-level result. However, I also keep extra columns such as group IDs, mapping level, confidence, score margin, and review flags to show whether the match is exact or still ambiguous within a group.
8. **Keep ambiguous cases instead of forcing false precision.**
   If multiple Panorama candidates are very close in score, or if a station-kV group contains multiple raw buses, I mark the case as ambiguous or needing review. I think this is more honest than forcing every Dayzer bus into a single one-to-one match when the provided data may not support that level of certainty.

Overall, my approach is:

$$
Raw\ Bus\ Graph
\rightarrow Station\text{-}kV\ Group\ Proxy
\rightarrow Group\text{-}Level\ Matching
\rightarrow Bus\text{-}Level\ Output
\rightarrow Review\ Flags
$$

This method is still not a perfect industrial production system, because I do not have official equipment IDs, breaker/switch status, or a known ground-truth mapping. However, it uses the main signals available in the assignment and tries to avoid the main risk of a simple fuzzy-name approach.

### Use a station-kV group proxy

Since the input files do not include breaker or switch status, I cannot exactly reconstruct the real-time electrical bus or true topology bus. For example, if two bus sections are connected through a closed breaker, they may behave as one topology bus. If the breaker is open, they should remain separate. This information is not available in the provided files.

To handle this limitation, I create a conservative **station-kV group proxy**. This is not a guaranteed physical electrical bus. It is an artificial grouping used for matching.

The idea is:

$$
Group(s, k) = \{ b : station(b) = s,\; kV(b) = k \}
$$

where \(s\) is the normalized station name and \(k\) is the nominal voltage level.

In other words, buses with the same station-like name and the same voltage level are grouped together first. This helps reduce false one-to-one assumptions when the two data sources split or merge station-level buses differently.

I do **not** collapse transformer ends into one group. A 345 kV bus and a 138 kV bus connected by a transformer are still different buses in a power-flow model because they have different voltage levels and the transformer has impedance. Transformer connections are used as topology information, but not as evidence that two buses should be merged.

## Results

Total Dayzer buses after deduplication: **10,558**

### Mapping level counts

| mapping_level          | count |
| :--------------------- | ----: |
| exact_bus              |  7178 |
| ambiguous_within_group |  3153 |
| unmatched              |   227 |

Mapping level counts

I use `mapping_level` to show how confident the algorithm is about the *level* of the mapping. Since the two data sources may split or merge buses differently inside the same station, I do not want to force every Dayzer bus into a perfect one-to-one Panorama bus match.

- `exact_bus` means I can provide a specific Panorama bus as the best match. These are the cleanest bus-level results.
- `ambiguous_within_group` means the algorithm can identify the likely Panorama station-kV group, but the exact bus inside that group is still not fully clear. I keep these rows because they are still useful, but they should be reviewed if exact bus-level accuracy is required.
- `unmatched` means the algorithm did not find a reliable candidate.

The result has 7,178 `exact_bus` mappings, 3,153 `ambiguous_within_group` mappings, and 227 `unmatched` buses. I think this is reasonable for this assignment because the problem is open-ended and the data does not include breaker/switch status or official bus IDs. In this situation, marking uncertain cases is better than forcing a match that may be wrong.

### Confidence counts

| confidence     | count |
| :------------- | ----: |
| matched_high   |  3216 |
| ambiguous      |  3030 |
| matched_medium |  2795 |
| matched_low    |  1290 |
| unmatched      |   227 |

Confidence counts

I use `confidence` to describe how reliable each suggested match is. This is not the same as `mapping_level`. The mapping level tells whether the result is an exact bus-level match or a group-level/ambiguous result. The confidence label tells how strong the evidence is.

The confidence is based on the final matching score and the margin between the best and second-best candidates:

$$
Margin = Score_{best} - Score_{second\ best}
$$

If the best score is high and clearly better than the second-best score, I label it as `matched_high`. If the score is acceptable but not very strong, I label it as `matched_medium` or `matched_low`. If two or more candidates have very similar scores, I label the row as `ambiguous`.

The categories mean:

- `matched_high`: strong automatic match
- `matched_medium`: reasonable match, but less certain
- `matched_low`: weak match, should be reviewed
- `ambiguous`: multiple candidates are close, so I do not force one answer
- `unmatched`: no reliable candidate found

The result has 3,216 high-confidence matches, 2,795 medium-confidence matches, 1,290 low-confidence matches, 3,030 ambiguous cases, and 227 unmatched buses. I think the number of ambiguous cases is reasonable because the task does not provide official bus IDs or breaker/switch status. In this situation, it is better to mark uncertain cases than to overstate the precision of the mapping.

## Insights

The high-confidence matches usually have consistent station names, the same voltage level, reasonable geographic distance, and similar local network neighborhoods. The topology information is useful because some buses have different naming conventions across the two sources, but their connected neighbors still look similar.

I also found that many cases should not be forced into an exact one-to-one bus match. Some are better interpreted as group-level or ambiguous-within-group matches. This is expected because the two sources may use different topology-processing or station-bus splitting conventions.

## Limitations

This method is still not an authoritative mapping. The data does not include breaker/switch status, CIM identifiers, ISO object IDs, or manually verified ground truth. Because of that, the station-kV group is only a practical proxy for an electrical/topological bus, not a guaranteed real-time electrical bus.

Transformer high-side and low-side buses are not collapsed because they are different voltage-level nodes in the power-flow model. They are used as topology evidence instead.

## Conclusion

The submitted CSV provides a proposed Panorama bus for each Dayzer bus, together with confidence scores and review flags. I think this is a more careful approach than forcing every bus into a single exact match, because it keeps the uncertain cases visible for review.
