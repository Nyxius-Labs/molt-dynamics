# Exposure definition used in the code

Implemented in `analysis/s04_exposure.py` (`exposure_cells`, `static_cox_rows`), which reproduces
`DiffusionModeler.compute_exposures` in `src/molt_dynamics/rq2_diffusion.py` (verified cell-for-cell
against a direct port of that loop on 43 cascades).

## Network

A directed, static graph over every agent that sent or received at least one interaction in the window.
There is an edge `u -> v` if `u` wrote a top-level comment on a post by `v`, or replied to a comment by `v`
(`analysis/graphs.py::build_edges`, the rule in `dataset_loader._extract_interactions`). Self-interactions
are dropped. The graph pools the whole window: an edge exists if the interaction happened at any time,
before or after the cascade. Edge weights (interaction counts) are not used for exposure.

## Adoption

Agent `a` adopts cascade `c` at `t_a(c)`, the timestamp of its first post or comment containing the
cascade's n-gram (meme), identical code block (skill) or formatting pattern (behavioral).

## What counts as one exposure

For a focal agent `v` and cascade `c`, each **distinct in-neighbour** `u` of `v` (an agent that has
commented on or replied to `v`) that adopted `c` counts as one exposure. A neighbour counts once,
however many times it used the n-gram and however many interactions it has with `v`.

## Over what window

* If `v` adopted `c`: only neighbours that adopted **strictly before** `v` (`t_u(c) < t_v(c)`). Ties do not count.
* If `v` never adopted `c`: **every** adopting neighbour over the whole observation window counts, with no time limit.
* Units of analysis: every node of the graph x every cascade. Agents with no interactions are not in the
  risk set, and adopters outside the graph are dropped.

## Repeated exposures from the same source

A neighbour is counted at most once per cascade. Repeated adoptions, repeated posts and multiple
interactions between the same pair are ignored.

## Consequences

The rule is asymmetric. Adopters are credited with exposures up to their adoption time, while non-adopters
are credited with exposures from the whole window. With a static graph this mechanically makes
non-adopters look more exposed, which biases the logistic quadratic term and the static Cox hazard
ratio towards "saturating" (HR < 1). The edge direction means that exposure flows from commenters to the
author they commented on, not from authors to their readers.

## Variants reported alongside

* **Time-varying Cox** (`tv_cox_rows`): cumulative count of distinct in-neighbours that have adopted by
  time `t`, with the clock starting at the cascade seed. Every graph node is at risk until it adopts or until
  the cascade's last adoption. This removes the asymmetry.
* **Reader direction** (logistic sensitivity): `v` is exposed by agents whose content `v` commented on
  (out-neighbours), i.e. content `v` plausibly read.
