# Fourth Quadrant Opening Audit

The two built-in Fourth Quadrant identities are sampled literal replay
prefixes for module version `1.32.7`. Each contains the submitted actions for
days 0 through 3 inclusive, exactly 96 primitive turns. They are not a
generalized Fourth Quadrant policy.

## Cross-replay audit

The available local replays were compared by canonical action dictionaries,
separately for the seat occupied by Fourth Quadrant:

| seat | replays | d0-d3 identical | first divergence | routing divergence |
| --- | --- | --- | --- | --- |
| 0 | 104670896, 104672565, 104675874, 104679205 | no | d0h1 market; d0h3 hands; d0h8 farmer | yes |
| 1 | 104671727, 104675042 | no | d0h1 market; d0h4 hands; d0h11 farmer | yes |

The first differences are market-order variation, but both seats later differ
in worker/farm routing. No generalized logic was inferred.

## Canonical traces

| identity | canonical source | source seed | source seat | source replay SHA-256 | content digest |
| --- | --- | ---: | ---: | --- | --- |
| `fourth_quadrant_s0` | episode 104670896 | 144368101 | 0 | `71eff635ee2ce8c95bf80e95665169bea4839b23d8cc2a6b6aaf7c73c9286f20` | `55ab1e931d83b9bc07bf52ab517f9f0fd3d2cc1f26156a07174f9a04e0973f9a` |
| `fourth_quadrant_s1` | episode 104671727 | 752669175 | 1 | `6071dad82af38ccbd82fd7dffea10182c634ab7e3bc91b7156bb128f76632ffa` | `3304cc35a91c3986be4f224d45f0efe9f19e7ab567199b2e5f584f86204945ed` |

The raw replay files remain local sample inputs and are not part of the
opening-book artifact.
