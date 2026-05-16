# CoVER-REL-Judge Explanation Examples

## yelpchi

- seed 42 node 225: verdict=real strength=strong key_relation=RUR explanation=RUR relation is open and shows benign-like prototype margin with high neighbor consistency, supporting realness.
- seed 42 node 1880: verdict=fake strength=strong key_relation=RUR explanation=RUR relation shows high feature deviation and low neighbor consistency, indicating potential fraud.
- seed 42 node 3845: verdict=fake strength=strong key_relation=RUR explanation=RUR relation shows high zscore outlier and benign prototype close, indicating potential fraud-like behavior.
- seed 42 node 4035: verdict=fake strength=strong key_relation=RUR explanation=RUR shows high feature deviation and low neighbor consistency, indicating potential fraud.
- seed 42 node 8467: verdict=real strength=strong key_relation=RUR explanation=RUR relation shows high neighbor consistency and benign-like margin with high z-score features and top10 degree, supporting realness.
- seed 42 node 8782: verdict=fake strength=strong key_relation=RUR explanation=RUR relation shows high zscore outlier and fraud prototype proximity, indicating potential fraud.

## amazon

- seed 42 node 129: verdict=uncertain strength=strong key_relation=UVU explanation=UVU relation shows benign-like prototype margin and medium deviation, but low neighbor consistency and open gate flags suggest potential structural anomalies.
- seed 42 node 180: verdict=fake strength=strong key_relation=UVU explanation=High z-score anomalies and low degree in UPU indicate potential fraud.
- seed 42 node 198: verdict=fake strength=strong key_relation=UVU explanation=UVU relation shows high zscore outlier and anomalous feature relation, indicating potential fraud.
- seed 42 node 328: verdict=fake strength=strong key_relation=UVU explanation=Multiple relations show close proximity to fraud prototypes, indicating potential falsity.
- seed 42 node 438: verdict=uncertain strength=strong key_relation=UVU explanation=UVU relation shows high degree and zscore outliers, but neighbor consistency is low and prototype margin is benign-like, indicating potential inconsistency.
- seed 42 node 573: verdict=fake strength=strong key_relation=USU explanation=USU shows high feature deviation and low neighbor consistency, indicating potential fraud.

