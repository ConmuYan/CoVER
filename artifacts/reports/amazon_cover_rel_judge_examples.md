# Amazon CoVER-REL-Judge Examples

## Seed 123

- node 834: verdict=uncertain strength=moderate key_relation=USU explanation=USU has low degree and high zscore outliers, indicating potential inconsistency.
- node 944: verdict=fake strength=moderate key_relation=USU explanation=Multiple relations show low neighbor consistency, suggesting potential fraud.
- node 1187: verdict=uncertain strength=moderate key_relation=UVU explanation=UVU relation shows low neighbor consistency and benign margin, with zscore outliers, suggesting potential fraud signals.
- node 1613: verdict=fake strength=strong key_relation=UVU explanation=UVU relation shows high degree and zscore outliers indicating potential fraud.
- node 1800: verdict=uncertain strength=strong key_relation=UVU explanation=UVU relation shows high neighbor consistency and benign prototype margin, with fraud prototype close tokens indicating potential anomaly.

## Seed 456

- node 153: verdict=fake strength=strong key_relation=UVU explanation=Low neighbor consistency across all relations indicates potential structural anomalies.
- node 248: verdict=fake strength=strong key_relation=UPU explanation=High zscore outliers and neighbor inconsistency in UPU and USU relations indicate potential fraud.
- node 399: verdict=uncertain strength=moderate key_relation=UVU explanation=UVU has high degree but low neighbor consistency and benign margin, suggesting potential anomaly.
- node 541: verdict=fake strength=strong key_relation=UVU explanation=High zscore outliers and fraud prototype proximity in USU and UVU relations indicate potential fraud.
- node 642: verdict=fake strength=moderate key_relation=USU explanation=USU relation shows fraud prototype close and low neighbor consistency, suggesting potential fraud.

## Seed 789

- node 341: verdict=uncertain strength=strong key_relation=UVU explanation=UVU relation shows high degree and zscore outliers, but neighbor consistency is low and prototype margin is benign-like, indicating potential inconsistency.
- node 680: verdict=fake strength=moderate key_relation=USU explanation=USU relation shows fraud prototype closeness and high zscore outliers, suggesting potential fraud.
- node 743: verdict=fake strength=strong key_relation=UVU explanation=UVU relation shows low degree and neighbor consistency with high zscore outliers, indicating structural inconsistency.
- node 919: verdict=fake strength=strong key_relation=UVU explanation=High z-score anomalies and low degree in UPU and USU suggest structural inconsistencies.
- node 932: verdict=real strength=strong key_relation=UVU explanation=UVU relation shows high neighbor consistency and benign-like prototype margin, with multiple zscore high indicators, supporting realness.

