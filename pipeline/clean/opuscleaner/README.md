# OpusCleaner

This folder contains tools for integrating [OpusCleaner](https://github.com/hplt-project/OpusCleaner), a data cleaning tool. It can be configured to run a variety of steps to help clean messy data. For instance, it can remove sentence pairs that do not have matching numbers on the source and target sentence. It's not helpful to translate between a source sentence with $10.32 USD and a target of $12.45 EUR, as the model will learn different things.

Different datasets have different levels of quality, and different languages have different issues, so each dataset can be cleaned with a different set of a rules. A default config is provided at [configs/default.filters.json](pipeline/clean/opuscleaner/configs/default.filters.json). In order to determine new rules to apply, load up [OpusCleaner](https://github.com/hplt-project/OpusCleaner)
