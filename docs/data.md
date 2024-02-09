---
layout: default
title: Datasets
nav_order: 4
---

# Dataset importers

Dataset importers can be used in the `datasets` sections of the [training config](https://github.com/mozilla/firefox-translations-training/tree/main/configs/config.test.yml).

Example:
```
  train:
    - opus_ada83/v1
    - mtdata_newstest2014_ruen
```

Use the [utils/find-corpus.py](https://github.com/mozilla/firefox-translations-training/tree/main/pipeline/utils/find-corpus.py) tool to find all of the datasets for an importer and get them formatted to use in config. First, set up a local [poetry](https://python-poetry.org/) environment. Then run:

```
make install-utils
poetry run utils/find-corpus.py en ru opus
poetry run utils/find-corpus.py en ru mtdata
poetry run utils/find-corpus.py en ru sacrebleu
```

Make sure to check licenses of the datasets before using them.


## [OPUS](https://opus.nlpl.eu/) - Parallel Corpora

OPUS (open parallel corpora) is a preferred website for finding datasets. It is packed with information on the sizes and provenance of datasets. It easily lets you filter for available information. Here is an example of available [Russian datasets](https://opus.nlpl.eu/results/en&ru/corpus-result-table).

Example dataset keys:

 - `opus_NLLB/v1`
 - `opus_OpenSubtitles/v2018`
 - `opus_WikiMatrix/v1`
 - `opus_TED2020/v1`

```
make install-utils
poetry run utils/find_corpus.py --importer opus en ru
```


## [MTData](https://github.com/thammegowda/mtdata) - Parallel Corpora

This is a python tool that supports my different datasets.

https://github.com/thammegowda/mtdata

Example dataset keys:

 - `mtdata_ELRC-wikipedia_health-1-eng-rus`
 - `mtdata_Facebook-wikimatrix-1-eng-rus`
 - `mtdata_LinguaTools-wikititles-2014-eng-rus`
 - `mtdata_Neulab-tedtalks_dev-1-eng-rus`

To view all available datasets for this specific importer, run:

```
make install-utils
poetry run utils/find_corpus.py --importer mtdata en ru
```

# [sacreBLEU](https://github.com/mjpost/sacrebleu)

sacreBLEU provides evaluation datasets with a focus on comparable and shareable BLEU scores. They should be used as test sets in the `datasets.test` config section and the `datasets.devtest` sections (with no overlap between each set).

https://github.com/mjpost/sacrebleu

Example dataset keys:

 - `sacrebleu_mtedx/test`
 - `sacrebleu_mtedx/valid`
 - `sacrebleu_wmt14`
 - `sacrebleu_wmt14/full`
 - `sacrebleu_wmt15`

[Flores](https://github.com/facebookresearch/flores)
flores
dev, devtest
corpus
Evaluation dataset from Facebook that supports 100 languages.

:pca; parallel
local-corpus
/tmp/test-corpus
corpus
Local parallel dataset that is already downloaded. The dataset name is an absolute path prefix without ".lang.gz"

[Paracrawl](https://paracrawl.eu/)
paracrawl-mono
paracrawl8
mono
Datasets that are crawled from the web. Only [mono datasets](https://paracrawl.eu/index.php/moredata) are used in this importer. Parallel corpus is available using opus importer.

[News crawl](http://data.statmt.org/news-crawl)
news-crawl
news.2019
mono
Some news monolingual datasets from [WMT21](https://www.statmt.org/wmt21/translation-task.html)

[Common crawl](https://commoncrawl.org/)
commoncrawl
wmt16
mono
Huge web crawl datasets. The links are posted on [WMT21](https://www.statmt.org/wmt21/translation-task.html)

GCS bucket
bucket
bucket/path/en-ca/dataset
mono/corpus
Local monolingual dataset that is already downloaded. The dataset name is an absolute path prefix without ".lang.gz"

Local mono
local-mono
/tmp/test-mono
mono
Local monolingual dataset that is already downloaded. The dataset name is an absolute path prefix without ".lang.gz"


||||||

Data source | Prefix | Name examples | Type | Comments
--- | --- | --- | ---| ---
[MTData](https://github.com/thammegowda/mtdata) | mtdata | newstest2017_ruen | corpus | Supports many datasets. Run `mtdata list -l ru-en` to see datasets for a specific language pair.
[OPUS](opus.nlpl.eu/) | opus | ParaCrawl/v7.1 | corpus | Many open source datasets. Go to the website, choose a language pair, check links under Moses column to see what names and version is used in a link.
[SacreBLEU](https://github.com/mjpost/sacrebleu) | sacrebleu | wmt20 | corpus | Official evaluation datasets available in SacreBLEU tool. Recommended to use in `datasets:test` config section. Look up supported datasets and language pairs in `sacrebleu.dataset` python module.
[Flores](https://github.com/facebookresearch/flores) | flores | dev, devtest | corpus | Evaluation dataset from Facebook that supports 100 languages.
:pca; parallel | local-corpus | /tmp/test-corpus | corpus | Local parallel dataset that is already downloaded. The dataset name is an absolute path prefix without ".lang.gz"
[Paracrawl](https://paracrawl.eu/) | paracrawl-mono | paracrawl8 | mono | Datasets that are crawled from the web. Only [mono datasets](https://paracrawl.eu/index.php/moredata) are used in this importer. Parallel corpus is available using opus importer.
[News crawl](http://data.statmt.org/news-crawl) | news-crawl | news.2019 | mono | Some news monolingual datasets from [WMT21](https://www.statmt.org/wmt21/translation-task.html)
[Common crawl](https://commoncrawl.org/) | commoncrawl | wmt16 | mono | Huge web crawl datasets. The links are posted on [WMT21](https://www.statmt.org/wmt21/translation-task.html)
GCS bucket | bucket | bucket/path/en-ca/dataset | mono/corpus | Local monolingual dataset that is already downloaded. The dataset name is an absolute path prefix without ".lang.gz"
Local mono | local-mono | /tmp/test-mono | mono | Local monolingual dataset that is already downloaded. The dataset name is an absolute path prefix without ".lang.gz"


## Adding a new importer

Just add a shell script to [corpus](https://github.com/mozilla/firefox-translations-training/tree/main/pipeline/data/importers/corpus) or [mono](https://github.com/mozilla/firefox-translations-training/tree/main/pipeline/data/importers/mono) which is named as `<prefix>.sh`
and accepts the same parameters as the other scripts from the same folder.
