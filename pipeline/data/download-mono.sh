#!/bin/bash
##
# Downloads monolingual datasets
#

set -x
set -euo pipefail

dataset=$1
dataset_sanitized=$2
lang=$3
max_sent=$4
output_path=$5
coef=0.1

COMPRESSION_CMD="${COMPRESSION_CMD:-pigz}"
ARTIFACT_EXT="${ARTIFACT_EXT:-gz}"

echo "###### Downloading monolingual data for language ${lang} dataset ${dataset}"

cd "$(dirname "${0}")"

tmp=$(dirname "${output_path}")/original
mkdir -p "${tmp}"

echo "### Downloading dataset"
original_prefix="${tmp}/${dataset_sanitized}.original.${lang}"
name=${dataset#*_}
type=${dataset%%_*}

echo "File size before downloading: `stat -f "%z" "${original_prefix}.${ARTIFACT_EXT}"`"

# Choose either the .sh or .py script.
if [[ -f "importers/mono/${type}.py" ]]; then
  script="python importers/mono/${type}.py"
else
  script="bash importers/mono/${type}.sh"
fi

test -s "${original_prefix}.${ARTIFACT_EXT}" ||
  ${script} "${lang}" "${original_prefix}" "${name}"

echo "File size before sampling: `stat -f "%z" "${original_prefix}.${ARTIFACT_EXT}"`"

echo "### Sampling dataset"
# temporary disable pipefail because perl operation causes SIGPIPE (141)
set +o pipefail
${COMPRESSION_CMD} -dc "${original_prefix}.${ARTIFACT_EXT}" |
shuf -n "$(bc -l <<<"scale=0; (${max_sent}+${max_sent}*${coef}) / 1")" |
perl -ne 'print if(split(/\s/, $_) < 100)' |
head -n "${max_sent}" |
${COMPRESSION_CMD} >"${output_path}"
set -o pipefail

echo "File size after sampling: `stat -f "%z" "${original_prefix}.${ARTIFACT_EXT}"`"

rm -rf "${original_prefix}.${ARTIFACT_EXT}"

echo "###### Done: Downloading monolingual data"
