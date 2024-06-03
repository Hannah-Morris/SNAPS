SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

TEST_DATA=${SCRIPT_DIR=}/../data/P3a_L273R

#TODO:  sed command can be removed once nef-pipelines is more sensible
  nef fasta import sequence --starts 235 ${TEST_DATA}/P3a_L273R.fasta           \
| nef sparky import shifts --frame-name  default ${TEST_DATA}/sparky_shifts.txt \
| sed -e 's/ HN / H  /g'                                                        \
> ${TEST_DATA}/P3a_L273R.nef
