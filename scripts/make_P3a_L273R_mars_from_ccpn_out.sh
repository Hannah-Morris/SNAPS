SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

TEST_DATA=${SCRIPT_DIR=}/../data/P3a_L273R

  python3 ${SCRIPT_DIR}/ccpn_to_mars.py ${TEST_DATA}/test_P3a_L273_ccpn.out \
| sed -e 's/nan/-  /g' \
> ${TEST_DATA}/mars_shifts.txt

