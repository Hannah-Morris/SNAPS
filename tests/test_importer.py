from pathlib import Path
import pandas as pd
import pytest
from numpy import NaN
from pandas.testing import assert_frame_equal
from pynmrstar import Entry

from SNAPS_assigner import SNAPS_assigner
from SNAPS_importer import SNAPS_importer, SnapsImportException


TEST_DATA = Path(__file__).parent.parent / 'test_data'

def test_import_aa_type_info_file():
    importer = SNAPS_importer()
    importer.import_obs_shifts(TEST_DATA / 'nef_resonances_4_test.nef', 'nef')
    result = importer.import_aa_type_info_file(TEST_DATA / 'test_aa_info.txt')
    data = {
        'Atom_type': ['236Pro', '237Ala', '238Met', '239Thr'],
        'SS_name': ['236Pro', '237Ala', '238Met', '239Thr'],
        'C': [176.44282, 177.76473, 175.89544, 173.0364],
        'CA': [62.82192, 52.38738, 55.48441, 62.02754],
        'CB': [32.26901, 19.14546, 32.90399, 69.45566],
        'H': [NaN, 8.54894, 8.37917, 7.88704],
        'HA': [4.43775, 4.27742, 4.41167, 3.88160],
        'N': [NaN, 124.63832, 119.82062, 115.46265],
        'SS_class': ['ACDEFGHIKLMNPQRSTVWY', 'AVI', 'AVI', 'ACDEFGHIKLMNPQRSTVWY'],
        'SS_class_m1': ['ACDEFGHIKLMNPQRSTVWY', 'ACDEFGHIKLMNPQRSTVWY', 'ACDEFGHIKLMNPQRSTVWY', 'ACDEFGHIKLMNPQRSTVWY']
    }
    expected = pd.DataFrame(data, index=data['SS_name'], columns=['SS_name', 'C', 'CA', 'CB', 'H', 'HA', 'N',
                                                                  'SS_class', 'SS_class_m1'])
    expected.columns.names = ['Atom_type']
    assert_frame_equal(expected, result, check_exact=False, rtol=0.01)


def test_import_aa_type_info_out():
    importer = SNAPS_importer()
    importer.import_obs_shifts(TEST_DATA / 'nef_resonances_4_test.nef', 'nef')
    result_out = importer.import_aa_type_info_file(TEST_DATA / 'test_aa_info_out.txt')
    data_out = {
        'Atom_type': ['236Pro', '237Ala', '238Met', '239Thr'],
        'SS_name': ['236Pro', '237Ala', '238Met', '239Thr'],
        'C': [176.44282, 177.76473, 175.89544, 173.0364],
        'CA': [62.82192, 52.38738, 55.48441, 62.02754],
        'CB': [32.26901, 19.14546, 32.90399, 69.45566],
        'H': [NaN, 8.54894, 8.37917, 7.88704],
        'HA': [4.43775, 4.27742, 4.41167, 3.88160],
        'N': [NaN, 124.63832, 119.82062, 115.46265],
        'SS_class': ['ACDEFGHIKLMNPQRSTVWY', 'CDEFGHKLMNPQRSTWY', 'CDEFGHKLMNPQRSTWY', 'ACDEFGHIKLMNPQRSTVWY'],
        'SS_class_m1': ['ACDEFGHIKLMNPQRSTVWY', 'ACDEFGHIKLMNPQRSTVWY', 'ACDEFGHIKLMNPQRSTVWY', 'ACDEFGHIKLMNPQRSTVWY']
    }

    expected_out = pd.DataFrame(data_out, index=data_out['SS_name'], columns=['SS_name', 'C', 'CA', 'CB', 'H', 'HA',
                                                                              'N', 'SS_class', 'SS_class_m1'])
    expected_out.columns.names = ['Atom_type']
    assert_frame_equal(expected_out, result_out, check_exact=False, rtol=0.01)


def test_headings_aa_info():
    importer = SNAPS_importer()
    importer.import_obs_shifts(TEST_DATA / 'nef_resonances_4_test.nef', 'nef')

    with pytest.raises(SnapsImportException) as e:
        importer.import_aa_type_info_file(TEST_DATA / 'test_aa_info_bad_header.txt')

    assert 'Unexpected column name(s) [AAA]' in str(e.value)
    assert 'expected column names are:' in str(e.value)
    for name in 'Type SS_name AA'.split():
        assert name in str(e.value)


def test_headings_aa_info_type_column_bad():
    importer = SNAPS_importer()
    importer.import_obs_shifts(TEST_DATA / 'nef_resonances_4_test.nef', 'nef')

    with pytest.raises(SnapsImportException) as e:
        importer.import_aa_type_info_file(TEST_DATA / 'test_aa_info_bad_type_column.txt')
    assert "Type column row error: 'Type' column rows can only contain 'in' or 'ex'" in str(e.value)


def test_lowercase_headings_bad():
    importer = SNAPS_importer()
    importer.import_obs_shifts(TEST_DATA / 'nef_resonances_4_test.nef', 'nef')

    with pytest.raises(SnapsImportException) as e:
        importer.import_aa_type_info_file(TEST_DATA / 'test_aa_info_lowercase_column_names.txt')


def test_aa_letters_bad():
    importer = SNAPS_importer()
    importer.import_obs_shifts(TEST_DATA / 'nef_resonances_4_test.nef', 'nef')

    with pytest.raises(SnapsImportException) as e:
        importer.import_aa_type_info_file(TEST_DATA / 'test_aa_info_letters_bad.txt')
    assert "AA type letters incorrect, Amino Acid letters can only be: 'ACDEFGHIKLMNPQRSTVWY'" in str(e.value)


def test_obs_data_bad():
    importer = SNAPS_importer()
    importer.import_obs_shifts(TEST_DATA / 'nef_resonances_4_test.nef', 'nef')

    with pytest.raises(SnapsImportException) as e:
        importer.import_aa_type_info_file(TEST_DATA / 'test_aa_info_bad_obs_data.txt')
    assert "Incorrect data given, unexpected spin system in aa types. the input spin systems should be in the chemical"\
           " shift list" in str(e.value)


def test_offset_data():
    importer = SNAPS_importer()
    importer.import_obs_shifts(TEST_DATA / 'nef_resonances_4_test.nef', 'nef')

    with pytest.raises(SnapsImportException) as e:
        importer.import_aa_type_info_file(TEST_DATA / 'test_aa_info_m1.txt')
    assert "offset can only be 0 or -1" in str(e.value)

@pytest.mark.skip('unknown errors')
def test_matrix_data():
    obs = pd.read_csv(TEST_DATA / 'test_obs.txt', sep='\s+')
    log = pd.read_csv(TEST_DATA / 'test_log.txt', sep='\s+')
    preds = pd.read_csv(TEST_DATA / 'test_preds.txt', sep='\s+')

    print('ibs', obs)
    print('log',log)
    print('preds', preds)

    original_log_prob_matrix = log.copy(deep=True)
    updated_log_prob_matrix = SNAPS_assigner._apply_ss_class_penalties(log, obs, preds)
    diffs = updated_log_prob_matrix - original_log_prob_matrix
    print("Differences in matrix", diffs)

# if diffs != 0:
    #with pytest.raises(SnapsImportException) as e:

    #assert "differences in the matrix should be 0"

EXPECTED_IN_DATA = {
        'Atom_type': ['236Pro', '237Ala', '238Met', '239Thr'],
        'SS_name': ['236Pro', '237Ala', '238Met', '239Thr'],
        'C': [176.44282, 177.76473, 175.89544, 173.0364],
        'CA': [62.82192, 52.38738, 55.48441, 62.02754],
        'CB': [32.26901, 19.14546, 32.90399, 69.45566],
        'H': [NaN, 8.54894, 8.37917, 7.88704],
        'HA': [4.43775, 4.27742, 4.41167, 3.88160],
        'N': [NaN, 124.63832, 119.82062, 115.46265],
        'SS_class': ['ACDEFGHIKLMNPQRSTVWY', 'AVI', 'AVI', 'ACDEFGHIKLMNPQRSTVWY'],
        'SS_class_m1': ['ACDEFGHIKLMNPQRSTVWY', 'ACDEFGHIKLMNPQRSTVWY', 'ACDEFGHIKLMNPQRSTVWY', 'ACDEFGHIKLMNPQRSTVWY']
    }
EXPECTED_COLUMNS = ['SS_name', 'C', 'CA', 'CB', 'H', 'HA', 'N', 'SS_class', 'SS_class_m1']
EXPECTED_IN = pd.DataFrame(EXPECTED_IN_DATA, index=EXPECTED_IN_DATA['SS_name'], columns=EXPECTED_COLUMNS)
EXPECTED_IN.columns.names = ['Atom_type']
def test_import_aa_type_info_nef():
    importer = SNAPS_importer()
    importer.import_obs_shifts(TEST_DATA / 'nef_resonances_4_test.nef', 'nef')

    with open(TEST_DATA / 'nef_aa_types.nef', 'r') as file_handle:
        entry = Entry.from_file(file_handle)
    result = importer.import_aa_type_info_nef(entry, frame_name='default')

    assert_frame_equal(EXPECTED_IN, result, check_exact=False, rtol=0.01)