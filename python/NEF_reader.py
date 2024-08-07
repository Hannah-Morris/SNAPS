import logging
from pathlib import Path
from typing import List
import sys
import pynmrstar
import pandas as pd

_CHEMICAL_SHIFT_LIST_FRAME = 'nef_chemical_shift_list'  # tag for the NEF chemical shift list frame
_CHEMICAL_SHIFT_LOOP = 'nef_chemical_shift'  # tag for the NEF chemical shift list loop
_CHAIN_CODE = 'chain_code'  # NEF loop heading for chain code
_SEQUENCE_CODE = 'sequence_code'  # NEF loop heading for sequence
_ATOM_NAME = 'atom_name'  # NEF loop heading for atom name
_RESIDUE_NAME = 'residue_name'  # NEF loop heading for residue name
_SHIFT_VALUE = 'value'  # NEF loop heading for  the chemical shift list value

_OFFSET_SLICE = slice(-2, None)  # equivalent to value[-2:]
_RESIDUE_CODE_SLICE = slice(0, -2)  # equivalent to value[:-2]
_PREVIOUS_RESIDUE_FLAG = '_m'  # the flag SNAPS uses to indicate the correlated shift from the previous residue
_PREVIOUS_RESIDUE_OFFSET = '-1'  # the offset to the previous residue as used by CCPN and NEF

_DEFAULT_SHIFT_LIST = 'default'

TRANSLATIONS_3_1_PROTEIN = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
TRANSLATIONS_1_3_PROTEIN = {
    value: key for (key, value) in TRANSLATIONS_3_1_PROTEIN.items()
}

# ignore pynmrstar warning about loops with no data
logging.getLogger('pynmrstar').addFilter(lambda record: "Loop with no data" not in record.msg)

def read_nef_shifts_from_file(file_name: Path, shift_list_name: str = _DEFAULT_SHIFT_LIST) -> List[List[str]]:
    """
    read nef shifts from a file given by filename and return a table of shifts with headings sequence_code, atom_name
    and shifts as a list of lists. note shifts from the previous residue are indicated by appending '_m' to the atom
    name as used by SNAPS

    :param file_name: file name
    :param shift_list_name: name of the shift list frame defaults to "default"
    :return: list of lists with each sub list being a row with headings sequence code, atom name and shift
    """
    with open(file_name, 'r') as file_handle:
        result_shifts = read_nef_shifts(file_handle, shift_list_name)

    return result_shifts


def read_nef_shifts(file_handle, shift_list_name=_DEFAULT_SHIFT_LIST, chain='A'):
    """

    :param file_handle:
    :param shift_list_name:
    :return:
    """
    file_name = file_handle.name


    entry = pynmrstar.Entry.from_string(file_handle.read())

    first_shift_frame = _read_named_shift_frame_or_error(entry, shift_list_name, file_name)

    chem_shift_loop = _frame_to_shift_loop_or_error(file_handle.name, first_shift_frame)

    #TODO should we be dealing with chains here
    chain_index = _read_heading_index_or_error(first_shift_frame, _CHAIN_CODE, chem_shift_loop, file_name)
    sequence_index = _read_heading_index_or_error(first_shift_frame, _SEQUENCE_CODE, chem_shift_loop, file_name)
    atom_index = _read_heading_index_or_error(first_shift_frame, _ATOM_NAME, chem_shift_loop, file_name)
    chem_shift_index = _read_heading_index_or_error(first_shift_frame, _SHIFT_VALUE, chem_shift_loop, file_name)
    residue_name_index = _read_heading_index_or_error(first_shift_frame, _RESIDUE_NAME, chem_shift_loop, file_name)

    output = []
    for row_index, row in enumerate(chem_shift_loop.data):

        chemical_shift = _read_shift_from_row_or_error(row[chem_shift_index], row_index, shift_list_name, row,
                                                       file_name)
        sequence_code = row[sequence_index]
        atom_name = row[atom_index]
        residue_name = row[residue_name_index]

        if sequence_code[_OFFSET_SLICE] == _PREVIOUS_RESIDUE_OFFSET:
            sequence_code = sequence_code[:-2]
            atom_name = atom_name + _PREVIOUS_RESIDUE_FLAG

        output_row = [sequence_code, atom_name, chemical_shift, residue_name]

        output.append(output_row)

    return output


def read_nef_obs_shifts_from_file_to_pandas(raw_file_name, chain):

    output = _raw_read_shifts_to_pandas(chain, raw_file_name)

    output = output.rename(columns={'sequence_code': 'SS_name', 'atom_name': 'Atom_type', 'value': 'Shift'})

    output['SS_name'] = output['SS_name'].astype(str) + output['residue_name']
    output['SS_name'] = output['SS_name'].str.title()
    del output['residue_name']

    return output

def read_nef_pred_shifts_from_file_to_pandas(raw_file_name, chain):

    output = _raw_read_shifts_to_pandas(chain, raw_file_name)

    output = output.rename(columns={'sequence_code': 'Res_N', 'atom_name': 'Atom_type', 'value': 'Shift', 'residue_name': 'Res_type'})

    output = output.replace({'Res_type':TRANSLATIONS_3_1_PROTEIN})

    return output



def _raw_read_shifts_to_pandas(chain, raw_file_name):
    if not Path(raw_file_name).exists():
        file_name, shift_list_name = _split_path_and_frame(raw_file_name)
    else:
        file_name = raw_file_name
        shift_list_name = _DEFAULT_SHIFT_LIST
    with open(file_name, 'r') as file_handle:
        output = read_nef_shifts_to_pandas(file_handle, shift_list_name, chain)
    return output


def _split_path_and_frame(file_name):
    if ':' in file_name:
        path_fields = file_name.split(':')
        shift_list_name = path_fields[-1].strip()
        if not shift_list_name:
            shift_list_name = _DEFAULT_SHIFT_LIST

        file_name = ':'.join(path_fields[:-1]).strip()
    else:
        shift_list_name = _DEFAULT_SHIFT_LIST
    return file_name, shift_list_name


def read_nef_shifts_to_pandas(file_handle, shift_list_name=_DEFAULT_SHIFT_LIST, chain="A"):
    snaps_shifts = read_nef_shifts(file_handle, shift_list_name, chain)

    pandas_columns = [_SEQUENCE_CODE, _ATOM_NAME, _SHIFT_VALUE, _RESIDUE_NAME]
    return pd.DataFrame(snaps_shifts, columns=pandas_columns)


def _read_shift_from_row_or_error(shift_string, row_index, row, loop_name, file_name):
    try:
        chemical_shift = float(shift_string)
    except ValueError:
        row = ' '.join(row)
        msg = f"Error: can't convert '{shift_string}' to float in row number {row_index} [{row}] " + \
              f"of loop {loop_name} in {file_name}"
        raise Exception(msg)

    return chemical_shift


def _frame_to_shift_loop_or_error(file_name, first_shift_frame):
    try:
        chem_shift_loop = first_shift_frame.get_loop(_CHEMICAL_SHIFT_LOOP)
    except KeyError:
        raise Exception(f'ERROR: there are no chemical shift list loops in {file_name}')
    return chem_shift_loop


def _read_named_shift_frame_or_error(entry, frame_name_selector, file_name):
    """

    :param entry: pynmrstar.Entry.from_file(file_handle)
    :param frame_name_selector:first_shift_frame
    :param file_name:file name
    :return error if there is no saveframe that reads 'chemical shift list' in the file. Otherwise, return a list of the
    chemical shifts from the file
    """
    chem_shift_saveframes = entry.get_saveframes_by_category(_CHEMICAL_SHIFT_LIST_FRAME)
    if len(chem_shift_saveframes) < 0:
        raise Exception(f'ERROR: there are no chemical shift list frames in {file_name}')

    result_shifts = None
    for frame in chem_shift_saveframes:
        frame_name = frame.name[len(frame.category):].lstrip('_')
        if frame_name == frame_name_selector:
            result_shifts = frame
            break

    if result_shifts is None:
        raise Exception(f'ERROR: there are no chemical shift list frame called {frame_name_selector} in {file_name}')

    return result_shifts


def _read_heading_index_or_error(save_frame, heading, loop, file_name):
    """

    :param save_frame: _CHEMICAL_SHIFT_LIST_FRAME
    :param heading: _SEQUENCE CODE
    :param loop: _CHEMICAL_SHIFT_LOOP
    :param file_name: file name
    :return: error if there is no sequence index/ heading  for {sequence code} in file name. Otherwise, return sequence
    index for headings in the saveframe
    """
    sequence_index = loop.tag_index(heading)
    if sequence_index is None:
        msg = f"ERROR: couldn't find a sequence heading {_SEQUENCE_CODE} in save frame {save_frame.name} in {file_name}"
        raise Exception(msg)
    return sequence_index


ERROR = 1

if __name__ == '__main__':
    num_args = len(sys.argv)
    required_args = 'a nef file name and an optional chain_code [default=A]'
    if num_args > 3:
        print(f'Error: arguments are {required_args}', file=sys.stderr)
        print('exiting...', file=sys.stderr)

        sys.exit(ERROR)
    elif num_args < 2:
        print(f'Error: I need {required_args}', file=sys.stderr)
        print('exiting...', file=sys.stderr)
        sys.exit(ERROR)

    file_path = Path(sys.argv[1])
    if num_args == 3:
        chain_code = sys.argv[2]
    else:
        chain_code = 'A'
    result = read_nef_obs_shifts_from_file_to_pandas(file_path, chain_code)
    print(result)
