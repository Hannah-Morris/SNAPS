import sys
from copy import deepcopy
from tabulate import tabulate
from string import ascii_letters

C_CONNECTED_ATOMS = 'CA', 'CB', 'C'  # atom through which spin systems are correlated
ATOM_TYPE = 'Atom_type'
SS_NAME = 'SS_name'
new_headings = '''\
    H N CA CA-1 CB CB-1 C C-1 HA
'''.split()
new_headings.insert(0, '')


def read_file(file_name):

    # read the table
    residue_shifts = read_table(file_name)

    # add residue - 1  chemical shifts
    residue_shifts_m1 = add_m1_shifts(residue_shifts)

    # format into tabulate table
    table = make_table(residue_shifts_m1)

    return table


def make_table(residue_shifts):
    new_table = [new_headings, ]
    for residue_number in sorted(residue_shifts):

        pseudo_residue_name = 'PR_' + str(residue_number)
        row = [pseudo_residue_name, ]
        new_table.append(row)

        current_shifts = residue_shifts[residue_number]
        for atom_name in new_headings:
            if atom_name != '':
                if atom_name in current_shifts:
                    current_shift = current_shifts[atom_name]
                else:
                    current_shift = '-'
                row.append(current_shift)
    return new_table


def add_m1_shifts(residue_shifts):

    residue_shifts = deepcopy(residue_shifts)

    for residue_number in residue_shifts:
        if residue_number - 1 in residue_shifts:
            residue_m1_shifts = residue_shifts[residue_number - 1]
            current_residue_shifts = residue_shifts[residue_number]
            for atom_name in C_CONNECTED_ATOMS:
                if atom_name in residue_m1_shifts:
                    shift = residue_m1_shifts[atom_name]
                    atom_name_m1 = atom_name + '-1'
                    current_residue_shifts[atom_name_m1] = shift

    return residue_shifts


def read_table(file_name):
    residue_shifts = {}
    headings_to_column_index = {}
    with open(file_name) as fh:
        for index, line in enumerate(fh):
            line = line.strip()
            if index == 0:
                headings = line.split()
                for column_number, heading in enumerate(headings):
                    headings_to_column_index[heading] = column_number

            elif index > 0 and len(line) > 0:

                fields = line.split()

                current_shifts = {}
                current_residue = None

                for atom_name in headings_to_column_index:

                    if atom_name == SS_NAME:
                        SS_name_column = headings_to_column_index[SS_NAME]
                        field = fields[SS_name_column]
                        current_residue = int(field.rstrip(ascii_letters))

                    elif atom_name == ATOM_TYPE:
                        pass

                    elif atom_name in new_headings:
                        column_index = headings_to_column_index[atom_name]
                        new_shift = fields[column_index]
                        if new_shift == 'NaN':
                            new_shift = '-'
                        current_shifts[atom_name] = new_shift

                    else:
                        pass

                residue_shifts[current_residue] = current_shifts

    return residue_shifts


if __name__ == '__main__':

    table = read_file(sys.argv[1])

    print(tabulate(table, tablefmt='plain'))
