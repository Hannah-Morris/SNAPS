import string
import sys
from math import isnan
from numpy import array
from tabulate import tabulate
from string import digits
from string import ascii_letters


new_headings = '''\
    Res_N Res_type SS_resid SS_resname CA CA_pred HA HA_pred H H_pred CB CB_pred
    C C_pred N N_pred Log_prob  
'''.split()

def read_file(file_name):
    headings_to_column_number = {}
    new_table = [new_headings]
    with open(file_name) as fh:
        for index, line in enumerate(fh):
            line = line.strip()
            if index == 0:
                headings = line.split()
                for column_number, heading in enumerate(headings):
                    headings_to_column_number[heading] = column_number
            elif index > 0 and len(line) > 0:
                new_row = []
                new_table.append(new_row)
                for heading_name in new_headings:
                    if heading_name == 'SS_resid':
                        SS_name_column = headings_to_column_number['SS_name']
                        field = fields[SS_name_column]
                        new_field = field.rstrip(ascii_letters)
                        new_row.append(new_field)
                        pass
                    elif heading_name == 'SS_resname':
                        SS_name_column = headings_to_column_number['SS_name']
                        field = fields[SS_name_column]
                        new_field = field.lstrip(digits)
                        new_row.append(new_field)
                    else:
                        column_number = headings_to_column_number[heading_name]
                        field = fields[column_number]
                        new_row.append(field)

        return new_table


if __name__ == '__main__':
    table = read_file(sys.argv[1])
    table_data_1 = read_file(sys.argv[1])


    print(tabulate(table_data_1, tablefmt='plain', headers= new_headings))

