#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main SNAPS script for assigning an observed shift list based on predicted shifts

@author: aph516
"""

import string
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from numpy import arange
from pynmrstar import Saveframe, Loop
from tabulate import tabulate

from SNAPS_importer import SNAPS_importer
from SNAPS_assigner import SNAPS_assigner

# TODO remove these from NEF importer
from NEF_reader import _split_path_and_frame, TRANSLATIONS_1_3_PROTEIN
import logging


def _get_arguments(system_args):
    import argparse

    parser = argparse.ArgumentParser(
            description="SNAPS (Simple NMR Assignments from Predicted Shifts)")

    # Mandatory arguments
    parser.add_argument("shift_file",
                        help="A table of observed chemical shifts, for nef you can append a data frame name after a colon [e.g. my_shifts.nef:observed].")
    parser.add_argument("pred_file",
                        default=None,
                        help="A table of predicted chemical shifts for nef can just be a dataframe name [default predicted].")
    parser.add_argument("output_file",
                        help="The file results will be written to.")

    # Information on input files and configuration options
    parser.add_argument("--shift_type",
                        choices=["snaps", "ccpn", "sparky", "mars",
                                 "xeasy", "nmrpipe", "nef", "test"],
                        default=None,
                        help="The format of the observed shift file.")
    parser.add_argument("--pred_type",
                        choices=["shiftx2", "sparta+", "nef"],
                        default=None,
                        help="The format of the predicted shifts")
    parser.add_argument("--out_type",
                        choices=["snaps", "nef"],
                        default=None,
                        help="The format of the predicted shifts")
    parser.add_argument("--aa_type",
                        choices=["snaps", "nef"],
                        default=None,
                        help="The format of the per residue amino acid type restraints")
    parser.add_argument("--pred_seq_offset", type=int, default=0,
                        help="""An offset to apply to the residue numbering in
                        the predicted shifts.""")
    parser.add_argument("-c", "--config_file",
                        default="../config/config.txt",
                        help="A file containing parameters for the analysis.")
    parser.add_argument("--test_aa_classes", default=None,
                        help="""For test data only.
                        A string containing a comma-separated list of the amino acid
                        classes for the i residue, a semicolon, then a list of AA
                        classes for the i-1 residue. No spaces.
                        eg. "ACDEFGHIKLMNPQRSTVWY;G,S,T,AVI,DN,FHYWC,REKPQML" for
                        a sequential HADAMAC """)
    parser.add_argument("--aa_restraints", default=None, nargs=1,
                        help="""A file containing restraints on the amino acid types of residues.""")
    parser.add_argument("--obs_chain", default='A',
                        help="""The chain to use for the observed shifts.""")
    parser.add_argument("--pred_chain", default='A',
                        help="""The chain to use for the predicted shifts.""")

    # Options controlling output files
    parser.add_argument("-l", "--log_file", default=None,
                        help="A file logging information will be written to.")

    parser.add_argument("--shift_output_file", default=None,
                        help="""The file the assigned shiftlist will be written to.""")
    parser.add_argument("--shift_output_type", default=None,
                        choices=["sparky", "xeasy", "nmrpipe", "nef"],
                        help="One or more output formats for chemical shift export")
    parser.add_argument("--shift_output_confidence", nargs="*",
                        choices=["High", "Medium", "Low", "Unreliable", "Undefined"],
                        default=["High", "Medium", "Low", "Unreliable", "Undefined"],
                        help="""Limits the shiftlist output to assignments with
                        particular confidence levels. More than one level is allowed""")

    parser.add_argument("--strip_plot_file",
                        default=None,
                        help="A filename for an output strip plot.")
    parser.add_argument("--hsqc_plot_file",
                        default=None,
                        help="A filename for an output HSQC plot.")

    args = parser.parse_args(system_args)

    # if input shifts are nef all file types are nef unless
    # otherwise specified
    if args.shift_type == "nef":
        if args.pred_type is None:
            args.pred_type = "nef"
        if args.aa_type is None:
            args.aa_type = "nef"
        if args.out_type is None:
            args.out_type = "nef"
        if args.shift_output_type is None:
            args.shift_output_type = "nef"

    if args.shift_type is None:
        args.shift_type = "snaps"
    if args.pred_type is None:
        args.pred_type = "shiftx2"
    if args.aa_type is None:
        args.aa_type = "snaps"
    if args.out_type is None:
        args.out_type ='snaps'
    if args.shift_output_type is None:
        args.shift_output_type = "sparky"

    # if False:   # For convenience when testing
    #     args = parser.parse_args(("data/P3a_L273R/naps_shifts.txt",
    #                               "data/P3a_L273R/shiftx2.cs",
    #                               "output/test.txt",
    #                               "--shift_type","snaps",
    #                               "--pred_type","shiftx2",
    #                               "-c","config/config_yaml_2.txt",
    #                               "-l","output/test.log"))
    return args


def run_snaps(system_args):

    #### Command line arguments
    args = _get_arguments(system_args)

    #### Set up logging
    logger = _setup_logger(args)


    #### Set up the SNAPS_assigner object
    assigner = SNAPS_assigner()

    # Import config file
    assigner.read_config_file(args.config_file)


    # Importer for observed and predicted shifts
    importer = SNAPS_importer()

    #### import observed shifts
    if args.shift_type=="test":
        _import_test_shifts(args, importer)
    else:
        _import_shifts(args, importer)
    assigner.obs = importer.obs
    logger.info("Finished reading in %d spin systems from %s",
                len(assigner.obs["SS_name"]), args.shift_file)


    # import predicted shifts
    #TODO move this to importer
    if args.pred_type == "nef":
        if not Path(args.pred_file).exists():
            file_name, shift_list_name = _split_path_and_frame(args.shift_file)
            pred_file = f"{file_name}:{args.pred_file}"

            assigner.import_pred_shifts(pred_file, args.pred_type, args.pred_chain, args.pred_seq_offset)


    else:
        assigner.import_pred_shifts(args.pred_file, args.pred_type, args.pred_seq_offset)

    #### import aa type restraints
    _import_aa_type_info(args, assigner, importer)

    #### Do the analysis
    assigner.prepare_obs_preds()
    assigner.calc_log_prob_matrix()
    assigner.calc_mismatch_matrix()

    if assigner.pars["iterate_until_consistent"]:
        assigner.assign_df = assigner.find_consistent_assignments(set_assign_df=True)
    else:
        assigner.assign_from_preds(set_assign_df=True)
        # breakpoint()
        assigner.add_consistency_info(threshold=assigner.pars["seq_link_threshold"])
        # breakpoint()

    #### Output the results
    _output_results(args, assigner, logger)

    #### Make some plots
    plots = _output_plots(args, assigner, logger)

    #### Close the log file
    if logger.handlers:
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)

    return(plots)


def _import_shifts(args, importer):
    importer.import_obs_shifts(args.shift_file, args.shift_type, SS_num=False, chain=args.obs_chain)


def _import_test_shifts(args, importer):
    if args.test_aa_classes is None:
        importer.import_testset_shifts(args.shift_file)
    else:
        aa_class, aa_class_m1 = args.test_aa_classes.split(";")
        importer.import_testset_shifts(args.shift_file,
                                       SS_class=aa_class.split(","),
                                       SS_class_m1=aa_class_m1.split(","))


def _output_plots(args, assigner, logger):
    plots = []
    if args.hsqc_plot_file is not None:
        hsqc_plot = assigner.plot_hsqc(args.hsqc_plot_file, "html")
        logger.info("Finished writing HSQC plot to %s", args.hsqc_plot_file)
        plots += [hsqc_plot]
    if args.strip_plot_file is not None:
        strip_plot = assigner.plot_strips(args.strip_plot_file, "html")
        logger.info("Finished writing strip plot to %s", args.strip_plot_file)
        plots += [strip_plot]

    return plots

@contextmanager
def _open_or_stdout(filename, *args, **kwargs):
    if filename == '-':
        yield sys.stdout
    else:
        with open(filename, *args, **kwargs) as f:
            yield f


def _output_results(args, assigner, logger,):
    headings = '''
        Res_name Res_N Res_type SS_name Dummy_res Dummy_SS CA CA_pred HA HA_pred H H_pred CB CB_pred
         C C_pred N N_pred Log_prob Max_mismatch_m1 Max_mismatch_p1 Num_good_links_m1 
    '''.split()

    if args.out_type == "nef":

        save_frame = Saveframe.from_scratch('nefpls_assignments_snaps', 'nefpls_assignments')

        save_frame.add_tag('category', f'nefpls_assignments_snaps')

        loop = Loop.from_scratch('nefpls_assignments')
        save_frame.add_loop(loop)

        # fragment_id
        tags = 'index chain_code sequence_code residue_name unassigned_sequence_code assigned merit'.split()
        loop.add_tag(tags) #snaps_log_prob snaps_max_mismatch_m1 snaps_max_mismatch_p1 snaps_num_good_links_m1

        assign_df = assigner.assign_df.sort_index()


        min_log_prob = -min(assign_df['Log_prob'])

        nef_out_frame = pd.DataFrame(assign_df[['SS_name', 'Res_N', 'Res_name',  'Res_type','Log_prob']])


        nef_out_frame = nef_out_frame.rename(columns={
            'SS_name': 'unassigned_sequence_code',
            'Res_N': 'sequence_code',
            'Res_type': 'residue_name',
            'Log_prob': 'merit'
        })



        nef_out_frame['assigned'] = ~nef_out_frame['Res_name'].str.startswith('DR')

        nef_out_frame = nef_out_frame.sort_values('sequence_code')

        # min_residue_sequence = sequence['Res_N'].min()
        # max_residue_sequence = sequence['Res_N'].max()

        min_residue = nef_out_frame['sequence_code'].min()
        max_residue = nef_out_frame['sequence_code'].max()

        # min_residue = min_residue_sequence if min_residue_sequence < min_residue else min_residue
        # max_residue = max_residue_sequence if max_residue_sequence > max_residue else max_residue

        new_index = pd.Index(arange(min_residue, max_residue + 1))
        nef_out_frame_assigned = nef_out_frame[nef_out_frame['assigned']]
        nef_out_frame_unassigned = nef_out_frame[~nef_out_frame['assigned']]

        nef_out_frame_assigned = nef_out_frame_assigned.set_index('sequence_code').reindex(new_index)

        nef_out_frame_assigned['sequence_code'] = nef_out_frame_assigned.index

        nef_out_frame_assigned = nef_out_frame_assigned.reset_index()
        # nef_out_frame_assigned.index += 1
        #
        # nef_out_frame_assigned['sequence_code'] = nef_out_frame_assigned['sequence_code'].astype(int)

        nef_out_frame_assigned['assigned'] = nef_out_frame['assigned'].replace({'NaN': False})

        nef_out_frame_assigned['merit'] = ((min_log_prob - nef_out_frame_assigned['merit']) / min_log_prob) - 1

        nef_out_frame_assigned['residue_name'] = nef_out_frame_assigned['residue_name'].replace(TRANSLATIONS_1_3_PROTEIN)

        nef_out_frame_assigned['unassigned_sequence_code'] = nef_out_frame_assigned['unassigned_sequence_code'].str.rstrip(string.ascii_letters)
        # nef_out_frame_assigned = nef_out_frame_assigned.replace({float.NaN: '.'})

        nef_out_frame_assigned['chain_code'] = args.obs_chain

        # nef_out_frame_assigned['fragment_id'] = (nef_out_frame_assigned['sequence_code'].diff() > 1).cumsum() + 1


        nef_out_frame_assigned['sequence_code'] = nef_out_frame_assigned['sequence_code'].astype(int).astype('str')

        nef_out_frame_assigned['index'] = nef_out_frame_assigned['index'] - nef_out_frame_assigned['index'][0] +1
        nef_out_frame_assigned['index'] = nef_out_frame_assigned['index'].astype(int).astype(str)

        nef_out_frame_assigned['unassigned_sequence_code'] = nef_out_frame_assigned['unassigned_sequence_code'].astype(str).replace({'nan': '.'})
        nef_out_frame_assigned.drop(columns=['Res_name'], inplace=True)

        nef_out_frame_assigned['residue_name'] = nef_out_frame_assigned['residue_name'].astype(str).replace({'nan': '.'})

        nef_out_frame_assigned_bad_residues = nef_out_frame_assigned[nef_out_frame_assigned['residue_name'] == '.']['sequence_code'].astype(int)

        nef_out_frame_unassigned = nef_out_frame_unassigned.drop(columns=['Res_name'])
        nef_out_frame_unassigned = nef_out_frame_unassigned.reset_index()
        nef_out_frame_unassigned['unassigned_sequence_code'] = nef_out_frame_unassigned['unassigned_sequence_code'].str.rstrip(string.ascii_letters)
        nef_out_frame_unassigned['sequence_code'] = nef_out_frame_unassigned['sequence_code'].astype('str').replace({'nan': '.'})
        nef_out_frame_unassigned['residue_name'] = nef_out_frame_unassigned['residue_name'].astype('str').replace({'nan': '.'})
        nef_out_frame_unassigned.index  = nef_out_frame_unassigned.index + nef_out_frame_assigned.index.max()+1
        nef_out_frame_unassigned['index'] = nef_out_frame_unassigned.index + 1

        nef_out_frame_unassigned['chain_code'] = '.'

        output_frame = pd.concat([nef_out_frame_assigned, nef_out_frame_unassigned], axis=0)
        output_frame['merit'] = output_frame['merit'].apply(lambda x: f'%0.3f' % x)
        output_frame['merit'] = output_frame['merit'].replace({'nan': '0.000'})

        loop.add_data(output_frame.to_dict('records'))
        output = str(save_frame)

    else:

        table = []
        for df_index, df_row in assigner.assign_df.iterrows():

            table_row = []
            table.append(table_row)
            for heading in headings:
                table_row.append(df_row[heading])
        output = tabulate(table, tablefmt='plain', headers=headings)


    with _open_or_stdout(args.output_file, 'w') as fp:
        print(output, file=fp)

    logger.info("Finished writing results to %s", args.output_file)
    #### Write chemical shift lists
    if args.shift_output_file is not None:
        assigner.output_shiftlist(args.shift_output_file, args.shift_output_type,
                                  confidence_list=args.shift_output_confidence)


def _import_aa_type_info(args, assigner, importer):
    if args.aa_restraints:
        if args.aa_type == "snaps":
            importer.import_aa_type_info_file(args.aa_restraints[0])
            assigner.pars["use_ss_class_info"] = True
        elif args.aa_type == 'nef':
            file_name = args.aa_restraints[0] if args.aa_restraints else args.shift_file
            importer.import_aa_type_info_nef(file_name)
            assigner.pars["use_ss_class_info"] = True
        else:
            print(f'wrong file format [{args.aa_type}] for aa type info should be one of nef or snaps', file=sys.stderr)
            sys.exit(1)
    else:
        assigner.pars["use_ss_class_info"] = False


def _setup_logger(args):
    
    # Create a logger
    logger = logging.getLogger("SNAPS")
    
    if args.log_file is not None:       
        logger.setLevel(logging.DEBUG)
        # Create a log handler that writes to a specific file.
        # In principle you could have multiple handlers, but here I just have one.
        # Need to explicitly define a handler so it can be explicitly closed
        # once the analysis is complete.
        log_handler = logging.FileHandler(args.log_file, mode='w')
        log_handler.setLevel(logging.DEBUG)
        # log_handler.setFormatter(logging.Formatter("%(levelname)s %(asctime)s %(module)s %(funcName)s %(message)s"))
        log_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(log_handler)

    else:
        logging.basicConfig(level=logging.ERROR)
        # logging.basicConfig(level=logging.DEBUG)
    return logger


#%% Run the actual script
if __name__ == '__main__':
    import sys

    run_snaps(sys.argv[1:])
