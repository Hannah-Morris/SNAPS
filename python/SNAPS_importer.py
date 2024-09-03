# -*- coding: utf-8 -*-
"""
Defines a class with functions related to importing peak lists and shift lists.

@author: Alex
"""
from textwrap import dedent

from pynmrstar import Entry
import pandas as pd
from Bio.SeqUtils import seq1
from math import sqrt

from NEF_reader import read_nef_obs_shifts_from_file_to_pandas, TRANSLATIONS_3_1_PROTEIN, _split_path_and_frame

POSSIBLE_1LET_AAS_STR = "ACDEFGHIKLMNPQRSTVWY"

class SnapsImportException(Exception):
    ...
# import nmrstarlib

class SNAPS_importer:
    # Attributes
    #    peak lists = {}
    #    assignments = {}
    #    roots = None
    #    peaks = None
    #    obs = None
    
    def __init__(self):
        self.peaklists = {}
        #assignments = {}
        self.roots = None
        self.obs = None
    
    # Functions
    def import_hsqc_peaks(self, filename, filetype):
        """ Import a peak list for an HSQC spectrum. 
        
        filename: Path to file containing an HSQC peak list.
        filetype: Allowed values are "ccpn", "sparky", "xeasy" or "nmrpipe"
        """
        # Import from file
        if filetype=="ccpn":
            hsqc = pd.read_table(filename) 
            hsqc = hsqc[["Assign F1","Position F1","Position F2", "Height"]]
            hsqc.columns = ["SS_name","H","N","Height"]      
            #hsqc.index = hsqc["SS_name"]
        elif filetype=="sparky":
            hsqc = pd.read_table(filename, sep=r"\s+")
            hsqc.columns = ["SS_name","H","N","Height"]
            # If assigned, Name has format "A123HN-A123N"
            # If unassigned, Name column contains "?-?"
            # Get just the first part before the hyphen
            tmp = list(zip(*hsqc["SS_name"].str.split("-")))[0]
            hsqc["SS_name"] = tmp
            
            # Give arbitrary names to the unassigned peaks
            N_unassigned = sum(hsqc["SS_name"]=="?")
            hsqc.loc[hsqc["SS_name"]=="?", "SS_name"] = list("x"+
                    pd.Series(range(N_unassigned)).astype(str))
            
        elif filetype=="xeasy":
            hsqc = pd.read_table(filename, sep=r"\s+", comment="#", header=None,
                                 usecols=[0,1,2,5], 
                                 names=["SS_name","H","N","Height"])
            hsqc["SS_name"] = "x" + hsqc["SS_name"].astype(str)
        elif filetype=="nmrpipe":
            # Work out where the column names and data are
            with open(filename, 'r') as f:
                for num, line in enumerate(f, 1):
                    if line.find("VARS")>-1:
                        colnames_line = num
                        colnames = line.split()[1:]
            
            hsqc = pd.read_table(filename, sep=r"\s+", skiprows=colnames_line+1,
                                names=colnames)
            hsqc = hsqc[["INDEX", "ASS", "X_PPM", "Y_PPM", "HEIGHT"]]
            hsqc.columns = ["ID", "SS_name", "H", "N", "Height"]
            # Use ASS as SS_name if available, otherwise use ID
            hsqc.loc[hsqc["SS_name"]!=None,"SS_name"] = \
                hsqc.loc[hsqc["SS_name"]!=None,"ID"].astype(str)
            hsqc = hsqc[["SS_name", "H", "N", "Height"]]
        else:
            print("import_hsqc_peaks: invalid filetype '%s'." % (filetype))
            return(None)
        
        # Check whether columns go mixed up somehow, and correct if so
        if hsqc["H"].mean() > hsqc["N"].mean():
            tmp = hsqc["N"]
            hsqc["N"] = hsqc["H"]
            hsqc["H"] = tmp
        
        hsqc.index = hsqc["SS_name"]
        
        self.peaklists["hsqc"] = hsqc
        self.roots = hsqc
        
        return(self.roots)
        
    def import_3d_peaks(self, filename, filetype, spectrum, 
                        assign_nearest_root=False):
        """Import a 3D peak list in various formats
        
        filetype: one of "ccpn", "sparky", "xeasy" or "nmrpipe"
        spectrum: one of "hnco", "hncaco", "hnca", "hncoca", "hncacb",
            "hncocacb" or "hnha"
        assign_nearest_root: If True, this will assign each peak to the spin 
            system with the closest root (hsqc) peak. If False, peaks will be 
            assigned to spin systems based on information in the original file. 
            In this case, the proton assignment is used for CCPN and Sparky, 
            while the ASS column is used for nmrPipe. Xeasy peaklists alone 
            do not seem to contain assignment information.
        """
        if filetype == "ccpn":
            peaks = pd.read_table(filename,
                                  usecols=["Position F1","Position F2",
                                           "Position F3","Assign F1",
                                           "Assign F2","Assign F3",
                                           "Height"])
            peaks.columns = ["F1","F2","F3","A1","A2","A3","Height"]
        elif filetype == "sparky":
            peaks = pd.read_table(filename, sep=r"\s+")
            peaks.columns=["Name","F1","F2","F3","Height"]
            peaks["A1"], peaks["A2"], peaks["A3"] = list(zip(
                                                *peaks["Name"].str.split("-")))
            #return(peaks)
        elif filetype == "xeasy":
            peaks = pd.read_table(filename, sep=r"\s+", comment="#", header=None,
                                 usecols=[1,2,3,6], 
                                 names=["F1","F2","F3","Height"])
            peaks["SS_name"] = None
        elif filetype == "nmrpipe":
            # Work out where the column names and data are
            with open(filename, 'r') as f:
                for num, line in enumerate(f, 1):
                    if line.find("VARS")>-1:
                        colnames_line = num
                        colnames = line.split()[1:]
                        
            peaks = pd.read_table(filename, sep=r"\s+", skiprows=colnames_line+1,
                                names=colnames)
            peaks = peaks[["ASS", "X_PPM", "Y_PPM", "Z_PPM", "HEIGHT"]]
            peaks.columns = ["SS_name", "F1", "F2", "F3", "Height"]
            
        else:
            print("import_3d_peaks: invalid filetype '%s'." % (filetype))
            return(None)
            
        # Work out which column contains which dimension
        dim = {}
        for i in ["1","2","3"]:
            if peaks["F"+i].mean() > 150:
                dim["C"] = i
            elif peaks["F"+i].mean() > 100:
                dim["N"] = i
            elif peaks["F"+i].mean() > 15:
                dim["C"] = i
            elif peaks["F"+i].mean() > 6:
                dim["H"] = i
            elif peaks["F"+i].mean() > 0:
                dim["HA"] = i
            else: 
                dim["Unknown"] = i
        
        # Check that the correct dimensions were identified
        # Also check that spectrum argument is valid
        if spectrum in ["hnco", "hncaco", "hnca","hncoca","hncacb","hncocacb"]:
            if set(dim.keys()) != set(["H","N","C"]):
                print("Error: couldn't identify "+spectrum+" columns.") 
        elif spectrum == "hnha":
            if set(dim.keys()) != set(["H","N","HA"]):
                print("Error: couldn't identify "+spectrum+" columns.")
        else:
            print("Invalid value of argument: spectrum.")
        
        # Populate SS_name column
        if filetype in ["ccpn", "sparky"]:
            peaks["SS_name"] = peaks["A"+dim["H"]]
        elif filetype == "nmrpipe":
            peaks["SS_name"] = peaks["ASS"]

        # Choose and rename columns. 
        peaks = peaks[["SS_name", "F"+dim["H"], "F"+dim["N"], "F"+dim["C"], 
                       "Height"]]
        peaks.columns = ["SS_name", "H", "N", "C", "Height"]
        peaks["Spectrum"] = spectrum
        
        # If assign_nearest_root, find closest root resonance for each peak 
        # and set that as SS_name.
        if assign_nearest_root or spectrum=="xeasy":
            peaks["SS_name"] == None
            
            roots = self.roots
            for i in peaks.index:
                delta = ((roots["H"]-peaks.loc[i,"H"])**2 
                         + (0.2*(roots["N"]-peaks.loc[i,"N"]))**2).apply(sqrt)
                peaks.loc[i, "SS_name"] = roots.loc[delta.idxmin(), "SS_name"]

        # Also, only keep spin systems that are in self.roots
        #peaks = peaks.loc[peaks["SS_name"].isin(self.roots["SS_name"])]
        
        # Add peaks to the peaklist dictionary
        self.peaklists[spectrum] = peaks
            
        return(self.peaklists[spectrum])
        
    def find_shifts_from_peaks(self):
        """ Work out chemical shifts for each spin system from peak lists
        
        Will use all spectra in the self.peaklists dictionary. The following
        spectra are supported: hnco, hncaco, hnca, hncoca, hncacb, hncocacb.
        """
        # Possible extension: could put a parameter to control hncacb sign interpretation
        obs = pd.DataFrame({"SS_name": self.roots["SS_name"], 
                            "H": self.roots["H"],
                            "N": self.roots["N"]})
        obs.index = obs["SS_name"]
        obs.index.name = None
        for spec in self.peaklists.keys():
            peaks = self.peaklists[spec]
            for ss in obs["SS_name"]:
                if ss in peaks["SS_name"].values:
                    ss_peaks = peaks.loc[peaks["SS_name"]==ss,:]
                    if spec=="hnco":
                        # Set C_m1 shift to strongest peak in this spin system.         
                        i = ss_peaks.loc[:,"Height"].idxmax()
                        obs.loc[ss, "C_m1"] = ss_peaks.loc[i, "C"]
                    elif spec=="hncaco":
                        # Set C shift to strongest peak in this spin system.         
                        i = ss_peaks.loc[:,"Height"].idxmax()
                        obs.loc[ss, "C"] = ss_peaks.loc[i, "C"]
                    elif spec=="hncoca":
                        # Set CA_m1 shift to strongest peak in this spin system.         
                        i = ss_peaks.loc[:,"Height"].idxmax()
                        obs.loc[ss, "CA_m1"] = ss_peaks.loc[i, "C"]
                    elif spec=="hnca":
                        # Set CA shift to strongest peak in this spin system.         
                        i = ss_peaks.loc[:,"Height"].idxmax()
                        obs.loc[ss, "CA"] = ss_peaks.loc[i, "C"]
                    elif spec=="hncocacb":
                        # Use a simple heuristic to guess if peak is CA or CB:
                        # - If only 1 peak, CA if shift >41 ppm, otherwise CB
                        # - If >1 peak, only keep the two with highest 
                        #   (absolute) intensity. 
                        # - If both >48 ppm, the largest shift is CB. 
                        #   Otherwise, the smallest shift is CB
                        if sum(peaks["SS_name"]==ss)==1:
                            if ss_peaks["C"].item()>41:
                                obs.loc[ss, "CA_m1"] = ss_peaks["C"].item()
                            else:
                                obs.loc[ss, "CB_m1"] = ss_peaks["C"].item()
                        else:
                            ss_peaks["Abs_height"] = ss_peaks["Height"].abs()
                            # Above line throws a SettingWithCopy warning, 
                            # but I can't seem to fix it
                            
                            ss_peaks = ss_peaks.sort_values(by="Abs_height",
                                                            ascending=False)
                            ss_peaks = ss_peaks.iloc[0:2,:]
                            C_max = ss_peaks["C"].max()
                            C_min = ss_peaks["C"].min()
                            if C_max>48 and C_min>48:
                                obs.loc[ss,"CA_m1"] = C_min
                                obs.loc[ss,"CB_m1"] = C_max
                            else:
                                obs.loc[ss,"CA_m1"] = C_max
                                obs.loc[ss,"CB_m1"] = C_min
                    elif spec=="hncacb":
                        # Use a simple heuristic to guess if peak is CA or CB:
                        # - If only 1 peak, CA if shift >41 ppm, otherwise CB
                        # - If >1 peak, only keep the two with highest 
                        #   (absolute) intensity. 
                        # - If strongest peak is 41-48 ppm and >twice height of
                        #   next highest, then it's glycine CA
                        # - Else, if both >48 ppm, the largest shift is CB. 
                        #   Otherwise, the smallest shift is CB
                        if sum(peaks["SS_name"]==ss)==1:
#                            print(ss_peaks)
#                            print(ss_peaks["C"])
#                            print(ss_peaks["C"].item())
                            if ss_peaks["C"].item()>41:
                                obs.loc[ss, "CA"] = ss_peaks["C"].item()
                            else:
                                obs.loc[ss, "CB"] = ss_peaks["C"].item()
                        else:
                            ss_peaks["Abs_height"] = ss_peaks["Height"].abs()
                            # Above line throws a SettingWithCopy warning, 
                            # but I can't seem to fix it
                            ss_peaks = ss_peaks.sort_values(by="Abs_height",
                                                            ascending=False)
                            ss_peaks = ss_peaks.iloc[0:2,:]
                            C_max = ss_peaks["C"].max()
                            C_min = ss_peaks["C"].min()
                            if (ss_peaks.iloc[0,:]["C"]>41 and 
                                ss_peaks.iloc[0,:]["C"]<48 and 
                                ss_peaks.iloc[0,:]["Abs_height"] >
                                2*ss_peaks.iloc[1,:]["Abs_height"]):
                                
                                obs.loc[ss,"CA"] = ss_peaks.iloc[0,:]["C"]
                            elif C_max>48 and C_min>48:
                                obs.loc[ss,"CA"] = C_min
                                obs.loc[ss,"CB"] = C_max
                            else:
                                obs.loc[ss,"CA"] = C_max
                                obs.loc[ss,"CB"] = C_min
                    else:
                        print("Spectrum type %s not recognised" % spec)
                        break
            
        self.obs = obs
        return(self.obs)
            
    def import_obs_shifts(self, filename, filetype, SS_num=False, chain='A'):
        """ Import a chemical shift list
        
        filename: Path to text file containing chemical shifts.
        filetype: Allowed values are "snaps", "ccpn", "sparky", "xeasy", 
            "nmrpipe" or "mars"
            The "ccpn" option is for importing a Resonance table exported from 
            Analysis v2.x. The "snaps" option is for importing an unassigned 
            shift table previously exported from SNAPS
        SS_num: If true, will extract the longest number from the SS_name and 
        treat it as the residue number. Without this, it is not possible to get
        the i-1 shifts for each spin system.
            
        """
        # Import from file
        if filetype=="snaps":
            obs = pd.read_table(filename)
        elif filetype=="ccpn":
            obs = pd.read_table(filename)
            obs = obs.loc[:,["Residue", "Assign Name", "Shift"]]
            obs.columns = ["SS_name", "Atom_type", "Shift"]
            obs["Atom_type"] = obs["Atom_type"].str.upper()
        elif filetype=="sparky":
            obs = pd.read_table(filename, sep=r"\s+")
            obs = obs.loc[:,["Group", "Atom", "Shift"]]
            obs.columns = ["SS_name", "Atom_type", "Shift"]
            obs.loc[obs["Atom_type"]=="HN", "Atom_type"] = "H"
        elif filetype=="xeasy":
            obs = pd.read_table(filename, sep=r"\s+",
                                header=None, na_values="999.000",
                                names=["i","Shift","SD","Atom_type","SS_name"])
            obs = obs.loc[:, ["SS_name", "Atom_type", "Shift"]]
            obs["SS_name"] = obs["SS_name"].astype(str)
            obs = obs.dropna(subset=["Shift"])
            obs.loc[obs["Atom_type"]=="HN", "Atom_type"] = "H"
        elif filetype=="nmrpipe":
            # Work out where the column names and data are
            with open(filename, 'r') as f:
                for num, line in enumerate(f, 1):
                    if line.find("VARS") > -1:
                        colnames_line = num
            
            obs = pd.read_table(filename, sep=r"\s+", skiprows=colnames_line+1,
                                names=["SS_name","Res_type","Atom_type","Shift"])
            obs = obs.loc[:, ["SS_name", "Atom_type", "Shift"]]
            obs["SS_name"] = obs["SS_name"].astype(str)
            obs.loc[obs["Atom_type"] == "HN", "Atom_type"] = "H"
        elif filetype == "mars":
            obs_wide = pd.read_table(filename, sep=r"\s+", na_values="-")
            obs_wide = obs_wide.rename(columns={"CO":"C","CO-1":"C_m1",
                                                "CA-1":"CA_m1","CB-1":"CB_m1"})
            obs_wide = obs_wide.drop(columns="HA-1")
            obs_wide["SS_name"] = obs_wide.index.astype(str)
            obs = obs_wide.melt(id_vars="SS_name", 
                                value_vars=["H","HA","N","C","CA",
                                            "CB","C_m1","CA_m1","CB_m1"], 
                                var_name="Atom_type", value_name="Shift")
        elif filetype =='nef':
            obs = read_nef_obs_shifts_from_file_to_pandas(filename, chain)

#        elif filetype == "nmrstar":
#            tmp = nmrstarlib.read_files(filename)
#            return(tmp)
        else:
            print("import_obs_shifts: invalid filetype '%s'." % (filetype))
            return(None)
        
        # Restrict to backbone atom types
        obs = obs.loc[obs["Atom_type"].isin(["H","HA","N","C","CA","CB",
                                             "C_m1","CA_m1","CB_m1"]),:]
        
        # Convert from long to wide
        obs = obs.pivot(index="SS_name", columns="Atom_type", values="Shift")
        obs.insert(0, "SS_name", obs.index.values)
        
        # Extract residue number from SS_name (if present), and get m1 shifts
        if SS_num:
            obs["Res_N"] = obs["SS_name"].str.extract(r"(\d+)").astype(int)
            
            obs.index = obs["Res_N"]
            obs_m1 = obs[list({"C", "CA", "CB"}.intersection(obs.columns))]
            obs_m1.index = obs_m1.index+1
            obs_m1.columns = obs_m1.columns + "_m1"
            obs = pd.merge(obs, obs_m1, how="left", 
                           left_index=True, right_index=True)
            obs = obs.drop(columns="Res_N")
            
            # Set index back to SS_name
            obs.index = obs["SS_name"]
            obs.index.name = None

        # the index name shouldn't match a column name!
        obs.index.name = None
        self.obs = obs
        return self.obs

    def import_aa_type_info_nef(self, file_name:str):
        """ Add amino acid type information to previously-imported observed
        shifts

        file_name: a a path to a NEF file containing amino acid information
                    with a frame name attached eg xyz.nef:frame_name

        """

        def is_int(str):
            result = True
            try:
                int(str)
            except Exception:
                result = False
            return result

        RESIDUE_TYPES_FRAME = 'nefpls_residue_types'
        RESIDUE_TYPES_TAG = '_nefpls_residue_type'
        REQUIRED_TAG_SET = set([f'{RESIDUE_TYPES_TAG}.{tag}' for tag in 'chain_code sequence_code residue_type'.split()])
        LEN_RESIDUE_TYPES = len(RESIDUE_TYPES_FRAME)

        file_name, frame_name = _split_path_and_frame(file_name)
        entry = Entry.from_file(file_name)

        frames = entry.get_saveframes_by_category('nefpls_residue_types')
        frames = [frame for frame in frames if frame.name[LEN_RESIDUE_TYPES:].strip('_') == frame_name]

        restraints = {}

        for frame in frames:
            if RESIDUE_TYPES_TAG in frame:
                loop = frame.get_loop(RESIDUE_TYPES_TAG)

                if not REQUIRED_TAG_SET.issubset(loop.get_tag_names()):
                    missing_tags = REQUIRED_TAG_SET - set(loop.get_tag_names())
                    msg = \
                    f"""
                        Missing required tags in the frame {frame_name} {missing_tags}, the missing tags are:
                        {missing_tags}
                    """
                    raise SnapsImportException(msg)

                chain_code_index = loop.tag_index('chain_code')
                sequence_code_index = loop.tag_index('sequence_code')
                residue_type_index = loop.tag_index('residue_type')

                for row in loop:
                    chain_code = row[chain_code_index]
                    sequence_code = row[sequence_code_index]
                    residue_type = row[residue_type_index]

                    sequence_code_offset = 0
                    if not isinstance(sequence_code, int) and '-' in sequence_code:
                        sequence_code_fields =  sequence_code.split('-')
                        if is_int(sequence_code[-1]):
                            sequence_code_offset = int(sequence_code_fields[:-1])
                            sequence_code = '-'.join(sequence_code_fields[:-1])

                    # TODO: we should select a chain!
                    ss_name = f'{sequence_code}'


                    if residue_type in TRANSLATIONS_3_1_PROTEIN:
                        residue_type = TRANSLATIONS_3_1_PROTEIN[residue_type]
                    else:
                        msg = \
                        f"""
                            Non protein residue type {residue_type} in frame {frame_name}
                        """
                        raise SnapsImportException(msg)


                    key = ss_name, sequence_code_offset
                    residue_types = restraints.setdefault(key, '')
                    residue_types += residue_type
                    restraints[key] = residue_types


        residue_restraints = [[ss, residue_types, 'in', offset] for (ss, offset), residue_types in restraints.items()]
        residue_type_frame = pd.DataFrame(residue_restraints, columns='SS_name AA Type Offset'.split())

        #TODO: why is this title case
        residue_type_frame['SS_name'] = residue_type_frame['SS_name'].str.title()
        return self._import_aa_type_info(residue_type_frame, source=f'{entry.entry_id}.{frame_name}')

    def import_aa_type_info_file(self, file_name):
        """ Add amino acid type information to previously-imported observed
        shifts

        file_name: Path to a file containing amino acid information
            This should have the following columns:
            SS_name   AVI     in   Offset
            SS_name_1   AVI   in   # to set AVI as the only allowed aa types
            SS_name_2   T     ex   # to exclude T from the allowed aa types
        """
        # Import file
        df = pd.read_table(file_name, sep=r"\s+", comment="#", header=0)

        if 'Offset' not in df.columns:
            df['Offset'] = 0

        return self._import_aa_type_info(df, file_name)

    def _import_aa_type_info(self, aa_info_df, source):
        """ Add amino acid type information to previously-imported observed 
        shifts

        source: a pandas data frame with amino acid information
            This should have the following columns:
            SS_name_1   AVI   in   # to set AVI as the only allowed aa types
            SS_name_2   T     ex   # to exclude T from the allowed aa types
            offset: either "i" or "i-1". Whether the aa type restriction
            applies to the i spin system or to the preceeding i-1 spin system.
        """
        self.check_required_headings_and_raise_if_bad(aa_info_df, source)
        self.check_aa_letters_correct_and_raise_if_bad(aa_info_df)
        self.check_type_column_categories_and_raise_if_bad(aa_info_df)

        if 'Offset' in aa_info_df.columns:
            self._check_bad_offset_raise_if_bad(aa_info_df)

            result_df_0 = aa_info_df[aa_info_df['Offset'] == 0]
            self._process_aa_type_info_single_offset(result_df_0, 0)

            result_df_m1 = aa_info_df[aa_info_df['Offset'] == -1]
            self._process_aa_type_info_single_offset(result_df_m1, -1)
        else:
            self._process_aa_type_info_single_offset(aa_info_df, 0)
        return self.obs

    def _check_bad_offset_raise_if_bad(self, aa_info_df):
        result_df_other = aa_info_df[(aa_info_df['Offset'] != 0) & (aa_info_df['Offset'] != -1)]
        msg = 'offset can only be 0 or -1'
        if len(result_df_other.index):
            raise SnapsImportException(msg)

    def _process_aa_type_info_single_offset(self, aa_info_df, offset):
        if offset == 0:
            ss_class_col = "SS_class"
        elif offset == -1:
            ss_class_col = "SS_class_m1"
        else:
            raise SnapsImportException(f"in import_aa_type_info invalid value of offset: must be 0 or -1 i got "
                                       f"{offset}")

        # puts aa into ss class column
        aa_info_df[ss_class_col] = aa_info_df["AA"]

        self.check_spin_systems_from_obs_raise_if_bad(aa_info_df)

        # For rows with Type=="in", the SS_class is the same as AA
        # For rows with Type=="ex", the SS_class is all aminos *except* AA

        ex_mask = (aa_info_df["Type"] == "ex")
        for row_index in aa_info_df.index[ex_mask]:
            expected_aas = POSSIBLE_1LET_AAS_STR
            for aa_1let in aa_info_df.loc[row_index, "AA"]:
                expected_aas = expected_aas.replace(aa_1let, "")
            aa_info_df.loc[row_index, ss_class_col] = expected_aas
        aa_info_df.index = aa_info_df["SS_name"]

        # Create SS_class column in obs DataFrame if it doesn't already exist.

        if ss_class_col not in self.obs.columns:
            self.obs[ss_class_col] = POSSIBLE_1LET_AAS_STR
        # else:
        #     raise SnapsImportException("SS_class column needs to be present in the Obs DataFrame")

        # Write SS_class info into obs data frame. Overwrite any previous info
        # for these spin systems, but keep SS_class info for any spin systems
        # not in df
        self.obs.loc[aa_info_df.index, ss_class_col] = aa_info_df.loc[:, ss_class_col]

        # Nan's can be any amino acid
        self.obs[ss_class_col] = self.obs[ss_class_col].fillna(POSSIBLE_1LET_AAS_STR)

    def check_spin_systems_from_obs_raise_if_bad(self, aa_info_df):
        expected_obs = set()
        if 'SS_name' in self.obs.columns:
            expected_obs.update(set(self.obs['SS_name']))
        if 'SS_name_m1' in self.obs.columns:
            expected_obs.update(set(self.obs['SS_name_m1']))
        given_obs = [elem in expected_obs for elem in aa_info_df['SS_name']]
        if not all(given_obs):
            missing_names = set(aa_info_df['SS_name']) - expected_obs
            msg = \
            f"""
                Incorrect data given, unexpected spin system in aa type restraints.
                The restraint spin systems should be in the chemical shift list.
                The missing spin systems are:
                {', '.join(missing_names)}
            """
            msg=dedent(msg)
            raise SnapsImportException(msg)

        # puts aa into ss class column

    def check_aa_letters_correct_and_raise_if_bad(self, aa_info_df):

        expected_1let_aa_set = set(POSSIBLE_1LET_AAS_STR)
        all_aa_1lets_ok = [set(elem).issubset(expected_1let_aa_set) for elem in aa_info_df['AA']]

        if not all(all_aa_1lets_ok):
            msg = "AA type letters incorrect, Amino Acid letters can only be: 'ACDEFGHIKLMNPQRSTVWY'"
            raise SnapsImportException(msg)

    def check_type_column_categories_and_raise_if_bad(self, aa_info_df):

        ex_mask = (aa_info_df["Type"] == "ex")
        in_mask = (aa_info_df["Type"] == "in")
        expected_type_column = ex_mask ^ in_mask
        if not all(expected_type_column):
            msg = "Type column row error: 'Type' column rows can only contain 'in' or 'ex'"
            raise SnapsImportException(msg)

    def check_required_headings_and_raise_if_bad(self, df, filename):
        expected_column_names = {"SS_name", "AA", "Type"}
        found_column_names = set(df.columns)

        # if 'offset' in found_column_names:
        #     found_column_names.loc[:, found_column_names.columns != 'offset'])

        if not expected_column_names.issubset(found_column_names):
            bad_column_names = found_column_names - expected_column_names
            if 'Offset' in bad_column_names:
                bad_column_names.remove('Offset')
            bad_column_names = ', '.join(bad_column_names)
            expected_column_names = ', '.join(expected_column_names)
            msg = f"""\
                   Unexpected column name(s) [{bad_column_names}]
                   in file {filename}
                   expected column names are: {expected_column_names}"
               """
            raise SnapsImportException(msg)

    def import_testset_shifts(self, filename, remove_Pro=True,
                          short_aa_names=True, SS_class=None, SS_class_m1=None):
        """ Import observed chemical shifts from testset data
        
        This function is intended for use with test data only, and is unlikely 
        to work well on 'real' data.
        
        filename: The simplified BMRB file containing observed shift info.
        remove_Pro: If True, remove proline residues from output
        short_aa_names: If True, single letter aa codes are used, otherwise 3
            letter codes are used
        SS_class: Either None or a list of strings, each of which is a list of 
            amino acids (eg. ["VIA","G","S","T","DN","FHYWC","REKPQML"] would 
            give the HADAMAC classes). If not None, a column SS_class will be 
            created which gives the class containing the residue type.
        SS_class_m1: as above, but for the i-1 residue.
        
        """
        # Import the observed chemical shifts
        obs_long = pd.read_table(filename)
        obs_long = obs_long[["Residue_PDB_seq_code","Residue_label",
                             "Atom_name","Chem_shift_value"]]
        obs_long.columns = ["Res_N","Res_type","Atom_type","Shift"]
        # Convert residue type to single-letter code
        if short_aa_names: 
            obs_long["Res_type"] = obs_long["Res_type"].apply(seq1)
            obs_long["SS_name"] = (obs_long["Res_N"].astype(str) + 
                    obs_long["Res_type"])
            obs_long["SS_name"] = obs_long["SS_name"].str.rjust(5)
        else:
            obs_long["SS_name"] = (obs_long["Res_N"].astype(str) + 
                                    obs_long["Res_type"])
            obs_long["SS_name"] = obs_long["SS_name"].str.rjust(7)
            obs_long["Res_type"] = obs_long["Res_type"].apply(seq1)
        obs_long = obs_long.reindex(columns=["Res_N","Res_type","SS_name",
                                             "Atom_type","Shift"])

        # Convert from long to wide
        obs = obs_long.pivot(index="Res_N", columns="Atom_type", 
                             values="Shift")
        
        # Add the other columns back in
        tmp = obs_long[["Res_N","Res_type","SS_name"]]
        tmp = tmp.drop_duplicates(subset="SS_name")
        tmp.index = tmp["Res_N"]
        obs = pd.concat([tmp, obs], axis=1)
        
        # Make columns for the i-1 observed shifts of C, CA and CB
        obs_m1 = obs[list({"C", "CA", "CB", "Res_type"}.intersection(obs.columns))]
        obs_m1.index = obs_m1.index+1
        obs_m1.columns = obs_m1.columns + "_m1"
        obs = pd.merge(obs, obs_m1, how="left", left_index=True, 
                       right_index=True)
        
        # Restrict to specific atom types
        atom_set = {"H", "N", "C", "CA", "CB", "C_m1", "CA_m1", "CB_m1", "HA"}
        obs = obs[["Res_N", "Res_type", "Res_type_m1", "SS_name"]+
                  list(atom_set.intersection(obs.columns))]
        
        # Add SS_class information
        if SS_class is not None:
            obs["SS_class"]=obs["Res_type"]
            for g in SS_class:
                obs["SS_class"] = obs["SS_class"].str.replace("["+g+"]", g)
        if SS_class_m1 is not None:
            obs["SS_class_m1"]=obs["Res_type_m1"]
            for g in SS_class_m1:
                obs["SS_class_m1"] = obs["SS_class_m1"].str.replace("["+g+"]", g)
        
        obs.index = obs["SS_name"]
        obs.index.name = None
        
        if remove_Pro:
            # Remove prolines, as they wouldn't be observed in a real spectrum
            obs = obs.drop(obs.index[obs["Res_type"].isin(["PRO", "P"])])
        
        self.obs = obs
        return(self.obs)
    
    def export_obs_shifts(self, filename):
        """ Export a chemical shift list.
        
        The purpose of this function is to make a shift list which the user can 
        manually modify/correct, and then reimport.
        """
        
        df = self.obs.melt(id_vars="SS_name", 
                      value_vars=set(self.obs.columns).intersection({"H", "N",
                                    "HA", "C", "CA", "CB", "C_m1", "CA_m1", "CB_m1"}),
                      var_name="Atom_type", value_name="Shift")
        df = df.sort_values(by=["SS_name", "Atom_type"])
            
        df.to_csv(filename, sep="\t", float_format="%.3f", index=False)
        
        return(df)
        


