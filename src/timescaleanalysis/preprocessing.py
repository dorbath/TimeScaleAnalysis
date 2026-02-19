import warnings

from genericpath import isfile, isdir
import numpy as np
import os
import timescaleanalysis.io as io
import sys


class Preprocessing:
    """Preprocessing class to initiate and prepare the data for the evaluation

    Parameters
    ----------
    data_dir: str, directory to studied data where all files that fullfil
            data_dir* (path/to/file*) are used
    data_arr: list, list of loaded trajectories (each trajectory is a np.array)
    data_mean: np.array, mean of loaded trajectories
    data_sem: np.array, standard error of the mean of loaded trajectories
    n_steps: int, maximum number of steps in the data
    folder_prefix: str, prefix of folder path to data
    input_directories: list, list of input files/trajectories in folder_prefix
             with correct prefix
    labels_lst: list, list of labels for each observable in the data

    Attributes
    ----------
    sim_file: str, directory to file with simulation parameters
            (usually .mdp file, entries 'dt = ' and 'nstxtcout = ' are needed)
    time: array, time vector of the data
    label_file: str, file directory with label for each observable in the data
    """
    DEFAULTS = {
        "sim_file": None,
        "times": None,
        "label_file": None,
    }

    def __init__(self, data_dir, **kwargs):
        self.data_dir = data_dir
        self.data_arr = []
        self.data_mean = None
        self.data_sem = None
        self.n_steps = 0
        self.folder_prefix = ''
        self.input_directories = None
        self.labels_lst = None

        self.options = self.DEFAULTS | kwargs

    def generate_input_trajectories(self):
        """Get all files/trajectories in 'data_path' with the correct prefix.
        All files that fulfill data_path* are taken as input.
        """

        # Isolate folder path and trajectory prefix
        data_path_split = self.data_dir.split("/")
        folder_suffix = data_path_split[-1]
        for n in range(len(data_path_split)-1):
            self.folder_prefix += data_path_split[n]+'/'
        # If prefix matches exactly with a file, take this file as input
        # Otherwise take all files with matching prefix
        if isfile(self.folder_prefix+'/'+folder_suffix):
            self.input_directories = [
                folder_suffix
            ]
        else:
            self.input_directories = [
                path for path in os.listdir(self.folder_prefix)
                if path.startswith(folder_suffix)
            ]

    def load_absorption_spectra(self):
        """Load single absorption spectrum of experimental
        data and corresponding times and frequencies"""
        # TODO Add here the 1D (Exp data) part

    def load_trajectories(self,
                          n_traj_conc: int = None,
                          averaged: bool = False):
        """Load trajectories from 'data_dir' path
        There are three ways of loading data:
            1) Multiple trajectories, each is assumed to be a
                    separate time trace. (default)
            2) Concantenated trajectories, where multiple trajectories
                    are concatenated in one file.
            3) Averaged trajectories with standard error of the mean
                    in second column.

        Parameters
        ----------
        n_traj_conc: int, number of trajectories that are concatenated in file.
                Trajectories must be of same length (default: None)
        averaged: bool, True if the input trajectory is already averaged.
                Trajectory must contain two columns, 1st with data points,
                2nd with standard error of the mean (SEM). (default: False)
        """

        def _load_single_file(folder_dir: str,
                              file_name: str,
                              precision=np.float32,
                              averaged: bool = False,
                              verbose: bool = False):
            """Load a single trajectory file
            Delimiter is assumed to be whitespace,
            comment lines starting with '#' are ignored.

            Parameters
            ----------
            folder_dir: str, path to folder with trajectories
            file_name: str, name of single file/trajectory in folder_dir
            precision: np.float16/32/64, precision of loaded data
                    (default: np.float32)
            averaged: bool, True if the input trajectory is already averaged.
            verbose: bool, print loading information (default: False)

            Return
            ------
            data_np: np.array, contains loaded data.
                     Each column corresponds to one observable
            """
            if precision not in (np.float16, np.float32, np.float64):
                raise TypeError(
                    "precision must be a NumPy float dtype (np.float16/32/64)"
                )
            if not isfile(folder_dir+'/'+file_name):
                raise FileNotFoundError(
                    f"File {folder_dir+'/'+file_name} does not exist!"
                )

            temp_load = np.loadtxt(
                folder_dir+'/'+file_name,
                dtype=precision,
                comments='#',
            )
            if not temp_load.ndim <= 2:
                raise ValueError(
                    "Loaded data must be 1D or 2D array!"
                    f" Loaded data has {temp_load.ndim} dimensions!"
                )
            # For 2D array, each column is assumed to be
            # a coordinate of the same trajectory
            if temp_load.ndim == 2 and not averaged and verbose:
                warnings.showwarning(
                    "2D array given, each column is assumed as coordinate "
                    "of same trajectory!",
                    category=Warning,
                    filename=folder_dir+'/'+file_name,
                    lineno=0
                )
            return temp_load

        def _load_averaged_trajectory():
            if len(self.input_directories) != 1:
                raise ValueError(
                    "If trajectory is already averaged, "
                    "only one file should be provided as input!"
                )
            temp_traj = _load_single_file(
                self.folder_prefix,
                self.input_directories[0],
                np.float32,
                averaged=True
            )
            if temp_traj.ndim != 2 or temp_traj.shape[1] != 2:
                raise Exception(
                    "Averaged trajectory is assumed. Must contain two columns,"
                    " first column with data points, "
                    "second with standard error of the mean (SEM)!"
                )
            # Data points are in first column
            # Standard error of the mean in second column
            self.data_arr.append(temp_traj)
            self.n_steps = len(temp_traj)

        def _load_concatenated_trajectories(n_traj_conc: int):
            if not isinstance(n_traj_conc, int):
                raise TypeError("n_traj_conc must be an integer")
            if n_traj_conc <= 0:
                raise ValueError("n_traj_conc must be a positive integer")

            for inDir in self.input_directories:
                print(inDir)
                temp_traj = _load_single_file(
                    self.folder_prefix,
                    inDir,
                    np.float16
                )
                if len(temp_traj)/n_traj_conc < 1:
                    raise Exception(
                        f"The number of trajectories to concatenate "
                        f"(n_traj_conc={n_traj_conc}) is larger than the "
                        f"number of frames in a trajectory ({len(temp_traj)})!"
                    )
                length_per_traj = int(len(temp_traj)/n_traj_conc)
                for i in range(n_traj_conc):
                    self.data_arr.append(temp_traj[
                        i*length_per_traj:(i+1)*length_per_traj
                    ])
                    if length_per_traj > self.n_steps:
                        self.n_steps = length_per_traj

        def _load_multiple_trajectories():
            for inDir in self.input_directories:
                print(inDir)
                temp_traj = _load_single_file(
                    self.folder_prefix,
                    inDir,
                    np.float32
                )
                self.data_arr.append(temp_traj)
                if len(temp_traj) > self.n_steps:
                    self.n_steps = len(temp_traj)

        if averaged and n_traj_conc is not None:
            raise Exception(
                "Provide either 'averaged=True' or 'n_traj_conc', not both!"
            )

        if averaged:
            _load_averaged_trajectory()
            self.data_mean = self.data_arr[0][:, 0]
            self.data_sem = self.data_arr[0][:, 1]
        elif n_traj_conc is not None:
            _load_concatenated_trajectories(n_traj_conc)
            # Reshape all trajectories to same length if needed
            temp_lengths = {len(traj) for traj in self.data_arr}
            if len(temp_lengths) > 1:
                self.reshape_same_length()
            self.derive_average_trajectory()
        else:
            _load_multiple_trajectories()
            # Reshape all trajectories to same length if needed
            temp_lengths = {len(traj) for traj in self.data_arr}
            if len(temp_lengths) > 1:
                self.reshape_same_length()
            self.derive_average_trajectory()

    def reshape_same_length(self, insert_nan: bool = True):
        """Ensure that all trajectories are of same length
        by extending/appending constant values.
        This makes calculations with numpy much more efficient.

        Parameters
        ----------
        insert_nan: bool, (false)-> extend final frame of trajectory,
                          (true)-> append np.nan values (default)"""

        def _reshape_single_trajectory(
                trajectory: np.array,
                n_steps: int,
                insert_nan: bool = True):
            """Reshape a single trajectory to match the shape of the longest
               trajectory by extending/appending constant values.

            Parameters
            ----------
            trajectory: array, single trajectory
            n_steps: int, length of longest trajectory
            insert_nan: bool, (false)-> extend final frame of trajectory,
                              (true)-> append np.nan values (default)

            Return
            ------
            temp_np: np.array, reshaped trajectories of same length
            """

            temp_np_traj = np.asarray(trajectory, dtype=np.float32)
            if temp_np_traj.ndim == 1:
                temp_np_traj = temp_np_traj[:, None]
                squeeze = True
            elif temp_np_traj.ndim == 2:
                squeeze = False
            else:
                raise ValueError(
                    "Input data must be 1D or 2D. "
                    f"Loaded data has {temp_np_traj.ndim} dimensions!"
                )
            T, n_cols = temp_np_traj.shape
            reshape_traj = np.zeros((n_steps, n_cols), dtype=np.float32)
            reshape_traj[:T] = temp_np_traj
            if insert_nan:
                reshape_traj[T:] = np.nan
            else:
                reshape_traj[T:] = temp_np_traj[-1]   # broadcast last row
            if squeeze:
                reshape_traj = reshape_traj[:, 0]      # back to 1D
            return reshape_traj

        for idxD, data in enumerate(self.data_arr):
            self.data_arr[idxD] = _reshape_single_trajectory(
                data,
                self.n_steps,
                insert_nan=insert_nan,
            )

    def derive_average_trajectory(self):
        """"Average data for each column"""
        # Correct number of simulations at each timestep (relevant if reshaped)
        n_sim_not_nan = np.sum(~np.isnan(self.data_arr), axis=0)
        self.data_mean = np.nanmean(
            np.array(self.data_arr),
            axis=0, dtype=np.float32
        )
        self.data_sem = np.nanstd(
            np.array(self.data_arr),
            axis=0, ddof=0, dtype=np.float32
        )/np.sqrt(n_sim_not_nan)

    def get_time_array(self):
        """Derive time array for the data"""
        def _load_simulation_parameters():
            """Get simulaitons parameters from .mdp file and create time array
            Be aware of the used units in the file (usually dt is given in ps)
            """
            with open(self.options['sim_file']) as f:
                for ln in f:
                    if not ln.strip():
                        continue
                    if ln.strip().split()[0] == 'dt':
                        # get time step size
                        time_step = float(ln.strip().split()[2])
                    if ln.strip().split()[0] == 'nstxtcout':
                        # get steps at which coordinates are saved
                        coord_step = int(ln.strip().split()[2])
            dt = time_step*coord_step
            self.options['times'] = np.arange(
                0, self.n_steps, 1, dtype=np.float64
            )*dt

        if self.options['sim_file'] is not None:
            if not isfile(self.options['sim_file']):
                raise FileNotFoundError(
                    f"File with simulation parameters "
                    f"{self.options['sim_file']} does not exist!"
                )
            _load_simulation_parameters()
        else:
            print("No simulation parameters provided, generate time array.")
            self.options['times'] = np.arange(
                0, self.n_steps, 1, dtype=np.float64
            )

    def save_preprocessed_data(self, output_path: str = None):
        """Save preprocessed data as json file.
        Preprocessing step can be skipped with saved files

        Parameters
        ----------
        output_path: str, path to save preprocessed data
                     (optional, default: None)
                     If None, data is saved in path
                     folder_prefix/preprocessed_data/preprocessed_data
        """
        if not output_path:
            output_path = self.folder_prefix+'/preprocessed_data/'

        if not isdir(output_path):
            warnings.showwarning(
                f"Output path {output_path} does not exist, creating it!",
                category=Warning,
                filename=output_path,
                lineno=0
            )
            os.makedirs(output_path)

        if (self.data_mean is None or
                self.data_sem.any() is None or
                self.options['times'] is None):
            raise Warning(
                "Not all data is correctly prepared!\n"
                "One of self.data_mean, self.data_sem or "
                "self.options['times'] is not correctly prepared."
            )

        # Verify correct dimensions and shape of data_mean, data_sem and times
        if not np.shape(self.data_mean) == np.shape(self.data_sem):
            raise ValueError(
                "Data mean and sem must have the same shape! "
                f"Data mean shape: {np.shape(self.data_mean)}, "
                f"Data sem shape: {np.shape(self.data_sem)}"
            )
        if not self.data_mean.shape[0] == self.options['times'].shape[0]:
            raise ValueError(
                "Data mean and time array must have the same length! "
                f"Data mean length: {self.data_mean.shape[0]}, "
                f"Time array length: {self.options['times'].shape[0]}"
            )

        # Transform single observable into nested array
        # for consistent handling of multiple observables in later steps
        if self.data_mean.ndim == 1:
            self.data_mean = np.array([self.data_mean], dtype=np.float32).T
            self.data_sem = np.array([self.data_sem], dtype=np.float32).T

        # Load label file
        if self.options['label_file'] is not None:
            if not isfile(self.options['label_file']):
                raise FileNotFoundError(
                    f"Label file {self.options['label_file']} does not exist!"
                )
            labels_lst = np.loadtxt(
                self.options['label_file'],
                dtype=str,
                comments='#',
                ndmin=1
            )
            n_observables = self.data_mean.shape[1]
            if len(labels_lst) != self.data_mean.shape[1]:
                raise ValueError(
                    "Number of labels in label file must match number of "
                    "observables in data! "
                    f"Number of labels: {len(labels_lst)}, "
                    f"Number of observables: {n_observables}"
                )
        else:
            labels_lst = np.array(
                [f'x{i+1}' for i in range(self.data_mean.shape[1])],
                dtype=str
            )
        self.labels_lst = labels_lst

        # Save data in json file
        output_dic = {
            'data_mean': self.data_mean.tolist(),
            'data_sem': self.data_sem.tolist(),
            'times': self.options['times'].tolist(),
            'labels': labels_lst.tolist()
        }
        self.data_dir = io.save_json(output_dic, output_path)
