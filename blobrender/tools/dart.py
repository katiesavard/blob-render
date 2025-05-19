import numpy as np
import os, time, warnings
import numba as nb

class Mesh:
    """
    this class is a wrapper for an arbitrary number of MeshBlocks, each containing emissivity data
    Mesh objects are passed to Screen in order to generate images
    """

    def __init__(self, meshblocks=[]):
        self.meshblocks = meshblocks
        self.num_meshblocks = len(meshblocks)
        self.time = None
        self.baked = False
        self.baked_screen = None

    def add_mb(self, mb):
        self.meshblocks.append(mb)
        self.num_meshblocks += 1

    def import_hdata(self, load_str, origin=np.array([0,0,0]), bounds=None, npy_str=None, verbose=True, homogenize=True, level=3, target=[0,1], emm_mode="prs"):
        """
        load data from Athena++ hdf5 file, data is loaded as multiple MeshBlocks or single homogenized MeshBlock
        :param load_str: path to .hdf5 file
        :param origin: center point for cutout
        :param bounds: spatial extent for cutout (if None import all)
        :param npy_str: path to .npy data with black hole properties for centering if target specified
        :param verbose: boolean flag for verbose return
        :param homogenize: boolean flag, if True all MeshBlocks homogenized into single regular MeshBlock
        :param level: level of adaptive mesh refinement for homogenized block
        :param target: index of target black hole if npy_str passed
        :return: None
        """
        self.empty()
        if verbose: print("Importing data from {0} to Mesh".format(load_str))
        HData = HydroData(load_str)
        self.time = HData.Time
        if npy_str is not None and target is not None: # import origin from bh data
            if np.size(target) == 1:
                bh = BlackHole(target, npy_str)
                i = np.where(bh.t >= HData.Time)[0][0]
                origin = np.array([bh.x[i], bh.y[i], bh.z[i]])
            else:
                bh1 = BlackHole(0, npy_str)
                bh2 = BlackHole(1, npy_str)
                i = np.where(bh1.t >= HData.Time)[0][0]
                origin = np.array([bh1.m[i] * bh1.x[i] + bh2.m[i] * bh2.x[i],
                                   bh1.m[i] * bh1.y[i] + bh2.m[i] * bh2.y[i],
                                   bh1.m[i] * bh1.z[i] + bh2.m[i] * bh2.z[i]]) / (bh1.m[i] + bh2.m[i])

            if verbose: print("shifted origin to (x,y,z) = ({0},{1},{2})".format(*origin))

        if homogenize: # apply regularisation routines before import
            bounds_3d = np.array(bounds)
            for i in range(3):
                bounds_3d[i,:] += origin[i]
            if emm_mode is "rho":
                homo_data = HData.homogenize(level=level, homo_vars=["rho"], bounds=bounds_3d)
                emm = np.einsum("kji->ijk", homo_data["rho"])
            elif emm_mode is "prs":
                homo_data = HData.homogenize(level=level, homo_vars=["prs"], bounds=bounds_3d)
                emm = np.einsum("kji->ijk", homo_data["prs"])
            else:
                raise Exception("unable to parse emm_mode")
            bounds = [[np.min(homo_data["x"]), np.max(homo_data["x"])],
                      [np.min(homo_data["y"]), np.max(homo_data["y"])],
                      [np.min(homo_data["z"]), np.max(homo_data["z"])]]
            mb = MeshBlock(bounds, emm=emm)
            self.add_mb(mb)
            if verbose: print("Finished import, added single homogenous MeshBlock to Mesh.")
            del homo_data, emm
        else: # selectively insert meshblocks into mesh without regularisation
            for n in range(0, HData.NumMeshBlocks):
                mb_x = HData.x1f[n, :] - origin[0] # use face coords to define limits
                mb_y = HData.x2f[n, :] - origin[1]
                mb_z = HData.x3f[n, :] - origin[2]
                if not self.in_bounds([mb_x, mb_y, mb_z], bounds): continue
                bbox = [[mb_x[0], mb_x[-1]],
                        [mb_y[0], mb_y[-1]],
                        [mb_z[0], mb_z[-1]]]
                if emm_mode is "rho":
                    emm = HData.rho[n, ...] # TODO: make this user specified
                elif emm_mode is "prs":
                    emm = HData.P[n, ...]
                else:
                    raise Exception("unable to parse emm_mode")
                mb = MeshBlock(bbox, emm)
                self.add_mb(mb)
                if verbose: print("added MeshBlock {0} to Mesh...".format(n))
            del emm
            if verbose: print("Finished import, added {0}/{1} MeshBlocks to Mesh.".format(self.num_meshblocks, HData.NumMeshBlocks))
        del HData

    def import_pdata(self, load_str, fill_quadrants=True, inherit_bake=False, bbox=None):
        """
        load data from PLUTO .npy file, with option to duplicate data over all x-y quadrants
        data is assumed to be single homogenous array and is inserted as single MeshBlock
        :param load_str: path to .npy file containing emissivty data
        :param fill_quadrants: boolean flag to duplicate data over all x-y quadrants
        :param inherit_bake: copy pre-existing bake data from primary MeshBlock if present
        :param bbox: bounding box for emissivity data, in units common to Screen
        :return:
        """
        # load PLUTO emmisivity data from .npy file, replicate over 4 quadrants
        pdata = np.load(load_str)

        l = 0.5
        if fill_quadrants:
            quart_emm = np.einsum("kji->ijk", pdata)
            quart_dims = np.shape(quart_emm)
            hdim = quart_dims[0]
            # populate FULL array (x4 size)
            emm = np.zeros(shape=(2 * quart_dims[0], 2 * quart_dims[1], quart_dims[2]))
            emm[hdim:, hdim:, :] = quart_emm  # +x, +y quadrant
            emm[hdim:, :hdim, :] = quart_emm[:, ::-1, :]  # +x, -y quadrant
            emm[:hdim, hdim:, :] = quart_emm[::-1, :, :]  # -x, +y quadrant
            emm[:hdim, :hdim, :] = quart_emm[::-1, ::-1, :]  # -x, -y quadrant
            if bbox is None: bbox = [[-l, l], [-l, l], [-l, l]]
        else:
            emm = np.einsum("kji->ijk", pdata)
            if bbox is None: bbox = [[0, l], [0, l], [-l, l]]


        mb = MeshBlock(bbox, emm)
        if inherit_bake and self.num_meshblocks != 0:
            # copy bake from old MeshBlock
            mb.baked_rays = self.meshblocks[0].baked_rays
            mb.baked_dwells = self.meshblocks[0].baked_dwells
        self.empty()
        self.add_mb(mb)

    def in_bounds(self, test_data, bounds):
        """
        quick check to determine overlap between data and region
        require test_data and bounds to have same top-level length
        test_span can be single data point, or extended N-dim block
        bounds can be full span in each dim, or endpoints but must be ordered and array-like
        """

        if bounds is None: return True

        # compare binding across arbitrary number of dimensions
        for test_span, bound in zip(test_data, bounds):
            if np.min(test_span) > bound[-1]:
                return False
            elif np.max(test_span) < bound[0]:
                return False

        return True

    def empty(self):

        self.meshblocks = []
        self.nummeshblocks = 0

class HydroData:
    """
    this class loads single HDF5 snapshots into python memory retaining labels
    it also features regularisation routines to compile MeshBlocks into homogenous meshes
    """

    def __init__(self, h_str, user_vars="T"):
        import h5py
        print("Loading {0} as HydroData".format(h_str))
        self.coord_str = ["x", "y", "z"]
        # rename variables for user ease
        self.variable_dict = {
            # primitive
            "rho": "rho",
            "press": "P",
            "vel1": "vx",
            "vel2": "vy",
            "vel3": "vz",
            # conservative
            "dens": "rho",  # degenerate for non-relativistic simulations
            "Etot": "E",
            "mom1": "Mx",
            "mom2": "My",
            "mom3": "Mz",
        }
        # accept user specified labels for user_out_var
        if user_vars is not None:
            for i, user_var in enumerate(user_vars):
                self.variable_dict.update({"user_out_var" + str(i): user_var})
        # load data from HDF5 file
        with h5py.File(h_str, 'r') as f:
            # copy topline attributes
            for attr in list(f.attrs):
                setattr(self, attr, f.attrs.get(attr))
            # copy coordinates
            for attr in ("x1f", "x2f", "x3f", "x1v", "x2v", "x3v", "Levels", "LogicalLocations"):
                setattr(self, attr, np.array(f[attr]))
            # dupe coordinates for ease of access
            self.x = self.x1v
            self.y = self.x2v
            self.z = self.x3v
            # copy hydro data
            variable_names = np.array([x.decode("ascii", "replace") for x in f.attrs["VariableNames"][:]])
            dataset_sizes = f.attrs['NumVariables'][:]
            dataset_names = np.array([x.decode('ascii', 'replace') for x in f.attrs['DatasetNames'][:]])
            for dataset_index, dataset_name in enumerate(dataset_names):
                variable_begin = sum(dataset_sizes[:dataset_index])
                variable_end = variable_begin + dataset_sizes[dataset_index]
                variable_names_local = variable_names[variable_begin:variable_end]
                for variable_index, variable_name in enumerate(variable_names_local):
                    if variable_name in self.variable_dict: # if new label exists, relabel
                        attr_name = self.variable_dict[variable_name]
                    else:
                        attr_name = variable_name
                    setattr(self, attr_name, np.array(f[dataset_name][variable_index, ...]))

    def homogenize(self, level=None, homo_vars=None, verbose=False, bounds=None):
        mb_size = self.MeshBlockSize
        max_level = np.max(self.Levels)
        # test restriction limits
        if level is None:
            level = max_level
        if level > max_level:
            warnings.warn("target level {0} exceeds maximum mesh level {1}".format(level, max_level))
        else:
            max_restrict = 2 ** (max_level - level)
            for d in range(0, 3):
                if mb_size[d] != 1 and mb_size[d] < max_restrict:
                    limit = max_level - int(np.log2(mb_size[d]))
                    warnings.warn(
                        "target level " + str(level) + " too low for restriction routine, must be >= " + str(limit))

        # define dimensions of output array
        nx_vals = []
        for d in range(3): # handle slicing here?
            if mb_size[d] == 1: # do not expand along unexpanded dimension
                nx_vals.append(self.RootGridSize[d])
            else:
                nx_vals.append(self.RootGridSize[d] * 2 ** level)
        nx1 = nx_vals[0]
        nx2 = nx_vals[1]
        nx3 = nx_vals[2]

        if verbose:
            print("starting homogenization routine...")
            print("root grid                            [nk, nj, ni] = [{0}, {1}, {2}]".format(*self.RootGridSize[-1::-1]))
            print("max mesh level                                    = {0}".format(max_level))
            print("homogenous level                                  = {0}".format(level))
            print("master grid                          [nk, nj, ni] = [{0}, {1}, {2}]".format(nx3, nx2, nx1))

        # populate coordinate arrays
        data = {}
        for d, (nx, c) in enumerate(zip(nx_vals, self.coord_str)):
            xmin = getattr(self, "RootGridX" + str(d+1))[0]
            xmax = getattr(self, "RootGridX" + str(d+1))[1]
            data[c] = np.linspace(xmin, xmax, nx+1)

        # account for domain selection
        index_lim = np.zeros(shape=(3,2), dtype=np.int32)
        index_lim[0, 1] = nx1
        index_lim[1, 1] = nx2
        index_lim[2, 1] = nx3
        trims = np.array([False, False, False])
        slices = np.array([False, False, False])
        err_string = "{0} must be {1} than {2} in order to overlap domain"
        if np.any(bounds is not None): # trim domain to spec
            # test user input
            if np.shape(bounds) != (3, 2):
                raise Exception("Invalid pass to bounds, require input shape (3,2)")
            # test if bounds in domain
            for d, c in enumerate(self.coord_str):
                bound = bounds[d, :]
                if bound[0] is not None and bound[0] >= data[c][0]:
                    if bound[0] >= data[c][-1]:
                        raise Exception(err_string.format(c + "_min", "less", data[c][-1]))
                    index_lim[d, 0] = np.where(data[c] <= bound[0])[0][-1]
                    trims[d] = True
                if bound[1] is not None and bound[1] <= data[c][-1]:
                    if bound[1] <= data[c][0]:
                        raise Exception(err_string.format(c, "_max", "greater", data[c][0]))
                    index_lim[d, 1] = np.where(data[c] >= bound[1])[0][0]
                    trims[d] = True
                if nx_vals[d] != 1: # if extended dimension, check for slice
                    if bound[0] == bound[1] and (bound[0] is not None) and (bound[1] is not None): # select slice
                        slices[d] = True
                        index_lim[d, 1] += 1 # bump to allow for single value

        # trim data arrays
        for d, c in enumerate(self.coord_str):
            if trims[d]:
                data[c] = data[c][index_lim[d, 0]:index_lim[d, 1] + 1]

        # unpack indices
        i_min = index_lim[0, 0]
        i_max = index_lim[0, 1]
        j_min = index_lim[1, 0]
        j_max = index_lim[1, 1]
        k_min = index_lim[2, 0]
        k_max = index_lim[2, 1]

        # identify output variables to merge
        if homo_vars is None:  # merge all variables
            homo_vars = []
            for q, (x, variable_name) in enumerate(self.variable_dict.items()):
                if hasattr(self, variable_name):
                    homo_vars.append(variable_name)
        elif isinstance(homo_vars, str):  # accept single variable specified
            homo_vars = [homo_vars]
        elif not (isinstance(homo_vars, list) or isinstance(homo_vars, np.ndarray)): # accept list of str
            raise Exception("invalid pass to homo_vars")

        # purge improper variables
        for merge_var in homo_vars:
            if not hasattr(self, merge_var):
                homo_vars.remove(merge_var)
                print("removing {0}" + str(merge_var) + " from list")

        # build output array
        for merge_var in homo_vars:
            data.update({merge_var: np.zeros((k_max - k_min, j_max - j_min, i_max - i_min))})

        if verbose:
            print("apply bounds [xmin, xmax, ymin, ymax, zmin, zmax] = [{0}, {1}, {2}, {3}, {4}, {5}]".format(*np.ravel(bounds)))
            print("homogenous grid                      [nk, nj, ni] = [{0}, {1}, {2}]".format(k_max - k_min, j_max - j_min, i_max - i_min))
            print("homogenizing hydro variables                       ", homo_vars)

        # iterate over mb
        for mb_num in range(self.NumMeshBlocks):
            mb_level = self.Levels[mb_num]
            mb_location = self.LogicalLocations[mb_num, :]

            # apply prolongation to coarse, copy same-level
            if mb_level <= level:
                # scale multiplier
                s = 2 ** (level - mb_level)
                # destination indices in merged
                il_d = (mb_location[0] * mb_size[0] * s
                        if nx1 > 1 else 0)
                jl_d = (mb_location[1] * mb_size[1] * s
                        if nx2 > 1 else 0)
                kl_d = (mb_location[2] * mb_size[2]* s
                        if nx3 > 1 else 0)
                iu_d = il_d + mb_size[0] * s if nx1 > 1 else 1
                ju_d = jl_d + mb_size[1] * s if nx2 > 1 else 1
                ku_d = kl_d + mb_size[2] * s if nx3 > 1 else 1

                # Calculate (prolongated) source indices, with selection
                il_s = max(il_d, i_min) - il_d
                jl_s = max(jl_d, j_min) - jl_d
                kl_s = max(kl_d, k_min) - kl_d
                iu_s = min(iu_d, i_max) - il_d
                ju_s = min(ju_d, j_max) - jl_d
                ku_s = min(ku_d, k_max) - kl_d
                if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
                    continue

                # Account for selection in destination indices
                il_d = max(il_d, i_min) - i_min
                jl_d = max(jl_d, j_min) - j_min
                kl_d = max(kl_d, k_min) - k_min
                iu_d = min(iu_d, i_max) - i_min
                ju_d = min(ju_d, j_max) - j_min
                ku_d = min(ku_d, k_max) - k_min

                # insert values
                for merge_var in homo_vars:
                    mb_data = getattr(self, merge_var)[mb_num, ...]
                    if s > 1: # level != mb_level, prolongate data and insert
                        if nx1 > 1:
                            mb_data = np.repeat(mb_data, s, axis=2)[:, :, il_s:iu_s]
                        if nx2 > 1:
                            mb_data = np.repeat(mb_data, s, axis=1)[:, jl_s:ju_s, :]
                        if nx3 > 1:
                            mb_data = np.repeat(mb_data, s, axis=0)[kl_s:ku_s, :, :]
                        data[merge_var][kl_d:ku_d, jl_d:ju_d, il_d:iu_d] = mb_data
                    else: # level match, insert directly
                        data[merge_var][kl_d:ku_d, jl_d:ju_d, il_d:iu_d] = mb_data[kl_s:ku_s,
                                                                   jl_s:ju_s,
                                                                   il_s:iu_s]
            else: # restrict fine data
                # Calculate scale
                s = 2 ** (mb_level - level)

                # Calculate destination indices, without selection
                il_d = mb_location[0] * mb_size[0] // s if nx1 > 1 else 0
                jl_d = mb_location[1] * mb_size[1] // s if nx2 > 1 else 0
                kl_d = mb_location[2] * mb_size[2] // s if nx3 > 1 else 0
                iu_d = il_d + mb_size[0] // s if nx1 > 1 else 1
                ju_d = jl_d + mb_size[1] // s if nx2 > 1 else 1
                ku_d = kl_d + mb_size[2] // s if nx3 > 1 else 1

                # Calculate (restricted) source indices, with selection
                il_s = max(il_d, i_min) - il_d
                jl_s = max(jl_d, j_min) - jl_d
                kl_s = max(kl_d, k_min) - kl_d
                iu_s = min(iu_d, i_max) - il_d
                ju_s = min(ju_d, j_max) - jl_d
                ku_s = min(ku_d, k_max) - kl_d
                if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
                    continue

                # Account for selection in destination indices
                il_d = max(il_d, i_min) - i_min
                jl_d = max(jl_d, j_min) - j_min
                kl_d = max(kl_d, k_min) - k_min
                iu_d = min(iu_d, i_max) - i_min
                ju_d = min(ju_d, j_max) - j_min
                ku_d = min(ku_d, k_max) - k_min

                # Account for restriction in source indices
                if nx1 > 1:
                    il_s *= s
                    iu_s *= s
                if nx2 > 1:
                    jl_s *= s
                    ju_s *= s
                if nx3 > 1:
                    kl_s *= s
                    ku_s *= s

                # Apply subsampling
                # Calculate fine-level offsets (nearest cell at or below center)
                o1 = s // 2 - 1 if nx1 > 1 else 0
                o2 = s // 2 - 1 if nx2 > 1 else 0
                o3 = s // 2 - 1 if nx3 > 1 else 0

                # Assign values
                for merge_var in homo_vars:
                    data[merge_var][kl_d:ku_d,
                    jl_d:ju_d,
                    il_d:iu_d] = getattr(self, merge_var)[mb_num, kl_s + o3:ku_s:s,
                                 jl_s + o2:ju_s:s, il_s + o1:iu_s:s]

        if verbose:
            print("finished homogenizing.")

        return data

class BlackHole:
    """
    this class loads BH data for a single BH into memory from a txt or npy file, labelling and retaining header info
    """
    def __init__(self, bh_index, inp, n_data=15):
        # pass user input, accept inp as path to bh_data.txt or preloaded npy object
        if isinstance(inp, str):
            bh_data = np.load(inp)
        elif isinstance(inp, np.ndarray):
            bh_data = inp
        else:
            raise Exception("Invalid type for inp, accept path to bh_data.txt or npy array")

        # load header data
        self.head_list = ["b", "rho0", "Omega0", "xl", "xr", "yl", "yr", "zl", "zr"]
        for i, head in enumerate(self.head_list):
            setattr(self, head, bh_data[0, i])
        bh_data = bh_data[1:, :]

        # load bh data
        self.var_list = ["m", "x", "y", "z", "vx", "vy", "vz", "ax_gas", "ay_gas", "az_gas", "ax_acc", "ay_acc", "az_acc", "J", "m_obs"]
        self.t = bh_data[:, 0]
        js = 1 + bh_index * n_data
        for i, var in enumerate(self.var_list):
            setattr(self, var, bh_data[:, js + i])

        del bh_data

    def clip(self, start_index=None, stop_index=None):

        if start_index is not None:
            if stop_index is None:
                slicer = np.s_[start_index:]
            else:
                slicer = np.s_[start_index:stop_index]
        elif stop_index is not None:
            slicer = np.s_[:stop_index]
        else:
            return

        self.t = self.t[slicer]
        for attr in self.var_list:
            setattr(self, attr, getattr(self, attr)[slicer])

class MeshBlock:

    def __init__(self, bbox, emm, vels=None, c_light=None):
        self.bbox = bbox # [[xl,xr],[yl,yr],[zl,zr]]
        self.dims = np.shape(emm)
        self.emm = emm
        self.vels = vels # [vx, vy, vz]
        self.c_light = c_light # speed of light in same units as vels
        self.dx = [(self.bbox[i][1] - self.bbox[i][0]) / self.dims[i] for i in range(3)] # cell sizes
        self.axes_bitmap = np.array([2,1,2,1,2,2,0,0], dtype=np.int32) # traversal bitmap
        self.doppler_fac = None # populate this array only if autoboost called
        self.baked_rays = None
        self.baked_dwells = None

    def calc_intercept(self, r):
        """
        determine intersection locations between ray and bbox of MeshBlock
        :param r: ray
        :return: boolean hit/miss, distance to entrance, distance to exit
        """
        for i in range(3):
            tcmin = (self.bbox[i][r.sign[i]] - r.O[i]) * r.invN[i]
            tcmax = (self.bbox[i][1 - r.sign[i]] - r.O[i]) * r.invN[i]
            if i == 0:
                tmin = tcmin
                tmax = tcmax
                continue

            # discard collisions with misordered intercepts
            if (tmin > tcmax) or (tcmin > tmax):
                return False, None, None

            # reallocate extrema to internal values
            if tcmin > tmin:
                tmin = tcmin

            if tcmax < tmax:
                tmax = tcmax

        return True, tmin, tmax

    def calc_path(self, r, t_entry, t_exit):
        """
        determine all cell indices in MeshBlock that lies on ray trajectory
        :param r: ray
        :param t_entry: ray distance at MeshBlock entrance
        :param t_exit: ray distance at MeshBlock exit
        :return:
        """
        # define internals
        cell = np.zeros(3, dtype=np.int32)
        dt = np.zeros(3, dtype=np.float64)
        next_t_cross = np.zeros(3, dtype=np.float64)
        exit_cond = np.zeros(3, dtype=np.int32)
        step_dir = np.zeros(3, dtype=np.int32)
        mb_entrance = r.march(t_entry)
        for i in range(3):
            ray_mb_origin = mb_entrance[i] - self.bbox[i][0]
            cell[i] = self.int_clamp(ray_mb_origin / self.dx[i], 0, self.dims[i]-1)
            if r.sign[i]:
                dt[i] = -self.dx[i] * r.invN[i]
                next_t_cross[i] = t_entry + (cell[i] * self.dx[i] - ray_mb_origin) * r.invN[i]
                exit_cond[i] = -1 # exit condition on left side
                step_dir[i] = -1 # reverse traversal
            else:
                dt[i] = self.dx[i] * r.invN[i]
                next_t_cross[i] = t_entry + ((cell[i] + 1) * self.dx[i] - ray_mb_origin) * r.invN[i]
                exit_cond[i] = self.dims[i]
                step_dir[i] = 1

        # traverse meshblock
        mb_exit = r.march(t_exit)
        mb_span = mb_exit - mb_entrance
        crossings = np.ceil(np.abs(mb_span / self.dx))
        max_depth = 1 + np.sum(crossings)
        path, dwells = self.walk_path(t_entry, cell, dt, next_t_cross, exit_cond, step_dir, max_depth, self.axes_bitmap)
        true_depth = np.nonzero(path[:,0] == -1)[0]
        if np.size(true_depth) == 0:
            return path, dwells
        else:
            return path[:true_depth[0]], dwells[:true_depth[0]]

    def calc_cc(self):
        """
        calculate cell-centered grid positions from bbox
        :return: (x,y,z) triple
        """

        x_cc_and_fc = np.linspace(self.bbox[0][0], self.bbox[0][1], 2 * self.dims[0] + 1)
        self.x_cc = x_cc_and_fc[1:-1:2]
        y_cc_and_fc = np.linspace(self.bbox[1][0], self.bbox[1][1], 2 * self.dims[1] + 1)
        self.y_cc = y_cc_and_fc[1:-1:2]
        z_cc_and_fc = np.linspace(self.bbox[2][0], self.bbox[2][1], 2 * self.dims[2] + 1)
        self.z_cc = z_cc_and_fc[1:-1:2]

        return self.x_cc, self.y_cc, self.z_cc

    def int_clamp(self, input, minval, maxval):
        """
        restrict input value to supplied bounds and integer value
        :param input: value to clamp
        :param minval: lower thresh
        :param maxval: upper thresh
        :return:
        """
        return max(minval, min(np.floor(input), maxval))

    @staticmethod
    @nb.jit(nb.types.Tuple((nb.int32[:, :], nb.float64[:]))(nb.float64, nb.int32[:], nb.float64[:], nb.float64[:],
                                                         nb.int32[:], nb.int32[:], nb.int64, nb.int32[:]), nopython=True)
    def walk_path(t_entry, cell, dt, next_t_cross, exit_cond, step_dir, max_depth, axes_bitmap):
        """
        for a given ray entry, traverse the MeshBlock and store all indices, dwells on path
        :param t_entry: ray distance to first bbox intersection
        :param cell: cell index at ray entry (i,j,k)
        :param dt: ray distance between successive cell crossings in (x,y,z)
        :param next_t_cross: ray distance to next cell crossing in (x,y,z)
        :param exit_cond: index for walk termination (i_exit, j_exit, k_exit)
        :param step_dir: -1/+1 for ray anti-alignment with coordinate axes
        :param max_depth: maximum propagation depth before walk termination
        :param axes_bitmap: copy of map to convert bitshift to axes direction
        :return:
        """

        current_t = t_entry
        depth = 0
        path = np.full(shape=(max_depth, 3), fill_value=-1, dtype=np.int32)
        dwells = np.full(shape=(max_depth), fill_value=-1, dtype=np.float64)
        while depth < max_depth:
            # add current position to path
            path[depth, :] = cell
            # find next cell on path
            k = (((next_t_cross[0] < next_t_cross[1]) << 2) +
                 ((next_t_cross[0] < next_t_cross[2]) << 1) +
                 ((next_t_cross[1] < next_t_cross[2])))
            axis = axes_bitmap[k]

            # update position of ray head
            dwell = next_t_cross[axis] - current_t
            dwells[depth] = dwell
            current_t = next_t_cross[axis]

            # traverse to next cell
            cell[axis] += step_dir[axis]
            # terminate iteration if domain escaped on next cell
            if cell[axis] == exit_cond[axis]:
                break
            next_t_cross[axis] += dt[axis]
            depth += 1

        return path, dwells

class Ray:

    def __init__(self, origin, normal):
        self.O = origin
        self.N = normal / np.linalg.norm(normal)
        self.invN = 1 / self.N
        self.sign = np.array((self.invN < 0), dtype=np.int32) # int cast to avoid depreciation warning

    def march(self, t):
        """
        calculate vector position of ray head post-propagation
        :param t: ray length
        :return: np.array([x,y,z]) position
        """
        return self.O + t * self.N

class Screen:

    def __init__(self, R, theta, phi, sdim, pdim, bias=np.array([0,0,1]), tilt=None):
        self.O = R * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        self.sdim = sdim
        self.pdim = pdim
        self.Xhat = np.cross(bias, self.O)
        self.Xhat /= np.linalg.norm(self.Xhat)
        self.Yhat = np.cross(self.O, self.Xhat)
        self.Yhat /= np.linalg.norm(self.Yhat)
        self.view_dir = - self.O / np.linalg.norm(self.O) # enforced plane parallel cast
        if tilt is not None:
            self.Xhat = self.rotate_about(self.Xhat, self.view_dir, tilt)
            self.Yhat = self.rotate_about(self.Yhat, self.view_dir, tilt)
        self.UL = self.O - 0.5 * sdim[0] * self.Xhat + 0.5 * sdim[1] * self.Yhat
        self.img = np.zeros(shape=(self.pdim[1], self.pdim[0]))
        # timers
        self.render_time = 0
        self.alloc_time = 0
        self.bake_time = 0
        self.sum_time = 0

    def __eq__(self, other):
        for var in self.__dict__:
            if var in ["img", "render_time", "alloc_time", "bake_time", "sum_time"]: continue
            if np.any(getattr(self, var) != getattr(other, var)): return False
        return True

    def rotate_about(self, v, k, theta):
        """
        brief implementation of Rodrigues' rotation formula to allow for screen tilt
        :param v: vector to be rotated
        :param k: vector about which to rotate
        :param theta: rotation angle (right-handed)
        :return: rotated vector
        """
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return v * cos_theta + np.cross(k, v) * sin_theta + k * np.dot(k, v) * (1 - cos_theta)

    def calc_pixel_value(self, i, j, mesh, auto_boost=False):
        """
        calculate total emission in ray line-of-sight cast from pixel (i,j)
        :param i: pixel index in neg Y dir on screen
        :param j: pixel index in pos X dir on screen
        :param mesh: Mesh containing scene data
        :param auto_boost: boolean, if True apply Dopper boost
        :return: emission summation over ray path
        """
        offsetY = - (i / self.pdim[1]) * self.sdim[1]
        offsetX = (j / self.pdim[0]) * self.sdim[0]
        pixel_pos = self.UL + self.Xhat * offsetX + self.Yhat * offsetY
        ray = Ray(pixel_pos, self.view_dir)  # TODO: allow for arb view angle
        # test intersection of ray with all mb in scene
        pixel_value = 0
        for mb in mesh.meshblocks:
            hit, tmin, tmax = mb.calc_intercept(ray)
            if not hit: continue
            path, dwells = mb.calc_path(ray, tmin, tmax)
            ndim = np.size(np.shape(path))
            if ndim == 1:  # single cell intersection
                on_path = np.s_[*path]
            else:
                on_path = np.s_[path[:, 0], path[:, 1], path[:, 2]]
            if not auto_boost: # boosting negligible or already applied to emm data
                weighted_emm = mb.emm[on_path] * dwells
            else: # apply doppler boosting
                weighted_emm = mb.emm[on_path] * dwells * mb.doppler_fac[on_path]
            pixel_value += np.sum(weighted_emm)
        return pixel_value

    def render(self, mesh, verbose=False, auto_boost=False, use_bake=False, bake_float_type=np.float32, bake_int_type=np.int16, report_count=10, rebake=False):
        """
        populate an image using rays cast into a scene
        :param mesh: Mesh containing scene data
        :param verbose: boolean, if True report render progress
        :param auto_boost: boolean, if True apply Doppler boost
        :param use_bake: boolean, if True pre-calculate ray paths, or use existing bake map
        :param bake_float_type: fixed float precision for baking
        :param bake_int_type: fixed int precision for baking
        :param rebake: boolean, if True bake but ignore pre-existing bake map
        :return: img numpy array
        """
        # check validity of auto boost
        if auto_boost:
            # determine if required data defined for Doppler boosting
            for mb in mesh.meshblocks:
                if mb.vels is None:
                    raise Exception("autoboost requires population of velocity data in MeshBlocks")
                if mb.c_light is None:
                    raise Exception("autoboost requires speed of light definition in MeshBlocks")
                # calculate doppler boosting for all cells in mb
                beta = mb.vels / mb.c_light
                inv_gamma = np.sqrt(1 - beta ** 2)
                vmag = np.sqrt(mb.vels[0] ** 2 + mb.vels[1] ** 2 + mb.vels[2] ** 2) # view_dir already normalised
                cos_theta = (self.view_dir[0] * mb.vels[0] + self.view_dir[1] * mb.vels[1] + self.view_dir[2] * mb.vels[2]) / vmag
                mb.doppler_fac = inv_gamma / (1 - beta * cos_theta)

        ind_i = range(self.pdim[1])
        ind_j = range(self.pdim[0])
        ii, jj = np.meshgrid(ind_i, ind_j)
        indices = list(zip(ii.flatten(), jj.flatten()))
        num_pixels = len(indices)
        coarse = int(num_pixels / report_count)
        if coarse == 0: coarse = 1
        render_start = time.perf_counter_ns()

        # run preallocation if necessary
        if use_bake:
            if rebake:
                if verbose: print("rebake called, calling bake routine...")
                self.alloc_bake(mesh, verbose, bake_float_type, bake_int_type)
                self.bake(mesh, indices, num_pixels, coarse, verbose)
            elif not mesh.baked:
                if verbose: print("bake map unallocated, calling bake routine...")
                self.alloc_bake(mesh, verbose, bake_float_type, bake_int_type)
                self.bake(mesh, indices, num_pixels, coarse, verbose)
            elif not mesh.baked_screen == self:
                if verbose: print("different screen used for previous bake, calling bake routine")
                self.alloc_bake(mesh, verbose, bake_float_type, bake_int_type)
                self.bake(mesh, indices, num_pixels, coarse, verbose)

        if verbose: print("starting render...")
        sum_start = time.perf_counter_ns()
        if use_bake:
            for pixel_num, index_pair in enumerate(indices):
                pixel_value = 0
                for mb in mesh.meshblocks:
                    path = mb.baked_rays[*index_pair, :, :]
                    dwells = mb.baked_dwells[*index_pair, :]
                    live = (dwells != -1) # remove dead indices
                    path = path[live]
                    dwells = dwells[live]
                    on_path = np.s_[path[:,0], path[:,1], path[:, 2]]
                    if not auto_boost:
                        weighted_emm = mb.emm[on_path] * dwells
                    else:
                        weighted_emm = mb.emm[on_path] * dwells * mb.doppler_fac[on_path]
                    pixel_value += np.sum(weighted_emm)
                self.img[*index_pair] = pixel_value
                if verbose and pixel_num % coarse == 0:
                    print("summed path for pixel {0}/{1}, sum {2}% complete".format(pixel_num, num_pixels,
                                                                       round(pixel_num / num_pixels * 100, 1)))
            if verbose: print("summation finished, 100% complete")
        else:
            for pixel_num, index_pair in enumerate(indices):
                pixel_value = self.calc_pixel_value(*index_pair, mesh)
                self.img[*index_pair] = pixel_value
                if verbose and pixel_num % coarse == 0:
                    print("pixel {0}/{1} drawn, render {2}% complete".format(pixel_num, num_pixels, round(pixel_num/num_pixels * 100,1)))
            if verbose: print("render finished, 100% complete")

        if verbose:
            sum_end = time.perf_counter_ns()
            self.sum_time = (sum_end - sum_start) * 1e-9
            render_end = time.perf_counter_ns()
            self.render_time = (render_end - render_start) * 1e-9
            if use_bake:
                print("{0}x{1} image rendered in {2}s, ({3}s alloc, {4}s bake, {5}s sum)".format(*self.pdim, round(self.render_time,3), round(self.alloc_time,3), round(self.bake_time,3), round(self.sum_time,3)))
            else:
                print("{0}x{1} image rendered in {2}s".format(*self.pdim, round(self.render_time, 3)))

        return self.img

    def alloc_bake(self, mesh, verbose=False, dtype_float=np.float32, dtype_int=np.int32):

        # initialise empty bake maps for all MeshBlocks
        num_bytes = 0
        if verbose:
            print("allocating bake storage...")
            alloc_start = time.perf_counter_ns()

        for mb in mesh.meshblocks:
            # determine max depth, allocate
            max_depth = 1 + np.sum(mb.dims)
            mb.baked_rays = np.full(shape=(self.pdim[1], self.pdim[0], max_depth, 3), fill_value=-1, dtype=dtype_int)
            mb.baked_dwells = np.full(shape=(self.pdim[1], self.pdim[0], max_depth), fill_value=-1,
                                      dtype=dtype_float)
            if verbose: num_bytes += mb.baked_rays.nbytes + mb.baked_dwells.nbytes

        if verbose:
            alloc_end = time.perf_counter_ns()
            self.alloc_time = (alloc_end - alloc_start) * 1e-9
            print("{0}GB bake storage allocated in {1}s.".format(round(self.alloc_time,3), round(num_bytes / 1e9, 2)))

    def bake(self, mesh, indices, num_pixels, coarse, verbose=False):

        # cast rays into MeshBlocks, assign bake maps
        if verbose:
            print("starting bake...")
            bake_start = time.perf_counter_ns()

        for pixel_num, index_pair in enumerate(indices):
            # initialise ray
            offsetY = - (index_pair[0] / self.pdim[1]) * self.sdim[1]
            offsetX = (index_pair[1] / self.pdim[0]) * self.sdim[0]
            pixel_pos = self.UL + self.Xhat * offsetX + self.Yhat * offsetY
            ray = Ray(pixel_pos, self.view_dir)  # TODO: allow for arb view angle
            # cast ray through all MeshBlocks
            for mb in mesh.meshblocks:
                # find path, dwells
                hit, tmin, tmax = mb.calc_intercept(ray)
                if not hit: continue
                path, dwells = mb.calc_path(ray, tmin, tmax)
                # stash path, dwells
                depth = np.size(dwells)
                mb.baked_rays[*index_pair, :depth, :] = path
                mb.baked_dwells[*index_pair, :depth] = dwells
            if verbose and pixel_num % coarse == 0:
                print("baked path for pixel {0}/{1}, bake {2}% complete".format(pixel_num, num_pixels, round(pixel_num / num_pixels * 100, 1)))

        mesh.baked = True
        mesh.baked_screen = self
        if verbose:
            bake_end = time.perf_counter_ns()
            self.bake_time = (bake_end - bake_start) * 1e-9
            print("{0}x{1} image baked in {2}s".format(*self.pdim, round(self.bake_time,3)))

    def empty(self):
        self.img = np.zeros(shape=(self.pdim[1], self.pdim[0]))