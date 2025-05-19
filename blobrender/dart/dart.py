"""
This file contains the basis functions for the DART (DDA Accelerated Ray Tracing) module, allowing for line-of-sight summation on 3D data.
Data may be passed as a single homgenous (fixed resolution) mesh, but support is presented for adaptive mesh refinement.
"""

import numpy as np
import os, time, warnings
import numba as nb

class Mesh:
    """
    Create a container for an arbitrary number of MeshBlocks, each containing emissivity data
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

class MeshBlock:
    """
    Create a container for a single homogenous grid of hydrodynamic data. 
    """
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
    """
    Create a ray as defined by a vector origin and normal
    """
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
    """
    Create a Screen, defined by an orientation and position in physical space.
    Screen objects allow for image rendering, when passed a Mesh to act as a scene.
    """
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