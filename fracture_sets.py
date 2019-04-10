#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 12:54:36 2018

@author: Eirik Keilegavlens
"""
import numpy as np
import scipy
import scipy.stats as stats
import logging
import pdb

from fracture_generation import fracture_network_analysis, distributions
import porepy as pp


logger = logging.getLogger(__name__)


class StochasticFractureNetwork2d(pp.FractureNetwork2d):
    def __init__(self, pts=None, edges=None, domain=None, network=None, tol=1e-8):
        if network is not None:
            pts = network.pts
            edges = network.edges
            domain = network.domain
            tol = network.tol
        super(StochasticFractureNetwork2d, self).__init__(
            pts=pts, edges=edges, domain=domain, tol=tol
        )

        if self.edges.shape[0] == 2:
            self.edges = np.vstack(
                (self.edges, np.zeros((1, self.num_frac), dtype=np.int))
            )

        self.branches = self.num_frac * [[]]

        for fi in range(self.num_frac):
            self.branches[fi] = self.pts[:, self.edges[:2, fi]]

    def add_fracture(self, p0, p1, tag=None):

        num_pts = self.pts.shape[1]

        pt_arr = np.hstack((p0.reshape((-1, 1)), p1.reshape((-1, 1))))
        self.pts = np.hstack((self.pts, pt_arr))
        e = np.array([[num_pts], [num_pts + 1], [tag]], dtype=np.int)
        self.edges = np.hstack((self.edges, e))

        self.branches.append(pt_arr)
        self.num_frac += 1

    def _sort_branch_points(self, fi, p):
        start = self.start_points(fi)

        dist = np.sum((p - start) ** 2, axis=0)
        return p[:, np.argsort(dist)]

    def add_branch(self, fi, p):
        loc_branches = np.hstack((self.branches[fi], p.reshape((-1, 1))))
        loc_branches = self._sort_branch_points(fi, loc_branches)

    def branch_length(self, fi=None):
        if fi is None:
            fi = np.arange(self.num_frac)
        lengths = np.array(fi.size, dtype=np.object)
        for f in fi:
            lengths[f] = np.sqrt(
                np.sum(
                    (self.branches[f][:, 1:] - self.branches[f][:, :-1]) ** 2, axis=0
                )
            )

        return np.atleast_2d(lengths)

    def constrain_to_domain(self, domain=None):
        network = super(StochasticFractureNetwork2d, self).constrain_to_domain(domain)
        return StochasticFractureNetwork2d(network=network)

    def __str__(self):
        s = "Stochastic fracture set consisting of " + str(self.num_frac) + " fractures"
        if self.pts is not None:
            s += ", consisting of " + str(self.pts.shape[1]) + " points.\n"
        else:
            s += ".\n"
        if self.domain is not None:
            s += "Domain: "
            s += str(self.domain)
        return s

    def __repr__(self):
        return self.__str__()


class StochasticFractureGenerator(object):
    """ Factory class for stochastic fracture networks.

    The fractures generated are described by their orientation, length and location
    of their center points.

    """

    def __init__(
        self, dist_length=None, dist_orientation=None, dist_spacing=None, domain=None
    ):
        self.dist_length = dist_length
        self.dist_orientation = dist_orientation
        self.intensity = dist_spacing
        self.domain = domain

    def fit_distributions(self, **kwargs):
        """ Fit statistical distributions to describe the fracture set.

        The method will compute best fit distributions for fracture length,
        angle and position. These can later be used to generate realizations
        of other fracture network, using the current one as a base case.

        The length distribution can be either lognormal or exponential.

        The orientation is represented by a best fit of a von-Mises distribution.

        The fracture positions are represented by an intensity map, which
        divides the domain into subblocks and count the number of fracture
        centers per block.

        For more details, see the individual functions for fitting each of the
        distributions

        """
        logger.info("Fit length, angle and intensity distribution")
        self.fit_length_distribution(**kwargs)
        self.fit_angle_distribution(**kwargs)
        self.fit_intensity_map(**kwargs)

    def fit_length_distribution(self, ks_size=100, p_val_min=0.05, **kwargs):
        """ Fit a statistical distribution to describe the length of the fractures.

        The best fit is sought between an exponential and lognormal representation.

        The resulting distribution is represented in an attribute dist_length.

        The function also evaluates the fitness of the chosen distribution by a
        Kolgomorov-Smirnov test.

        Parameters:
            ks_size (int, optional): The number of realizations used in the
                Kolmogorov-Smirnov test. Defaults to 100.
            p_val_min (double, optional): P-value used in Kolmogorev-Smirnov test
                for acceptance of the chosen distribution. Defaults to 0.05.

        """
        # fit the lenght distribution
        candidate_dist = np.array([stats.expon, stats.lognorm])

        # fit the possible lenght distributions
        l = self.length()
        dist_fit = np.array([d.fit(l, floc=0) for d in candidate_dist])

        # determine which is the best distribution with a Kolmogorov-Smirnov test
        ks = lambda d, p: stats.ks_2samp(l, d.rvs(*p, size=ks_size))[1]
        p_val = np.array([ks(d, p) for d, p in zip(candidate_dist, dist_fit)])
        best_fit = np.argmax(p_val)

        if p_val[best_fit] < p_val_min:
            raise ValueError("p-value not satisfactory for length fit")

        # collect the data
        dist_l = {
            "dist": candidate_dist[best_fit],
            "param": dist_fit[best_fit],
            "p_val": p_val[best_fit],
        }
        # Logging
        stat_string = ["exponential", "log-normal"]
        logger.info(
            "Fracture length represented by a %s distribution ", stat_string[best_fit]
        )
        s = "Fracture parameters: "
        for p in dist_fit[best_fit]:
            s += str(p) + ", "
        logger.info(s)
        logger.info("P-value for fitting: %.3f", p_val[best_fit])

        self.dist_length = dist_l

    def fit_angle_distribution(self, ks_size=100, p_val_min=0.05, **kwargs):

        """ Fit a statistical distribution to describe the length of the fractures.

        The best fit is sought between an exponential and lognormal representation.

        The resulting distribution is represented in an attribute dist_angle.

        The function also evaluates the fitness of the chosen distribution by a
        Kolgomorov-Smirnov test.

        Parameters:
            ks_size (int, optional): The number of realizations used in the
                Kolmogorov-Smirnov test. Defaults to 100.
            p_val_min (double, optional): P-value used in Kolmogorev-Smirnov test
                for acceptance of the chosen distribution. Defaults to 0.05.

        """
        dist = stats.vonmises
        a = self.angle()
        dist_fit = dist.fit(a, fscale=1)

        # check the goodness of the fit with Kolmogorov-Smirnov test
        p_val = stats.ks_2samp(a, dist.rvs(*dist_fit, size=ks_size))[1]

        if p_val < p_val_min:
            raise ValueError("p-value not satisfactory for angle fit")

        # logging
        logger.info("Fracture orientation represented by a von mises distribution ")
        s = "Fracture parameters: "
        for p in dist_fit:
            s += str(p) + ", "
        logger.info(s)
        logger.info("P-value for fitting: %.3f", p_val)

        # collect the data
        self.dist_angle = {"dist": dist, "param": dist_fit, "p_val": p_val}

    def fit_intensity_map(self, p=None, e=None, domain=None, nx=10, ny=10, **kwargs):
        """ Divide the domain into boxes, count the number of fracture centers
        contained within each box, and divide by the measure of the domain.

        The resulting intensity map is stored in an attribute intensity.

        Parameters:
            p (np.array, 2 x n, optional): Point coordinates of the fractures. Defaults to
                this set.
            e (np.array, 2 x n, optional): Connections between the coordinates. Defaults to
                this set.
            domain (dictionary, optional): Description of the simulation domain. Should
                contain fields xmin, xmax, ymin, ymax. Defaults to this set.
            nx, ny (int, optional): Number of boxes in x and y direction. Defaults
                to 10.

        Returns:
            np.array (nx x ny): Number of centers within each box, divided by the measure
                of the specified domain.

        """
        if p is None:
            p = self.pts
        if e is None:
            e = self.edges
        if domain is None:
            domain = self.domain

        p = np.atleast_2d(p)

        # Special treatment when the point array is empty
        if p.shape[1] == 0:
            if p.shape[0] == 1:
                return np.zeros(nx)
            else:  # p.shape[0] == 2
                return np.zeros((nx, ny))

        pc = self._compute_center(p, e)

        if p.shape[0] == 1:
            x0, dx = self._decompose_domain(domain, nx, ny)
            num_occ = np.zeros(nx)
            for i in range(nx):
                hit = np.logical_and.reduce(
                    [pc[0] > (x0 + i * dx), pc[0] <= (x0 + (i + 1) * dx)]
                )
                num_occ[i] = hit.sum()

            return num_occ.astype(np.int) / self.domain_measure(domain)

        elif p.shape[0] == 2:
            x0, y0, dx, dy = self._decompose_domain(domain, nx, ny)
            num_occ = np.zeros((nx, ny))
            # Can probably do this more vectorized, but for now, a for loop will suffice
            for i in range(nx):
                for j in range(ny):
                    hit = np.logical_and.reduce(
                        [
                            pc[0] > (x0 + i * dx),
                            pc[0] < (x0 + (i + 1) * dx),
                            pc[1] > (y0 + j * dy),
                            pc[1] < (y0 + (j + 1) * dy),
                        ]
                    )
                    num_occ[i, j] = hit.sum()

            return num_occ / self.domain_measure(domain)

        else:
            raise ValueError("Have not yet implemented 3D geometries")

        self.intensity = num_occ

    def set_length_distribution(self, dist, params):
        self.dist_length = {"dist": dist, "param": params}

    def set_angle_distribution(self, dist, params):
        self.dist_angle = {"dist": dist, "param": params}

    def set_intensity_map(self, box):
        self.intensity = box

    def _fracture_from_center_angle_length(self, p, angles, lengths):
        """ Generate fractures from a marked-point representation.

        Parameters:
            p (np.array, 2 x num_frac): Center points of the fractures.
            angles (np.array, num_frac): Angle from the x-axis of the fractures.
                Measured in radians.
            lengths (np.array, num_frac): Length of the fractures

        Returns:
            np.array (2 x 2 * num_frac): Start and endpoints of the fractures
            np.array (2 x num_frac): For each fracture, the start and endpoint,
                in terms of indices in the point array.

        """
        num_frac = lengths.size
        # pdb.set_trace()

        start = p + 0.5 * lengths * np.vstack((np.cos(angles), np.sin(angles)))
        end = p - 0.5 * lengths * np.vstack((np.cos(angles), np.sin(angles)))

        pts = np.hstack((start, end))

        e = np.vstack((np.arange(num_frac), num_frac + np.arange(num_frac)))
        return pts, e

    def domain_measure(self, domain=None):
        """ Get the measure (length, area) of a given box domain, specified by its
        extensions stored in a dictionary.

        The dimension of the domain is inferred from the dictionary fields.

        Parameters:
            domain (dictionary, optional): Should contain keys 'xmin' and 'xmax'
                specifying the extension in the x-direction. If the domain is 2d,
                it should also have keys 'ymin' and 'ymax'. If no domain is specified
                the domain of this object will be used.

        Returns:
            double: Measure of the domain.

        """
        if domain is None:
            domain = self.domain
        if "ymin" and "ymax" in domain.keys():
            return (domain["xmax"] - domain["xmin"]) * (domain["ymax"] - domain["ymin"])
        else:
            return domain["xmax"] - domain["xmin"]

    def _define_centers_by_boxes(self, domain, distribution="poisson", intensity=None):
        """ Define center points of fractures, intended used in a marked point
        process.

        The domain is assumed decomposed into a set of boxes, and fracture points
        will be allocated within each box, according to the specified distribution
        and intensity.

        A tacit assumption is that the domain and intensity map corresponds to
        values used in and computed by count_center_point_densities. If this is
        not the case, scaling errors of the densities will arise. This should not
        be difficult to generalize, but there is no time right now.

        The implementation closely follows y Xu and Dowd:
            A new computer code for discrete fracture network modelling
            Computers and Geosciences, 2010

        Parameters:
            domain (dictionary): Description of the simulation domain. Should
                contain fields xmin, xmax, ymin, ymax.
            intensity (np.array, nx x ny): Intensity map, mean values for fracture
                density in each of the boxes the domain will be split into.
            distribution (str, default): Specify which distribution is followed.
                For now a placeholder value, only 'poisson' is allowed.

        Returns:
             np.array (2 x n): Coordinates of the fracture centers.

        Raises:
            ValueError if distribution does not equal poisson.

        """
        if distribution != "poisson":
            return ValueError("Only Poisson point processes have been implemented")
        # Intensity scaled to this domain
        if intensity is None:
            intensity = self.intensity * self.domain_measure(domain)

        nx, ny = intensity.shape
        num_boxes = intensity.size

        max_intensity = intensity.max()

        x0, y0, dx, dy = self._decompose_domain(domain, nx, ny)

        # It is assumed that the intensities are computed relative to boxes of the
        # same size that are assigned in here
        area_of_box = 1

        pts = np.empty(num_boxes, dtype=np.object)

        # First generate the full set of points with maximum intensity
        counter = 0
        for i in range(nx):
            for j in range(ny):
                num_p_loc = stats.poisson(max_intensity * area_of_box).rvs(1)[0]
                p_loc = np.random.rand(2, num_p_loc)
                p_loc[0] = x0 + i * dx + p_loc[0] * dx
                p_loc[1] = y0 + j * dy + p_loc[1] * dy
                pts[counter] = p_loc
                counter += 1

        # Next, carry out a thinning process, which is really only necessary if the intensity is non-uniform
        # See Xu and Dowd Computers and Geosciences 2010, section 3.2 for a description
        counter = 0
        for i in range(nx):
            for j in range(ny):
                p_loc = pts[counter]
                threshold = np.random.rand(p_loc.shape[1])
                delete = np.where(intensity[i, j] / max_intensity < threshold)[0]
                pts[counter] = np.delete(p_loc, delete, axis=1)
                counter += 1

        return np.array(
            [pts[i][:, j] for i in range(pts.size) for j in range(pts[i].shape[1])]
        ).T

    def _decompose_domain(self, domain=None, nx=1, ny=1):
        if domain is None:
            domain = self.domain
        x0 = domain["xmin"]
        dx = (domain["xmax"] - domain["xmin"]) / nx

        if "ymin" in domain.keys() and "ymax" in domain.keys():
            y0 = domain["ymin"]
            dy = (domain["ymax"] - domain["ymin"]) / ny
            return x0, y0, dx, dy
        else:
            return x0, dx

    def _generate_from_distribution(self, num_fracs, dist_a):
        if isinstance(dist_a, dict):
            if isinstance(dist_a["param"], dict):
                return dist_a["dist"].rvs(**dist_a["param"], size=num_fracs)
            else:
                return dist_a["dist"].rvs(*dist_a["param"], size=num_fracs)
        else:
            return dist_a.rvs(size=num_fracs)

    def _candidate_is_too_close(self, p_new, p, data):
        # Measure distance from a new fracture to existing fractures
        if p.shape[1] < 2:
            return False

        start_set = p[:, ::2]
        end_set = p[:, 1::2]

        dist, *rest = pp.cg.dist_segment_segment_set(
            p_new[:, 0], p_new[:, 1], start_set, end_set
        )

        allowed_dist = data.get("minimum_fracture_spacing", 0)
        return dist.min() < allowed_dist

    def _generate_by_intensity(self, domain=None):
        """ Generate a realization of a fracture network from the statistical distributions
        represented in this object.

        The function relies on the statistical properties of the fracture set
        being known, in the form of attributes:

            dist_angle: Statistical distribution of orientations. Should be a dictionary
                with fields 'dist' and 'param'. Here, 'dist' should point to a
                scipy.stats.distribution, or another object with a function
                rvs to draw random variables, while 'param' points to the parameters
                passed on to dist.rvs.

            dist_length: Statistical distribution of length. Should be a dictionary
                with fields 'dist' and 'param'. Here, 'dist' should point to a
                scipy.stats.distribution, or another object with a function
                rvs to draw random variables, while 'param' points to the parameters
                passed on to dist.rvs.

            intensity (np.array): Frequency map of fracture centers in the domain.

        By default, these will be computed by this method. The attributes can
        also be set externally.

        Parameters:
            domain (dictionary, not in use): Future use will include a scaling of
                intensity to fit with another domain. For now, this field is not
                used.
            fit_distributions (boolean, optional): If True, compute the statistical
                properties of the network. Defaults to True.

        Returns:
            FractureSet: A new fracture set generated according to the statistical
                properties of this object.

        """
        if domain is None:
            domain = self.domain

        # First define points
        p_center = self._define_centers_by_boxes(domain)
        # bookkeeping
        if p_center.size == 0:
            num_fracs = 0
        else:
            num_fracs = p_center.shape[1]

        # Then assign length and orientation
        angles = self._generate_from_distribution(num_fracs, self.dist_angle)
        lengths = self._generate_from_distribution(num_fracs, self.dist_length)

        p, e = self._fracture_from_center_angle_length(p_center, angles, lengths)

        return p, e

    def generate(self, criterion, data, return_network=True):
        if "domain" not in data.keys():
            data["domain"] = self.domain
        if criterion.lower().strip() == "counting":
            num_frac = 0
            p = np.zeros((2, 0))
            while num_frac < data["target_number"]:
                loc_p = self._generate_single_fracture(data)
                if not self._candidate_is_too_close(loc_p, p, data):
                    p = np.c_[p, loc_p]
                    num_frac += 1

            e = np.vstack((2 * np.arange(num_frac), 1 + 2 * np.arange(num_frac)))

        elif criterion.lower().strip() == "intensity":
            # Generate from specified intensity map
            p, e = self._generate_by_intensity()

        elif criterion.lower().strip() == "length":

            def dist_points(a, b):
                return np.sqrt(np.sum((a - b) ** 2))

            full_length = 0
            p = np.zeros((2, 0))
            while full_length < data["target_length"]:
                loc_p = self._generate_single_fracture(data)
                if not self._candidate_is_too_close(loc_p, p, data):
                    p = np.c_[p, loc_p]
                    full_length += dist_points(loc_p[:, 0], loc_p[:, 1])

            num_frac = p.shape[1] / 2
            e = np.vstack((2 * np.arange(num_frac), 1 + 2 * np.arange(num_frac)))

        else:
            raise ValueError("Unknown truncation rule" + str(criterion))

        if return_network:
            return StochasticFractureNetwork2d(
                p, e.astype(np.int), domain=data["domain"]
            )
        else:
            return p, e.astype(np.int)

    def _generate_single_fracture(self, data):
        if self.intensity is None:
            self.intensity = np.array([[10 / self.domain_measure()]])

        while True:
            cp_tmp = self._generate_centers(center_mode="poisson", data=data)
            if cp_tmp.size > 0:
                break

        # If the intensity map has several blocks, there will be a spatial
        # ordering associated with cp. Shuffle the
        shuffle_ind = np.argsort(np.random.rand(cp_tmp.shape[-1]))
        #   pdb.set_trace()
        cp = cp_tmp[:, shuffle_ind][:, 0].reshape((-1, 1))

        orientation = self._generate_from_distribution(1, self.dist_orientation)
        lengths = self._generate_from_distribution(1, self.dist_length)
        return self._fracture_from_center_angle_length(cp, orientation, lengths)[0]

    def _generate_centers(self, center_mode, data):
        domain = data.get("domain")
        if center_mode.lower().strip() == "poisson":
            intensity = data.get("center_intensity_map", None)  # Or self.intensity?

            cp = self._define_centers_by_boxes(intensity=intensity, domain=domain)

        elif center_mode.lower().strip() == "ladder":
            center_line = data.get("center_line").reshape((-1, 1))
            center_line = center_line / np.linalg.norm(center_line)
            start_of_line = data.get("center_line_start")
            spacing_along = data.get("spacing_along_line")
            cp = [start_of_line]
            counter = 0
            # Generate points along the line
            while True:
                dx = spacing_along.draw(1)
                cp.append(cp[counter] + center_line * dx)
                counter += 1
            # Perturb orthorgonally to the line
            wing_line = center_line[::-1]

            perturbation_across = data.get("perturbation_across_line", None)
            if perturbation_across is None:
                perturbation_across = distributions.Uniform(0)

            for pi, p in enumerate(cp):
                cp[pi] = p + wing_line * perturbation_across.rvs(1)
        else:
            raise ValueError("Unknown center distribution " + center_mode)

        return cp


class FractureChildrenGenerator(StochasticFractureGenerator):
    def compute_density_along_line(self, p, start, end, **kwargs):

        p_x, loc_edge, domain_loc, _ = self._project_points_to_line(p, start, end)

        # Count the point density along this fracture.
        return frac_gen.count_center_point_densities(
            p_x, loc_edge, domain_loc, **kwargs
        )

    def _project_points_to_line(self, p, start, end):
        if p.ndim == 1:
            p = p.reshape((-1, 1))
        if start.ndim == 1:
            start = start.reshape((-1, 1))
        if end.ndim == 1:
            end = end.reshape((-1, 1))

        def _to_3d(pt):
            return np.vstack((pt, np.zeros(pt.shape[1])))

        p -= start
        end -= start
        theta = np.arctan2(end[1], end[0])

        assert np.abs(end[0] * np.sin(theta) + end[1] * np.cos(theta)) < 1e-5

        start_x = 0
        p_x = p[0] * np.cos(theta) - p[1] * np.sin(theta)
        end_x = end[0] * np.cos(theta) - end[1] * np.sin(theta)

        if end_x < start_x:
            domain_loc = {"xmin": end_x, "xmax": start_x}
        else:
            domain_loc = {"xmin": start_x, "xmax": end_x}

        # The density calculation computes the center of each fracture,
        # based on an assumption that the fracture consist of two points.
        # Make a line out of the points, with identical start and end points
        loc_edge = np.tile(np.arange(p_x.size), (2, 1))
        return p_x, loc_edge, domain_loc, theta

    def generate(self):
        raise ValueError("Not implemented. Use a subclass")

    def _draw_num_children(self, parent_realiz, pi):
        """ Draw the number of children for a fracture based on the statistical
        distribution.

        Parameters:
            parent_realiz (FractureSet): Fracture set for
            pi (int):

            These parameters are currently not in use. In the future, the number
            of children should scale with the length of the parent fracture.
        """
        nc = frac_gen.generate_from_distribution(1, self.dist_num_children)
        return np.round(nc * parent_realiz.length()[pi]).astype(np.int)[0]


class ConstrainedChildrenGenerator(StochasticFractureGenerator):
    def __init__(self, parent, side_distribution=None, **kwargs):
        super(ConstrainedChildrenGenerator, self).__init__(**kwargs)

        self.parent = parent

        if side_distribution is None:
            self.dist_side = distributions.Signum()

    def generate(self, pi=None, data=None):
        """ Generate a fracture that has one node on a parent fracture.

        The fracture is constructed as a ray from the parent point, with length and
        orientation drawn according to the specified statistical distributions.

        The generated fracture may cross other fractures, parents or children.

        """
        if pi is None:
            pi = self._pick_parent()

        # Assign equal probability that the points are on each side of the parent
        side = self._generate_from_distribution(1, self.dist_side)

        # Draw length and distribution
        child_angle = self._generate_from_distribution(1, self.dist_orientation)
        child_length = self._generate_from_distribution(1, self.dist_length)
        # Vector that spans the segment of the new fracture
        vec = np.vstack((np.cos(child_angle), np.sin(child_angle))) * child_length

        # Place the new fracture randomly along the chosen parent
        parent_start, parent_end = self.parent.get_points(pi)
        along_parent = parent_end - parent_start
        child_start = parent_start + np.random.rand(1) * along_parent

        child_end = child_start + side * vec

        self.parent.add_branch(pi, child_start)

        return np.hstack((child_start, child_end))

    def _pick_parent(self):
        # Pick a parent, with probabilities scaling with parent length
        cum_length = self.parent.length().cumsum()
        cum_length /= cum_length[-1]

        r = np.random.rand(1)
        return np.nonzero(r < cum_length)[0][0]


class DoublyConstrainedChildrenGenerator(StochasticFractureGenerator):
    def __init__(self, parent, side_distribution=None, data=None, **kwargs):
        super(DoublyConstrainedChildrenGenerator, self).__init__(**kwargs)

        self.search_direction = self._get_search_vector()

        self.parent = parent

        if side_distribution is None:
            self.dist_side = distributions.Signum()

        pair_array, point_first, point_second = self._trace_rays_from_fracture_tips()
        self._pairs_of_parents(pair_array, point_first, point_second)

    def generate(self, pi=None, data=None):
        # pi is index of the parent pair
        if pi is None:
            pi = self._pick_parent_pair()

        fi = self.pairs[0, pi]
        si = self.pairs[1, pi]

        dist, *rest = pp.cg.dist_two_segments(*self.parent.get_points(fi),
                                              *self.parent.get_points(si))
        if dist > np.min(self.parent.length()[[fi, si]]):
            return np.zeros((2, 0))

        # Assign equal probability that the points are on each side of the parent
        side = self._generate_from_distribution(1, self.dist_side)

        vec = self.search_direction

        start = self.interval_points[pi][:, 0]
        end = self.interval_points[pi][:, 1]

        along_parent = end - start

        child_start = (start + np.random.rand(1) * along_parent).reshape((-1, 1))

        _, _, _, child_end, *rest = self._find_neighbors(vec, child_start)
        if child_end.shape[1] > 1:
            if side > 0:
                child_end = child_end[:, 0].reshape((-1, 1))
            else:
                child_end = child_end[:, 1].reshape((-1, 1))

        self.parent.add_branch(fi, child_start)
        self.parent.add_branch(si, child_end)

        return np.hstack((child_start, child_end))

    def _pick_parent_pair(self):

        cum_length = self.interval_lengths.cumsum()
        cum_length /= cum_length[-1]
        r = np.random.rand(1)
        return np.nonzero(r < cum_length)[0][0]


    def _pairs_of_parents(self, pair_array, point_first, point_second):
        # Find parents that are visible to each other along a ray of fixed orientation.

        # Parents
        pairs = []
        interval_first = {}
        interval_second = {}

        if len(pair_array) == 0:
            self.parent_pairs = pairs
            self.interval_first = interval_first
            self.interval_second = interval_second
            return

        # Loop over all fractures
        for fi in range(self.parent.num_frac):

            # Find all preliminary pairs where the current fracture is part
            hit_first = np.where(pair_array[0] == fi)[0]
            hit_second = np.where(pair_array[1] == fi)[0]

            start, end = self.parent.get_points(fi)

            # Set up a point set along the fracture consistng of end points, togehter
            # with all points where rays from other fractures hit.
            # Together, these will form the boundaries of the intervals where the
            # fracture has a certain neighbor (along the search vector)
            isect_pt = np.hstack(
                (start, end, point_first[:, hit_first], point_second[:, hit_second])
            )
            # Sort points along the fracture
            isect_pt, *rest = pp.utils.setmembership.unique_columns_tol(isect_pt)
            dist = np.sum((isect_pt - start) ** 2, axis=0)
            p = isect_pt[:, np.argsort(dist)]

            # Storage of what is the current neighbor on the two sides of the fracture
            active_neigh_pos = None
            active_neigh_neg = None

            vec = self.search_direction

            # Loop over intersection points. These all mark the start or end of a neighbor
            # interval.
            for pi in range(p.shape[1]):

                # Find neighbors of this point
                neigh, second_neigh, pt_self, pt_first, pt_second, is_pos = self._find_neighbors(
                    vec, p[:, pi].reshape((-1, 1))
                )
                # Loop over the neighbors (can be on both sides)
                for ni in range(len(neigh)):
                    # Closest and second closest neighbor
                    loc_neigh = neigh[ni]
                    loc_second_neigh = second_neigh[ni]
                    # Point on this fracture, on the closest fracture, and potentially
                    # on the second closest (if there is one)
                    pself = pt_self[:, ni].reshape((-1, 1))
                    pf = pt_first[:, ni].reshape((-1, 1))
                    if loc_second_neigh is not None:  # None signifies there are no neighbors here
                        # A bit of back and forth, depending on how many points there are
                        try:
                            psec = pt_second[:, ni].reshape((-1, 1))
                        except:
                            psec = pt_second[ni][0].reshape((-1, 1)).astype(np.float)

                    # Check which side of the fracture we are on
                    if is_pos[ni]:
                        # Check if there currently is a neighbor on this side
                        if active_neigh_pos is None:
                            # We have found the start of a new neighbor
                            active_neigh_pos = loc_neigh
                            pairs.append((fi, loc_neigh))
                            pos_pt_self = pself
                            pos_pt_other = pf
                            active_pos_pair = (fi, loc_neigh)

                        else:
                            # We found (this fracture was hit by) the end of a neighbor.
                            # The next neighbor is the second closest to the hit point in
                            # the direction of vec
                            if active_pos_pair in interval_second.keys():
                                tmp = interval_first[active_pos_pair]
                                tmp.append(np.hstack((pos_pt_self, pself)))
                                interval_first[active_pos_pair] = tmp
                            else:
                                interval_first[active_pos_pair] = [
                                    np.hstack((pos_pt_self, pself))
                                ]

                            if active_neigh_pos == loc_neigh:
                                active_neigh_pos = loc_second_neigh
                                if active_pos_pair in interval_second.keys():
                                    tmp = interval_second[active_pos_pair]
                                    tmp.append(np.hstack((pos_pt_other, pf)))
                                    interval_second[active_pos_pair] = tmp
                                else:
                                    interval_second[active_pos_pair] = [
                                        np.hstack((pos_pt_other, pf))
                                    ]

                            else:
                                active_neigh_pos = loc_neigh
                                if active_pos_pair in interval_second.keys():
                                    tmp = interval_second[active_pos_pair]
                                    tmp.append(np.hstack((pos_pt_other, psec)))
                                    interval_second[active_pos_pair] = tmp
                                else:
                                    interval_second[active_pos_pair] = [
                                        np.hstack((pos_pt_other, psec))
                                    ]
                            # End the current interval

                            if pi < (p.shape[1] - 1) and active_neigh_pos is not None:
                                pairs.append((fi, active_neigh_pos))
                                pos_pt_self = pself
                                if active_neigh_pos == loc_neigh:
                                    pos_pt_other = pf
                                else:
                                    pos_pt_other = psec
                                active_pos_pair = (fi, active_neigh_pos)
                    else:
                        # This is the other side of the fracture

                        # Check if there currently is a neighbor on this side
                        if active_neigh_neg is None:
                            # We have found the start of a new neighbor
                            active_neigh_neg = loc_neigh
                            pairs.append((fi, loc_neigh))
                            neg_pt_self = pself
                            neg_pt_other = pf
                            active_neg_pair = (fi, loc_neigh)

                        else:
                            # We found (this fracture was hit by) the end of a neighbor.
                            # The next neighbor is the second closest to the hit point in
                            # the direction of vec
                            if active_neg_pair in interval_second.keys():
                                tmp = interval_first[active_neg_pair]
                                tmp.append(np.hstack((neg_pt_self, pself)))
                                interval_first[active_neg_pair] = tmp
                            else:
                                interval_first[active_neg_pair] = [
                                    np.hstack((neg_pt_self, pself))
                                ]

                            # pdb.set_trace()
                            if active_neigh_neg == loc_neigh:
                                active_neigh_neg = loc_second_neigh
                                if active_neg_pair in interval_second.keys():
                                    tmp = interval_second[active_neg_pair]
                                    tmp.append(np.hstack((neg_pt_other, pf)))
                                    interval_second[active_neg_pair] = tmp
                                else:
                                    interval_second[active_neg_pair] = [
                                        np.hstack((neg_pt_other, pf))
                                    ]
                            else:
                                active_neigh_neg = loc_neigh
                                if active_neg_pair in interval_second.keys():
                                    tmp = interval_second[active_neg_pair]
                                    tmp.append(np.hstack((neg_pt_other, psec)))
                                    interval_second[active_neg_pair] = tmp
                                else:
                                    interval_second[active_neg_pair] = [
                                        np.hstack((neg_pt_other, psec))
                                    ]

                            # End the current interval

                            if pi < (p.shape[1] - 1) and active_neigh_neg is not None:
                                pairs.append((fi, active_neigh_neg))
                                neg_pt_self = pself
                                if active_neigh_neg == loc_neigh:
                                    neg_pt_other = pf
                                else:
                                    neg_pt_other = psec
                                active_neg_pair = (fi, active_neigh_neg)

        pairs = np.array([p for p in pairs]).T

        # We have found pairs more than once: First by tracing rays from both pair
        # members. Two fractures can also form multiple pairs (think long parallel
        # fratures, with a smaller inbetween at the middle of the larger ones)
        del_ind = []
        for fi in range(pairs.shape[1]):
            f = pairs[0, fi]
            s = pairs[1, fi]
            # Find other pairs of the same fracture
            other = np.where(
                np.logical_or(
                    np.logical_and(pairs[0] == f, pairs[1] == s),
                    np.logical_and(pairs[0] == s, pairs[1] == f),
                )
            )[0]
            assert other.size >= 1
            if f < s:
                for tmp in other:
                    if tmp > fi:
                        del_ind.append(tmp)
                try:
                    interval_first.pop((s, f))
                    interval_second.pop((s, f))
                except:
                    continue

        del_ind = np.unique(np.asarray(del_ind))
        pairs = np.delete(pairs, del_ind, axis=1)

        self.parent_pairs = pairs
        self.interval_first = interval_first
        self.interval_second = interval_second

        points = []
        lengths = []
        parent_pair = []

        for k, v in interval_first.items():
            for p in v:
                l = np.sqrt(np.sum((p[:, 1] - p[:, 0])**2))
                lengths.append(l)
                points.append(p)
                parent_pair.append(k)

        self.pairs = np.array([p for p in parent_pair]).T
        self.interval_lengths = np.cumsum(lengths)
        self.interval_points = points

    def _compare_arrays(self, a, b, tol=1e-4):
        """ Compare two arrays and check that they are equal up to a column permutation.

        Typical usage is to compare coordinate arrays.

        Parameters:
            a, b (np.array): Arrays to be compared. W
            tol (double, optional): Tolerance used in comparison.
            sort (boolean, defaults to True): Sort arrays columnwise before comparing

        Returns:
            True if there is a permutation ind so that all(a[:, ind] == b).
        """
        if not np.all(a.shape == b.shape):
            return False

        for i in range(a.shape[1]):
            dist = np.sum((b - a[:, i].reshape((-1, 1))) ** 2, axis=0)
            if dist.min() > tol:
                return False
        for i in range(b.shape[1]):
            dist = np.sum((a - b[:, i].reshape((-1, 1))) ** 2, axis=0)
            if dist.min() > tol:
                return False
        return True

    def _trace_rays_from_fracture_tips(self):
        """ Identify pairs of fractures that lie in direct sight of each other
        along a specified angle.

        The pairs are found by tracing rays from the end points of fractures,
        and look for intersections with other fractures. In the example below,
        all three (horizontal) left fractures find each other, while the
        right fractures hits nothing.
            ___________________
               /  /     /     /
              /  /_____/     /
             /  /     /     /      __________
            /__/_____/_____/___

        The current al

        Parameters:
            parents (FractureSet): Fracture set for which we look for pairs.
            angle (double, radians): Angle of search direction

        Returns:
            np.array, 2 X num_pairs: Indices of the edges forming unique pairs.
                Sorted along each column. The columns are ordered so that
                arr[0] is non-decreasing.

        """

        """
        Update: The algorithm above will find the intersection points between
        rays from endpoints of other fractures, but it cannot find all
        pairs of potential visibility combinations. For this, use a divide and
        conquer (like?) algorithm: Sort all intersection points from other fractures
        on the central line. Start on one end of the fracture, the ray from that
        point will

        for each fracture as main:



            if the ray hits another fracture, initialize the visibility branches
                with this pair
            else:
                visibility branch is empty, there are no active others

            while there are more intersection points (spatially sorted) with rays from other fractures
                Check whether the other fracture has been found before (it is active)
                if yes,
                    deactivate the other fracture,
                    mark this interval as a pairing between the main and other fracture
                    find the fracture on the other side of the other fracture - this is the new active one
                if not - we are meeting a new active fracture
                    mark the interval behind us as a pairing of the new and (old) active fracture
                    update index to new fracture

            To avoid double accounting of pairs, only add to global list if the
            main fracture has the lower index.
            If two fractures have multiple common intervals, store this as separate
            visibility branches

            Data structure necessary: All fractures must have access to intersection
            points from other fractures.
            When deactivating a fracture, it should be easy to find the one on the
            other side.


        NOTE: To handle intersections of other fractures, we need to work on
        branches, not full fractures.
        """
        # Data structure for storage
        parent_pairs = []

        # For all rays that hit another fracture, store the start point of the
        # ray in the point_first, and the intersection of the ray with the
        # other fracture in point_second
        point_first = np.zeros((2, 0))
        point_second = np.zeros((2, 0))

        # Search direction, use the same for all fractures
        vec = self.search_direction

        # Loop over all fractures, look for pairs that involves this fracture.
        # We may find the same pair twice, once for each member of the pair.
        # Uniqueness is enforced below
        for fi in range(self.parent.num_frac):
            # Start and end_points of this fracture
            start, end = self.parent.get_points(fi)

            loc_pairs, _, loc_isect_first, loc_isect_sec, *rest = self._find_neighbors(
                vec, start, end
            )
            # The ray hit nothing
            if len(loc_pairs) == 0:
                continue
            # We may get up to four hits: One on each side for start and endpoint of the
            # main fracture.
            # Stor intersection points
            point_first = np.hstack((point_first, loc_isect_first))
            point_second = np.hstack((point_second, loc_isect_sec))
            # We have found a new pair
            for pi in range(len(loc_pairs)):
                parent_pairs.append((fi, loc_pairs[pi]))

        pair_array = np.array([p for p in parent_pairs]).T

        return pair_array, point_first, point_second

    def _get_search_vector(self):
        """ Get angle of search for intersection.
        """
        # We are interested in any intersection in the direction of the specified angle.
        # Create a vector with the right direction, and length equal to the maximum
        # size of the domain.
        _, _, dx, dy = self._decompose_domain()
        length = np.maximum(dx, dy)

        angle = self._generate_from_distribution(1000, self.dist_orientation).mean()
        vec = np.vstack((np.cos(angle), np.sin(angle))) * length

        return vec

    def _find_neighbors(self, vec, start, end=None):
        # The start point of the segments are twice the start, twice the end
        # of this fracture.
        import pdb
        #pdb.set_trace()
        # Start and endpoints of the parents.
        start_parent, end_parent = self.parent.get_points()

        if end is None:
            offshots_start = np.hstack((start, start))
            start_pos = start + vec
            start_neg = start - vec
            # End points of the shooting segments
            offshots_end = np.hstack((start_pos, start_neg))
        else:
            offshots_start = np.hstack((start, start, end, end))

            # From the nodes of this fracture, shoot segments along the vector on
            # both sides of the fracture.
            start_pos = start + vec
            start_neg = start - vec
            end_pos = end + vec
            end_neg = end - vec
            # End points of the shooting segments
            offshots_end = np.hstack((start_pos, start_neg, end_pos, end_neg))

        neighbor = []
        second_neighbor = []
        isect_self = []
        isect_first = []
        isect_second = []
        is_pos = []

        # From the nodes of this fracture, shoot segments along the vector on
        # both sides of the fracture.

        # Loop over all off-shots, see if they hit other fractures in the set.
        for oi in range(offshots_start.shape[1]):
            # Start and endpoint of the offshot
            s = offshots_start[:, oi].reshape((-1, 1))
            e = offshots_end[:, oi].reshape((-1, 1))
            # Compute distance between this point and all other segments in the network
            d, cp, cg_seg = pp.cg.dist_segment_segment_set(
                s, e, start_parent, end_parent
            )
            # Count hits, where the distance is very small
            hit = np.where(d < self.parent.tol)[0]
            if hit.size == 0:
                # There should be at least one hit, namely the start and end point
                # of fi
                raise ValueError("Error when finding pairs of fractures")
            elif hit.size == 1:
                # The offshot did not hit anything. We can move on
                continue
            else:
                # The offshot has hit at least one fracture.
                # Compute distance from all closest points to the start
                dist_start = np.sqrt(np.sum((s - cp[:, hit]) ** 2, axis=0))
                # Find the first point along the line, away from the start
                first_constraint = np.argsort(dist_start)[1]
                neighbor.append(hit[first_constraint])
                # Store the intersection point of the first and second
                # fracture
                isect_self.append(s)
                isect_first.append(cp[:, hit[first_constraint]])

                if hit.size > 2:
                    second_constraint = np.argsort(dist_start)[2]
                    second_neighbor.append(hit[second_constraint])
                    isect_second.append(cp[:, hit[second_constraint]])
                else:
                    second_neighbor.append(None)
                    isect_second.append(None)

                is_pos.append(oi % 2 == 0)

        def arr_to_np(a):
            b = np.array([p for p in a]).T
            if b.ndim == 3:
                b = b.squeeze()
            if b.size == 2:
                return b.reshape((-1, 1))
            else:
                return b

        #        isect_self = np.array([p for p in isect_self]).T
        #        isect_first = np.array([p for p in isect_first]).T
        #        isect_second = np.array([p for p in isect_second]).T
        isect_self = arr_to_np(isect_self)
        isect_first = arr_to_np(isect_first)
        isect_second = arr_to_np(isect_second)

        return neighbor, second_neighbor, isect_self, isect_first, isect_second, is_pos


class IsolatedFractureChildernGenerator(FractureChildrenGenerator):
    def _generate_isolated_fractures(self, children_points, start_parent, end_parent):

        if children_points.size == 0:
            return np.empty((2, 0)), np.empty((2, 0))

        dx = end_parent - start_parent
        theta = np.arctan2(dx[1], dx[0])

        if children_points.ndim == 1:
            children_points = children_points.reshape((-1, 1))

        num_children = children_points.shape[1]

        dist_from_parent = frac_gen.generate_from_distribution(
            num_children, self.dist_from_parents
        )

        # Assign equal probability that the points are on each side of the parent
        side = frac_gen.generate_from_distribution(num_children, self.dist_side)

        # Vector from the parent line to the new center points
        vec = np.vstack((-np.sin(theta), np.cos(theta))) * dist_from_parent

        children_center = children_points + side * vec

        child_angle = frac_gen.generate_from_distribution(num_children, self.dist_angle)
        child_length = frac_gen.generate_from_distribution(
            num_children, self.dist_length
        )

        p_start = children_center + 0.5 * child_length * np.vstack(
            (np.cos(child_angle), np.sin(child_angle))
        )
        p_end = children_center - 0.5 * child_length * np.vstack(
            (np.cos(child_angle), np.sin(child_angle))
        )

        p = np.hstack((p_start, p_end))
        edges = np.vstack(
            (np.arange(num_children), num_children + np.arange(num_children))
        )

        return p, edges

    def _fit_dist_from_parent_distribution(self, ks_size=100, p_val_min=0.05):
        """ For isolated fractures, fit a distribution for the distance from
        the child center to the parent fracture, orthogonal to the parent line.

        The function also evaluates the fitness of the chosen distribution by a
        Kolgomorov-Smirnov test.

        The function should be called after the field self.isolated_stats['center_distance']
        has been assigned, e.g. by calling self.compute_statistics()

        IMPLEMENTATION NOTE: The selection of appropriate distributions is a bit
        unclear. For the moment, we chose between uniform, lognormal and
        exponential distributions. More generally, this function can be made
        much more advanced, see for instance Xu and Dowd (Computers and
        Geosciences, 2010).

        Parameters:
            ks_size (int, optional): The number of realizations used in the
                Kolmogorov-Smirnov test. Defaults to 100.
            p_val_min (double, optional): P-value used in Kolmogorev-Smirnov test
                for acceptance of the chosen distribution. Defaults to 0.05.

        Returns:
            dictionary, with fields 'dist': The distribution with best fit.
                                    'param': Fitted parameters for the best
                                        ditribution.
                                    'p_val': P-value for the best distribution
                                        and parameters.

            If the fracture set contains no isolated fractures, and empty
            dictionary is returned.

        Raises:
            ValueError if none of the candidate distributions give a satisfactory
                fit.

        """
        data = self.isolated_stats["center_distance"]

        # Special case of no isolated fractures.
        if data.size == 0:
            return {}

        # Set of candidate distributions. This is somewhat arbitrary, better
        # options may exist
        candidate_dist = np.array([stats.uniform, stats.lognorm, stats.expon])
        # Fit each distribution
        dist_fit = np.array([d.fit(data, floc=0) for d in candidate_dist])

        # Inline function for Kolgomorov-Smirnov test
        ks = lambda d, p: stats.ks_2samp(data, d.rvs(*p, size=ks_size))[1]
        # Find the p-value for each of the candidate disributions, and their
        # fitted parameters
        p_val = np.array([ks(d, p) for d, p in zip(candidate_dist, dist_fit)])
        best_fit = np.argmax(p_val)

        if p_val[best_fit] < p_val_min:
            raise ValueError("p-value not satisfactory for length fit")

        self.dist_from_parents = {
            "dist": candidate_dist[best_fit],
            "param": dist_fit[best_fit],
            "pval": p_val[best_fit],
        }

    def _compute_line_density_isolated_nodes(self, isolated):
        # To ultimately describe the isolated fractures as a marked point
        # process, with stochastic location in terms of its distribution along
        # the fracture and perpendicular to it, we describe the distance from
        # the child center to its parent line.

        # There may be some issues with accumulation close to the end points
        # of the parent fracture; in particular if the orientation of the
        # child is far from perpendicular (meaning that a lot of it is outside
        # the 'span' of the parent), or if multiple parents are located nearby,
        # and we end up taking the distance to one that is not the natural
        # parent, whatever that means.
        # This all seems to confirm that ideall, a unique parent should be
        # identified for all children.

        # Start and end points of the parent fractures
        start_parent, end_parent = self.parent.get_points()

        center_of_isolated = 0.5 * (
            self.pts[:, self.edges[0, isolated]] + self.pts[:, self.edges[1, isolated]]
        )
        dist_isolated, closest_pt_isolated = pp.cg.dist_points_segments(
            center_of_isolated, start_parent, end_parent
        )

        # Minimum distance from center to a fracture
        num_isolated = isolated.size
        closest_parent_isolated = np.argmin(dist_isolated, axis=1)

        def dist_pt(a, b):
            return np.sqrt(np.sum((a - b) ** 2, axis=0))

        num_isolated = isolated.size

        # Distance from center of isolated node to the fracture (*not* its
        # prolongation). This will have some statistical distribution
        points_on_line = closest_pt_isolated[
            np.arange(num_isolated), closest_parent_isolated
        ].T
        pert_dist_isolated = dist_pt(center_of_isolated, points_on_line)

        num_occ_all = np.zeros(self.parent.edges.shape[1])

        # Loop over all parent fractures that are closest to some children.
        # Project the children onto the parent, compute a density map along
        # the parent.
        for counter, fi in enumerate(np.unique(closest_parent_isolated)):
            hit = np.where(closest_parent_isolated == fi)[0]
            p_loc = points_on_line[:, hit]
            num_occ_all[fi] = self.compute_density_along_line(
                p_loc, start_parent[:, fi], end_parent[:, fi], nx=1
            )

        return num_occ_all, pert_dist_isolated


class ChildFractureSet(FractureChildrenGenerator):
    """ Fracture set that is defined based on its distance from a member of
    a parent family
    """

    def __init__(self, pts, edges, domain, parent):
        super(ChildFractureSet, self).__init__(pts, edges, domain)

        self.parent = parent

    def generate(self, parent_realiz, domain=None, y_separately=False):
        """ Generate a realization of a fracture network from the statistical distributions
        represented in this object.

        The function relies on the statistical properties of the fracture set
        being known, in the form of attributes:

            dist_angle: Statistical distribution of orientations. Should be a dictionary
                with fields 'dist' and 'param'. Here, 'dist' should point to a
                scipy.stats.distribution, or another object with a function
                rvs to draw random variables, while 'param' points to the parameters
                passed on to dist.rvs.

            dist_length: Statistical distribution of length. Should be a dictionary
                with fields 'dist' and 'param'. Here, 'dist' should point to a
                scipy.stats.distribution, or another object with a function
                rvs to draw random variables, while 'param' points to the parameters
                passed on to dist.rvs.

            dist_num_childern: Statistical distribution of orientations. Should be a
                scipy.stats.distribution, or another object with a function
                rvs to draw random variables.

            fraction_isolated, fraction_one_y: Fractions of the children that should
                on average be isolated and one-y. Should be doubles between
                0 and 1, and not sum to more than unity. The number of both-y
                fractures are 1 - (fraction_isolated + fraction_one_y)

            dist_from_parents: Statistical distribution that gives the distance from
                parent to isolated children, in the direction orthogonal to the parent.
                Should be a dictionary with fields 'dist' and 'param'. Here, 'dist' should
                point to a scipy.stats.distribution, or another object with a function
                rvs to draw random variables, while 'param' points to the parameters
                passed on to dist.rvs.

        These attributes should be set before the method is called.

        Parameters:
            parent_realiz (FractureSet): The parent of the new realization. This will
                possibly be the generated realization of the parent of this object.
            domain (dictionary, not in use): Future use will include a scaling of
                intensity to fit with another domain. For now, this field is not
                used, and the domain is taken as the same as for the original child set.

        Returns:
            FractureSet: A new fracture set generated according to the statistical
                properties of this object.

        """
        if domain is None:
            domain = self.domain

        num_parents = parent_realiz.edges.shape[1]

        # Arrays to store all points and fractures in the new realization
        all_p = np.empty((2, 0))
        all_edges = np.empty((2, 0))

        num_isolated = 0
        num_one_y = 0
        num_both_y = 0

        logger.info("Generate children for fracture set: \n" + str(parent_realiz))

        # Loop over all fractures in the parent realization. Decide on the
        # number of realizations.
        for pi in range(num_parents):
            # Decide on the number of children
            logging.debug("Parent fracture %i", pi)

            if self.points_along_fracture != "distribution":
                num_children = self._draw_num_children(parent_realiz, pi)
                logging.debug("Fracture has %i children", num_children)
            else:
                num_children = 0

            # Find the location of children points along the parent.
            # The interpretation of this point will differ, depending on whether
            # the child is chosen as isolated, one_y or both_y
            children_points = self._draw_children_along_parent(
                parent_realiz, pi, num_children
            )
            num_children = children_points.shape[1]
            # If this fracture has no children, continue
            if num_children == 0:
                continue

            # For all children, decide type of child
            if y_separately:
                is_isolated, is_one_y, is_both_y = self._draw_children_type(
                    num_children, parent_realiz, pi, y_separately=y_separately
                )
                num_isolated += is_isolated.sum()
                num_one_y += is_one_y.sum()
                num_both_y += is_both_y.sum()

                logging.debug(
                    "Isolated children: %i, one y: %i, both y: %i",
                    is_isolated.sum(),
                    is_one_y.sum(),
                    is_both_y.sum(),
                )

            else:
                is_isolated, is_y = self._draw_children_type(
                    num_children, parent_realiz, pi, y_separately=y_separately
                )
                num_isolated += is_isolated.sum()
                num_one_y += is_y.sum()

                logging.debug(
                    "Isolated children: %i, one y: %i", is_isolated.sum(), is_y.sum()
                )

            # Start and end point of parent
            start_parent, end_parent = parent_realiz.get_points(pi)

            # Generate isolated children
            p_i, edges_i = self._generate_isolated_fractures(
                children_points[:, is_isolated], start_parent, end_parent
            )
            # Store data
            num_pts = all_p.shape[1]
            all_p = np.hstack((all_p, p_i))
            edges_i += num_pts

            if y_separately:
                # Generate Y-fractures
                p_y, edges_y = self._generate_y_fractures(children_points[:, is_one_y])
                p_b_y, edges_b_y = self._generate_constrained_fractures(
                    children_points[:, is_both_y], parent_realiz
                )

                # Assemble points
                all_p = np.hstack((all_p, p_y, p_b_y))

                # Adjust indices in point-fracture relation to account for previously
                # added objects
                edges_y += num_pts + p_i.shape[1]
                edges_b_y += num_pts + p_i.shape[1] + p_y.shape[1]

                all_edges = np.hstack((all_edges, edges_i, edges_y, edges_b_y)).astype(
                    np.int
                )
            else:
                p_y, edges_y = self._generate_constrained_fractures(
                    children_points[:, is_y], parent_realiz
                )
                # Assemble points
                all_p = np.hstack((all_p, p_y))

                # Adjust indices in point-fracture relation to account for previously
                # added objects
                edges_y += num_pts + p_i.shape[1]

                all_edges = np.hstack((all_edges, edges_i, edges_y)).astype(np.int)

        new_child = ChildFractureSet(all_p, all_edges, domain, parent_realiz)

        logger.info("Created new child, with properties: \n" + str(new_child))
        logging.debug(
            "Isolated children: %i, one y: %i, both y: %i",
            num_isolated,
            num_one_y,
            num_both_y,
        )

        return new_child

    def _draw_children_along_parent(self, parent_realiz, pi, num_children):
        """ Define location of children along the lines of a parent fracture.

        The interpretation of the resulting coordinate depends on which type of
        fracture the child is: For an isolated node this will be the projection
        of the fracture center onto the parent. For y-nodes, the generated
        coordinate will be the end of the children that intersects with the
        parent.

        For the moment, the points are considered uniformly distributed along
        the parent fracture.

        Parameters:
            parent_realiz (FractureSet): Fracture set representing the parent
                of the realization being generated.
            pi (int): Index of the parent fracture these children will belong to.
            num_children (int): Number of children to be generated.

        Returns:
            np.array, 2 x num_children: Children points along the parent fracture.

        """
        # Start and end of the parent fracture
        start, end = parent_realiz.get_points(pi)
        # Vector along parent
        dx = end - start

        # Random distribution
        if self.points_along_fracture == "random":
            p = start + np.random.rand(num_children) * dx
        elif self.points_along_fracture == "uniform":
            dist = (0.5 + np.arange(num_children)) / (num_children + 1)
            p = start + dist * dx
        elif self.points_along_fracture == "distribution":
            nrm_dx = np.sqrt(np.sum(dx ** 2))
            length = 0
            p = np.empty((2, 0))
            while True:
                length += frac_gen.generate_from_distribution(
                    1, self.dist_along_fracture
                )
                if length > nrm_dx:
                    break
                p = np.hstack((p, start + length * dx / nrm_dx))

        if p.size == 2:
            p = p.reshape((-1, 1))
        return p

    def _draw_children_type(
        self, num_children, parent_realiz=None, pi=None, y_separately=False
    ):
        """ Decide on which type of fracture is child is.

        The probabilities are proportional to the number of different fracture
        types in the original child (this object).

        Parameters:
            num_children: Number of fractures to generate
            parent_realiz (optional, defaults to None): Parent fracture set for this
                realization. Currently not used.
            pi (optional, int): Index of the current parent in this realization.
                Currently not used.

        Returns:
            np.array, boolean, length num_children: True for fractures that are
                to be isolated.
            np.array, boolean, length num_children: True for fractures that will
                have one T-node.
            np.array, boolean, length num_children: True for fractures that will
                have two T-nodes.

            Together, the return arrays should sum to the unit vector, that is,
            all fractures should be of one of the types.

        """
        rands = np.random.rand(num_children)
        is_isolated = rands < self.fraction_isolated
        rands -= self.fraction_isolated

        is_one_y = np.logical_and(
            np.logical_not(is_isolated), rands < self.fraction_one_y
        )

        is_both_y = np.logical_not(np.logical_or(is_isolated, is_one_y))

        if np.any(np.add.reduce((is_isolated, is_one_y, is_both_y)) != 1):
            # If we end up here, it is most likely a sign that the fractions
            # of different fracture types in the original set (this object)
            # do not sum to unity.
            raise ValueError("All fractures should be I, T or double T")

        if y_separately:
            return is_isolated, is_one_y, is_both_y
        else:
            return is_isolated, np.logical_or(is_one_y, is_both_y)

    def _generate_y_fractures(self, start, length_distribution=None):
        """ Generate fractures that originates in a parent fracture.

        Parameters:
            start (np.array, 2 x num_frac): Start point of the fractures. Will
                typically be located at a parent fracture.
            distribution (optional): Statistical distribution of fracture length.
                Used to define fracture length. If not provided, the attribute
                self.dist_length will be used.

        Returns:
            np.array (2 x 2*num_frac): Points that describe the generated fractures.
                The first num_frac points will be identical to start.
            np.array (2 x num_frac): Connection between the points. The first
                row correspond to start points, as provided in the input.

        """

        if length_distribution is None:
            length_distribution = self.dist_length

        if start.size == 0:
            return np.empty((2, 0)), np.empty((2, 0))

        if start.ndim == 1:
            start = start.reshape((-1, 1))

        num_children = start.shape[1]

        # Assign equal probability that the points are on each side of the parent
        side = frac_gen.generate_from_distribution(num_children, self.dist_side)

        child_angle = frac_gen.generate_from_distribution(num_children, self.dist_angle)
        child_length = frac_gen.generate_from_distribution(
            num_children, length_distribution
        )

        # Vector from the parent line to the new center points
        vec = np.vstack((np.cos(child_angle), np.sin(child_angle))) * child_length

        end = start + side * vec

        p = np.hstack((start, end))
        edges = np.vstack(
            (np.arange(num_children), num_children + np.arange(num_children))
        )

        return p, edges

    def _generate_constrained_fractures(
        self, start, parent_realiz, constraints=None, y_separately=False
    ):
        """
        """

        # Eventual return array for points
        p_found = np.empty((2, 0))

        # Special treatment if no fractures are generated
        if start.size == 0:
            # Create empty field for edges
            return p_found, np.empty((2, 0))

        if constraints is None:
            constraints = parent_realiz

        if y_separately:
            # Create fractures with the maximum allowed length for this distribution.
            # For this, we can use the function generate_y_fractures
            # The fractures will not be generated unless they cross a constraining
            # fracture, and the length will be adjusted accordingly
            p, edges = self._generate_y_fractures(
                start, self.dist_max_constrained_length
            )
        else:
            # Create the fracture with the standard length distribution
            p, edges = self._generate_y_fractures(start)

        num_children = edges.shape[1]

        start_parent, end_parent = parent_realiz.get_points()

        for ci in range(num_children):
            start = p[:, edges[0, ci]].reshape((-1, 1))
            end = p[:, edges[1, ci]].reshape((-1, 1))
            d, cp, cg_seg = pp.cg.dist_segment_segment_set(
                start, end, start_parent, end_parent
            )

            hit = np.where(d < self.tol)[0]
            if hit.size == 0:
                raise ValueError(
                    "Doubly constrained fractures should be constrained at its start point"
                )
            elif hit.size == 1:
                if y_separately:
                    # The child failed to hit anything - this will not generate a
                    # constrained fracture
                    continue
                else:
                    # Attach this child as a singly-connected fracture
                    p_found = np.hstack((p_found, start, end))

            else:
                # The fracture has hit a constraint
                # Compute distance from all closest points to the start
                dist_start = np.sqrt(np.sum((start - cp[:, hit]) ** 2, axis=0))
                # Find the first point along the line, away from the start
                first_constraint = np.argsort(dist_start)[1]
                p_found = np.hstack(
                    (p_found, start, cp[:, hit[first_constraint]].reshape((-1, 1)))
                )

        # Finally define the edges, based on the fractures being ordered by
        # point pairs
        num_frac = p_found.shape[1] / 2
        e_found = np.vstack((2 * np.arange(num_frac), 1 + 2 * np.arange(num_frac)))

        return p_found, e_found

    def _identify_parent_pairs(self, parents, angle):
        """ Identify pairs of fractures that lie in direct sight of each other
        along a specified angle.

        The pairs are found by tracing rays from the end points of fractures,
        and look for intersections with other fractures. In the example below,
        all three (horizontal) left fractures find each other, while the
        right fractures hits nothing.
            ___________________
               /  /     /     /
              /  /_____/     /
             /  /     /     /      __________
            /__/_____/_____/___

        Parameters:
            parents (FractureSet): Fracture set for which we look for pairs.
            angle (double, radians): Angle of search direction

        Returns:
            np.array, 2 X num_pairs: Indices of the edges forming unique pairs.
                Sorted along each column. The columns are ordered so that
            arr[0] is non-decreasing.

        """
        # Data structure for storage
        parent_pairs = []
        # We are interested in any intersection in the direction of the specified angle.
        # Create a vector with the right direction, and length equal to the maximum
        # size of the domain.
        _, _, dx, dy = self._decompose_domain()
        length = np.maximum(dx, dy)
        vec = np.vstack((np.cos(angle), np.sin(angle))) * length

        # Start and endpoints of the parents.
        start_parent, end_parent = parents.get_points()

        # Loop over all fractures, look for pairs that involves this fracture.
        # We may find the same pair twice, once for each member of the pair.
        # Uniqueness is enforced below
        for fi in range(parents.edges.shape[1]):
            # Start and end_points of this fracture
            start, end = parents.get_points(fi)

            # The start point of the segments are twice the start, twice the end
            # of this fracture.
            offshots_start = np.hstack((start, start, end, end))

            # From the nodes of this fracture, shoot segments along the vector on
            # both sides of the fracture.
            start_pos = start + vec
            start_neg = start - vec
            end_pos = end + vec
            end_neg = end - vec
            # End points of the shooting segments
            offshots_end = np.hstack((start_pos, start_neg, end_pos, end_neg))
            # Loop over all off-shots, see if they hit other fractures in the set.
            for oi in range(offshots_start.shape[1]):
                # Start and endpoint of the offshot
                s = offshots_start[:, oi].reshape((-1, 1))
                e = offshots_end[:, oi].reshape((-1, 1))
                # Compute distance between this point and all other segments in the network
                d, cp, cg_seg = pp.cg.dist_segment_segment_set(
                    s, e, start_parent, end_parent
                )
                # Count hits, where the distance is very small
                hit = np.where(d < self.tol)[0]
                if hit.size == 0:
                    # There should be at least one hit, namely the start and end point
                    # of fi
                    raise ValueError("Error when finding pairs of fractures")
                elif hit.size == 1:
                    # The offshot did not hit anything. We can move on
                    continue
                else:
                    # The offshot has hit at least one fracture.
                    # Compute distance from all closest points to the start
                    dist_start = np.sqrt(np.sum((s - cp[:, hit]) ** 2, axis=0))
                    # Find the first point along the line, away from the start
                    first_constraint = np.argsort(dist_start)[1]
                    parent_pairs.append((fi, first_constraint))

        pair_array = np.array([p.T for p in parent_pairs])

        # Uniquify and sort the output array
        pair_array.sort(axis=0)
        pair_array = pp.utils.setmembership.unique_columns_tol(pair_array)

        sort_ind = np.argsort(pair_array[0])
        pair_array = pair_array[:, sort_ind]

        return pair_array

    def _fit_num_children_distribution(self):
        """ Construct a Poisson distribution for the number of children per
        parent.

        Right now, it is not clear which data this should account for.

        The number of children should also account for the length of the
        parent fractures.
        """

        # Compute the intensity (dimensionless number of children)
        intensity = np.hstack(
            (
                self.isolated_stats["density"]
                + self.one_y_stats["density"]
                + self.both_y_stats["density"]
            )
        ).astype(np.int)

        # For some reason, it seems scipy does not do parameter-fitting for
        # abstracting a set of data into a Poisson-distribution.
        # The below recipe is taken from
        #
        # https://stackoverflow.com/questions/25828184/fitting-to-poisson-histogram

        # Hand coded Poisson pdf
        def poisson(k, lamb):
            """poisson pdf, parameter lamb is the fit parameter"""
            return (lamb ** k / scipy.special.factorial(k)) * np.exp(-lamb)

        def negLogLikelihood(params, data):
            """ the negative log-Likelohood-Function"""
            lnl = -np.sum(np.log(poisson(data, params[0]) + 1e-5))
            return lnl

        # Use maximum likelihood fit. Use scipy optimize to find the best parameter
        result = scipy.optimize.minimize(
            negLogLikelihood,  # function to minimize
            x0=np.ones(1),  # start value
            args=(intensity,),  # additional arguments for function
            method="Powell",  # minimization method, see docs
        )
        ### End of code from stackoverflow

        # Define a Poisson distribution with the computed density function
        self.dist_num_children = stats.poisson(result.x)

    def fit_distributions(self, **kwargs):
        """ Compute statistical
        """

        # NOTE: Isolated nodes for the moment does not rule out that the child
        # intersects with a parent

        num_parents = self.parent.edges.shape[1]

        # Angle and length distribution as usual
        self.fit_angle_distribution(**kwargs)
        self.fit_length_distribution(**kwargs)

        node_types_self = fracture_network_analysis.analyze_intersections_of_sets(
            self, **kwargs
        )
        node_types_combined_self, node_types_combined_parent = fracture_network_analysis.analyze_intersections_of_sets(
            self, self.parent, **kwargs
        )

        # Find the number of Y-nodes that terminates in a parent node
        y_nodes_in_parent = (
            node_types_combined_self["y_nodes"] - node_types_self["y_nodes"]
        )

        # Fractures that ends in a parent fracture on both sides. If this is
        # a high number (whatever that means) relative to the total number of
        # fractures in this set, we may be better off by describing the set as
        # constrained
        both_y = np.where(y_nodes_in_parent == 2)[0]

        one_y = np.where(y_nodes_in_parent == 1)[0]

        isolated = np.where(node_types_combined_self["i_nodes"] == 2)[0]

        num_children = self.edges.shape[1]
        self.fraction_both_y = both_y.size / num_children
        self.fraction_one_y = one_y.size / num_children
        self.fraction_isolated = isolated.size / num_children

        self.isolated = isolated
        self.one_y = one_y
        self.both_y = both_y

        # Find the number of isolated fractures that cross a parent fracture.
        # Not sure how we will use this.
        # Temporarily disable this part until it has a purpose
        # x_nodes_with_parent = (
        #    node_types_combined_self["x_nodes"] - node_types_self["x_nodes"]
        # )
        # intersections_of_isolated_nodes = x_nodes_with_parent[isolated]

        # Start and end points of the parent fractures

        # Treat isolated nodes
        if isolated.size > 0:
            density, center_distance = self._compute_line_density_isolated_nodes(
                isolated
            )
            self.isolated_stats = {
                "density": density / self.parent.length(),
                "center_distance": center_distance,
            }
            num_parents_with_isolated = np.sum(density > 0)
        else:
            num_parents_with_isolated = 0
            # The density is zero for all parent fratures.
            # Center-distance observations are empty.
            self.isolated_stats = {
                "density": np.zeros(num_parents),
                "center_distance": np.empty(0),
            }

        self._fit_dist_from_parent_distribution()

        ## fractures that have one Y-intersection with a parent
        # First, identify the parent-child relation
        if one_y.size > 0:
            density = self._compute_line_density_one_y_node(one_y)
            self.one_y_stats = {"density": density / self.parent.length()}
        else:
            # The density is zero for all parent fractures
            self.one_y_stats = {"density": np.zeros(num_parents)}

        if both_y.size > 0:
            # For the moment, we use the same computation as for one_y nodes
            # This will count all fractures twice. We try to compencate by
            # dividing the density by two, in effect saying that the fracture
            # has probability 0.5 to start from the fracture.
            # Hopefully that should not introduce bias.
            density = self._compute_line_density_one_y_node(both_y)
            self.both_y_stats = {"density": 0.5 * density / self.parent.length()}
        else:
            # The density is zero for all parent fractures
            self.both_y_stats = {"density": np.zeros(num_parents)}

        self._fit_num_children_distribution()

    def _compute_line_density_one_y_node(self, one_y):
        num_one_y = one_y.size

        start_parent, end_parent = self.parent.get_points()

        start_y, end_y = self.get_points(one_y)

        # Compute the distance from the start and end point of the children
        # to all parents
        # dist_start will here have dimensions num_children x num_parents
        # closest_pt_start has dimensions num_children x num_parents x dim (2)
        # Dimensions for end-related fields are the same
        dist_start, closest_pt_start = pp.cg.dist_points_segments(
            start_y, start_parent, end_parent
        )
        dist_end, closest_pt_end = pp.cg.dist_points_segments(
            end_y, start_parent, end_parent
        )

        # For each child, identify which parent is the closest, and consider
        # only that distance and point
        closest_parent_start = np.argmin(dist_start, axis=1)
        dist_start = dist_start[np.arange(num_one_y), closest_parent_start]
        closest_pt_start = closest_pt_start[
            np.arange(num_one_y), closest_parent_start, :
        ].T
        # Then the end points
        closest_parent_end = np.argmin(dist_end, axis=1)
        dist_end = dist_end[np.arange(num_one_y), closest_parent_end]
        closest_pt_end = closest_pt_end[np.arange(num_one_y), closest_parent_end, :].T

        # At least one of the children end point should be on a parent.
        # The tolerance used here is arbitrary.
        assert np.all(np.logical_or(dist_start < 1e-4, dist_end < 1e-4))

        start_closest = dist_start < dist_end

        num_parent = self.parent.num_frac
        num_occ_all = np.zeros(num_parent, dtype=np.object)

        # Loop over all parents,
        for fi in range(num_parent):
            hit_start = np.logical_and(start_closest, closest_parent_start == fi)
            start_point_loc = closest_pt_start[:, hit_start]
            hit_end = np.logical_and(
                np.logical_not(start_closest), closest_parent_end == fi
            )
            end_point_loc = closest_pt_end[:, hit_end]
            p_loc = np.hstack((start_point_loc, end_point_loc))
            # Compute the number of points along the line.
            # Since we only ask for a single bin in the computation (nx=1),
            # we know there will be a single return value
            num_occ_all[fi] = self.compute_density_along_line(
                p_loc, start_parent[:, fi], end_parent[:, fi], nx=1
            )[0]

        return num_occ_all

    def snap(self, threshold):
        """ Modify point definition so that short branches are removed, and
        almost intersecting fractures become intersecting.

        Parameters:
            threshold (double): Threshold for geometric modifications. Points and
                segments closer than the threshold may be modified.

        Returns:
            FractureSet: A new ChildFractureSet with modified point coordinates.
        """
        # This function overwrites FractureSet.snap(), to ensure that the
        # returned fracture set also has a parent

        # We will not modify the original fractures
        p = self.pts.copy()
        e = self.edges.copy()

        # Prolong
        p = pp.cg.snap_points_to_segments(p, e, threshold)

        return ChildFractureSet(p, e, self.domain, self.parent)
