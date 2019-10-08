"""
This module contains classes and methods aimed at generation of stochastic fracture
networks, accounting for the topology of the fracture network intersections.

The module is implemented as an extension of PorePy, which should be available on
the system to run the simulation.
"""
import numpy as np
import scipy
import scipy.stats as stats
import logging, warnings

from fracture_generation import fracture_network_analysis, distributions
import porepy as pp


logger = logging.getLogger(__name__)


class StochasticFractureNetwork2d(pp.FractureNetwork2d):
    """ Mother class for the stochastic fracture network generators.

    The class is implemented as an extension of the standard class for representation of
    2d fracture networks in PorePy.

    The main intended usage is to fit statistical distributions to the fractures,
    and use this to generate realizations based on this statistics. The statistical
    properties of the fracture set is characterized in terms of fracture position,
    length and angle.

    It is assumed that the fractures can meaningfully be represented by a single
    statistical distribution. To achieve this, it may be necessary to divide a
    fracture network into several sets, and fit them separately. As an example,
    a network where the fractures have one out of two orientations which are orthogonal
    to each other will not be meaningfully be represented as a single set.

    Attributes:
        pts (np.array, 2 x num_pts): Start and endpoints of the fractures. Points
            can be shared by fractures.
        edges (np.array, (2 + num_tags) x num_fracs): The first two rows represent
            indices, refering to pts, of the start and end points of the fractures.
            Additional rows are optional tags of the fractures.
        domain (dictionary): The domain in which the fracture set is defined.
            Should contain keys 'xmin', 'xmax', 'ymin', 'ymax', each of which
            maps to a double giving the range of the domain. The fractures need
            not lay inside the domain.
        num_frac (int): Number of fractures in the domain.

    """

    def __init__(self, pts=None, edges=None, domain=None, network=None, tol=1e-8):
        """ Define the frature set.

        Parameters:
            pts (np.array, 2 x n): Start and endpoints of the fractures. Points
            can be shared by fractures.
        edges (np.array, (2 + num_tags) x num_fracs): The first two rows represent
            indices, refering to pts, of the start and end points of the fractures.
            Additional rows are optional tags of the fractures.
        domain (dictionary): The domain in which the fracture set is defined.
            Should contain keys 'xmin', 'xmax', 'ymin', 'ymax', each of which
            maps to a double giving the range of the domain.

        """
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
        """ Add a fracture to the network.

        Parameters:
            p0 (np.arary, 2 x 1): Start point of the fracture.
            p1 (np.arary, 2 x 1): End point of the fracture.

        """

        num_pts = self.pts.shape[1]
        # Add points to the end of the point array
        pt_arr = np.hstack((p0.reshape((-1, 1)), p1.reshape((-1, 1))))
        self.pts = np.hstack((self.pts, pt_arr))
        # Define the new connection between points.
        e = np.array([[num_pts], [num_pts + 1], [tag]], dtype=np.int)
        self.edges = np.hstack((self.edges, e))
        # Append a branch
        self.branches.append(pt_arr)
        # Increase the number of fractures.
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
        """ Constrain the fracture network to lay within a specified domain.

        Fractures that cross the boundary of the domain will be cut to lay
        within the boundary. Fractures that lay completely outside the domain
        will be dropped from the constrained description.

        TODO: Also return an index map from new to old fractures.

        Parameters:
            domain (dictionary, None): Domain specification, in the form of a
                dictionary with fields 'xmin', 'xmax', 'ymin', 'ymax'. If not
                provided, the domain of this object will be used.

        Returns:
            StochasticFractureNetwork2d: Initialized by the constrained fractures, and the
                specified domain.

        """

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

        dist, *rest = pp.distances.segment_segment_set(
            p_new[:, 0], p_new[:, 1], start_set, end_set
        )

        allowed_dist = data.get("minimum_fracture_spacing", 0)
        return dist.min() < allowed_dist

    def _generate_by_intensity(self, domain):
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
        """ Generate a fracture network from a specified criterion.

        The available criteria are:
            'counting': Generate a fixed number of fractures.
            'intensity': Generate according to the intensity map of this generator.
            'length': Generate fractures until a target length for the total
                generated length is reached.

        Parameters:
            criterion (str): Method used for generation.
            data (dict): Various parameters needed for generation.
            return_network (boolean, optional): If True (default), the generated
                data is returned in the form of a fracture netowrk. If not, points
                and edges are returned instead.

        Returns:
            Either a StochasticFractureNetwork2d (if return_network=True), or
                points and edges of the generated network.

        """
        if "domain" not in data.keys():
            data["domain"] = self.domain

        if criterion.lower().strip() == "counting":
            # Generate a fixed number of fractures
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
            # Generate fractures until a given target length for the total fracture
            # network is hit.

            # Utility function
            def dist_points(a, b):
                return np.sqrt(np.sum((a - b) ** 2))

            full_length = 0
            p = np.zeros((2, 0))
            # Until the target length is reached, create new fractures, add them
            # to the set (unless they are too close to existing fractures)
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
        """ Generate a single fracture, according to the statistics of this generator.

        The fracture position is set according to the specified intensity map of
        the generator - or placed in a random point if no intensity map is set.

        """
        # If no intensity map is set, use a single box for the entire domain
        if self.intensity is None:
            self.intensity = np.array([[10 / self.domain_measure()]])

        # Generate center point for the fracture
        # The while loop goes on until at least one center point is created.
        while True:
            cp_tmp = self._generate_centers(center_mode="poisson", data=data)
            if cp_tmp.size > 0:
                break

        # If the intensity map has several blocks, there will be a spatial
        # ordering associated with cp. Since we want a single fracture, we randomly
        # shuffle the center point coordinates.
        shuffle_ind = np.argsort(np.random.rand(cp_tmp.shape[-1]))
        cp = cp_tmp[:, shuffle_ind][:, 0].reshape((-1, 1))
        # Set orientation and length according to the set distributions
        orientation = self._generate_from_distribution(1, self.dist_orientation)
        lengths = self._generate_from_distribution(1, self.dist_length)
        return self._fracture_from_center_angle_length(cp, orientation, lengths)[0]

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


class ConstrainedChildrenGenerator(StochasticFractureGenerator):
    """ Generator of fractures that have a T-connection with a specified parent set.

    The generator produce fractures with one T-conneection; no conditions are
    put on the non-anchored endpoint. Also, the generated fracture may cross
    other fractures in the parent set.

    The intended use is as a second step in a sequential construction of the network,
    that is, the parent set must be available by the time of construction of the
    children generator.

    Attributes:
        parent (StochasticFractureNetwork2d): Parent network.

    """

    def __init__(self, parent, side_distribution=None, **kwargs):
        """ Initialize the generator.

        Parameters:
            parent (StochasticFractureNetwork2d: Parent fracture set.
            side_distribution (distribution): Probability distribution for whether
                the fracture will shoot to the left or right of the parent. Currently not
                well explored, leave as None (gives 50-50 chance of left or right),
                or use with care.
            kwargs: Length, orientation distributions. See StochasticFractureGenerator
                for more information.

        """
        super(ConstrainedChildrenGenerator, self).__init__(**kwargs)

        self.parent = parent

        if side_distribution is None:
            self.dist_side = distributions.Signum()
        else:
            warnings.warn("The side distribution option must be used with extreme care")
            self.dist_side = side_distribution

    def generate(self, pi=None, data=None):
        """ Generate a fracture that has one node on a parent fracture.

        The fracture is constructed as a ray from the parent point, with length and
        orientation drawn according to the specified statistical distributions.

        The generated fracture may cross other fractures, parents or children.

        Parameters:
            pi (int, optional): Index of the parent fracture. If not provided,
                a parent will be randomly picked.
            data (dict): Data used in generation process. Currently not active.

        Returns:
            np.array (2, 2): Start and end coordinates of the generated fracture.
                First column is start, second is end.

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
        """ Pick a parent among the parent set.

        The probability of picking a given parent fracture scales with the parents'
        length

        Returns:
            int: Index of the chosen parent

        """
        # Pick a parent, with probabilities scaling with parent length
        cum_length = self.parent.length().cumsum()
        cum_length /= cum_length[-1]

        r = np.random.rand(1)
        return np.nonzero(r < cum_length)[0][0]


class DoublyConstrainedChildrenGenerator(StochasticFractureGenerator):
    """ Generator of fractures that have T-connections in both ends.

    The generator produce fractures with T-conneections relative to a specified
    parent network in both ends. The generated fracture may cross other fractures
    in the parent set.

    The intended use is as a second step in a sequential construction of the network,
    that is, the parent set must be available by the time of construction of the
    children generator.

    Attributes:
        parent (StochasticFractureNetwork2d): Parent network.

    """

    def __init__(self, parent, side_distribution=None, data=None, **kwargs):
        super(DoublyConstrainedChildrenGenerator, self).__init__(**kwargs)

        self.search_direction = self._get_search_vector()

        self.parent = parent

        if side_distribution is None:
            self.dist_side = distributions.Signum()

        # Find pairs of neighboring fracutres, and the endpoints of their
        # overlapping segments.
        pair_array, point_first, point_second = self._trace_rays_from_fracture_tips()
        # Process the information.
        self._pairs_of_parents(pair_array, point_first, point_second)

    def generate(self, pi=None, data=None):
        """ Generate a fracture that has one node on a parent fracture.

        The fracture is constructed as a ray from the parent point, with length and
        orientation drawn according to the specified statistical distributions.

        The generated fracture may cross other fractures, parents or children.

        Parameters:
            pi (int, optional): Index of the parent fracture. If not provided,
                a parent will be randomly picked.
            data (dict): Data used in generation process. Currently not active.

        Returns:
            np.array (2, 2): Start and end coordinates of the generated fracture.
                First column is start, second is end.

        """

        if pi is None:
            pi = self._pick_parent_pair()

        fi = self.pairs[0, pi]
        si = self.pairs[1, pi]

        # Assign equal probability that the points are on each side of the parent
        side = self._generate_from_distribution(1, self.dist_side)

        vec = self.search_direction

        # Start and endpoint of the interval where the parent pairs are neighbors
        start = self.interval_points[pi][:, 0]
        end = self.interval_points[pi][:, 1]

        along_parent = end - start
        # The start of the new fracture is random along the interval
        child_start = (start + np.random.rand(1) * along_parent).reshape((-1, 1))

        # Find the endpoint as the intersection with the second fracture
        # By the definition of the fracture pairs, we know we can find the intersection
        # by tracing a ray from the start point
        _, _, _, child_end, *rest = self._find_neighbors(vec, child_start)

        # If there is more than one intersection point (both positive and negative side)
        # pick the right one.
        # If there is only one, this will be hte intersection
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
        """ Convert the pairs and intersection points into a format to be used
        in the generation.

        This
        """
        # Find parents that are visible to each other along a ray of fixed orientation.

        # Data storage structures Parents
        pairs = []
        interval_first = {}
        interval_second = {}

        tol = 1e-4

        # Special case of no neighboring fractures. It is not really clear how
        # generation with this will function, so raise a warning.
        if len(pair_array) == 0:
            raise ValueError("Found no neighboring parents. Cannot generate children.")
            # self.parent_pairs = pairs
            # self.interval_first = interval_first
            # self.interval_second = interval_second
            # return

        def dist_pts(a, b):
            if a.shape[0] < 2:
                a = a.reshape((-1, 1))
            if b.shape[0] < 2:
                b = b.reshape((-1, 1))

            return np.sqrt(np.sum((a - b) ** 2))

        # Loop over all fractures
        for fi in range(self.parent.num_frac):

            # Find all preliminary pairs where the current fracture is part
            hit_first = np.where(pair_array[0] == fi)[0]
            hit_second = np.where(pair_array[1] == fi)[0]

            # Start and end points of this fracture
            start, end = self.parent.get_points(fi)

            # Set up a point set along the fracture consistng of end points, togehter
            # with all points where rays from other fractures hit.
            # Together, these will form the boundaries of the intervals where the
            # fracture has a certain neighbor (along the search vector)
            isect_pt = np.hstack(
                (start, end, point_first[:, hit_first], point_second[:, hit_second])
            )
            # Sort points along the fracture.
            # First ensure unique points
            isect_pt, *rest = pp.utils.setmembership.unique_columns_tol(isect_pt)
            # Compute distances from the start point, and sort
            dist = np.sum((isect_pt - start) ** 2, axis=0)
            p = isect_pt[:, np.argsort(dist)]

            # Storage of what is the current neighbor on the two sides of the fracture
            # Will be either None (no neighbor found), or a pair of indices
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
                    if (
                        loc_second_neigh is not None
                    ):  # None signifies there are no neighbors here
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
                            # Note the start points of the neighbor segments on
                            # this and the neighboring fracture
                            pos_pt_self = pself
                            pos_pt_other = pf
                            # Register the new active pair
                            active_pos_pair = (fi, loc_neigh)

                        else:
                            # Check if the point on the first fracture is indeed an endpoint,
                            # or if the reason we shoot from pself is that we were hit
                            # from the other side
                            p0, p1 = self.parent.get_points(loc_neigh)
                            if dist_pts(p0, pf) > tol and dist_pts(p1, pf) > tol:
                                continue

                            # We found (this fracture was hit by) the end of a neighbor.
                            # The next neighbor is the second closest to the hit point in
                            # the direction of vec

                            # Register the new interval on this fracture.
                            # The interval is defined by pos_pt_self (found when
                            # this interval started) and the current hitpoint
                            # on this farcture
                            if active_pos_pair in interval_first.keys():
                                tmp = interval_first[active_pos_pair]
                                tmp.append(np.hstack((pos_pt_self, pself)))
                                interval_first[active_pos_pair] = tmp
                            else:
                                interval_first[active_pos_pair] = [
                                    np.hstack((pos_pt_self, pself))
                                ]

                            # Next, process the neighbor
                            # If the ray we were hit by stems from the current active
                            # neighbor, the interval of overlap ends.
                            if active_neigh_pos == loc_neigh:
                                # Register the interval on the neighboring fracture
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
                                # A new fracture started between the main one and
                                # what was up to now the closest neighbor.
                                active_neigh_pos = loc_neigh
                                # An interval has ended for the secondary fracture
                                if active_pos_pair in interval_second.keys():
                                    tmp = interval_second[active_pos_pair]
                                    tmp.append(np.hstack((pos_pt_other, psec)))
                                    interval_second[active_pos_pair] = tmp
                                else:
                                    interval_second[active_pos_pair] = [
                                        np.hstack((pos_pt_other, psec))
                                    ]

                            # End the current interval by registering the new pair
                            # and adding points, but only if we have not reached the
                            # end of the main fracture, or there is no neighbor.
                            if pi < (p.shape[1] - 1) and active_neigh_pos is not None:
                                pairs.append((fi, active_neigh_pos))
                                pos_pt_self = pself
                                if active_neigh_pos == loc_neigh:
                                    pos_pt_other = pf
                                else:
                                    pos_pt_other = psec
                                active_pos_pair = (fi, active_neigh_pos)

                    else:
                        # This is the negative side of the fracture; the algorithm is
                        # identical to the one commented above.
                        # Implementation note: It should be possible to unify the
                        # positive and negative sides instead of copying code, but
                        # this has not been prioritized

                        # Check if there currently is a neighbor on this side
                        if active_neigh_neg is None:
                            # We have found the start of a new neighbor
                            active_neigh_neg = loc_neigh
                            pairs.append((fi, loc_neigh))
                            neg_pt_self = pself
                            neg_pt_other = pf
                            active_neg_pair = (fi, loc_neigh)

                        else:
                            # Check if the point on the first fracture is indeed an endpoint,
                            # or if the reason we shoot from pself is that we were hit
                            # from the other side
                            p0, p1 = self.parent.get_points(loc_neigh)
                            if dist_pts(p0, pf) > tol and dist_pts(p1, pf) > tol:
                                continue

                            # We found (this fracture was hit by) the end of a neighbor.
                            # The next neighbor is the second closest to the hit point in
                            # the direction of vec.

                            # Add the interval to the pairing of the main and first fracture
                            # The pairing is either non-empty (another interval has been
                            # found before), or we need to make a new dictionary item
                            if active_neg_pair in interval_first.keys():
                                tmp = interval_first[active_neg_pair]
                                tmp.append(np.hstack((neg_pt_self, pself)))
                                interval_first[active_neg_pair] = tmp
                            else:
                                interval_first[active_neg_pair] = [
                                    np.hstack((neg_pt_self, pself))
                                ]

                            # If the nearest neighbor is the one that shot the ray we were
                            # hit by, the current interval is ending
                            if active_neigh_neg == loc_neigh:
                                # The next active neighbor is the second one. This may be
                                # None, which signifies that no neighbor exists
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
        # fratures, with a smaller inbetween at the middle of the larger ones).
        # Process the data to obtain a unique set of pairs

        for fi in range(pairs.shape[1]):
            f = pairs[0, fi]
            s = pairs[1, fi]

            # If the first row contains the fracture with the lower index,
            # we move in to delete other instances of this pair, including those
            # that have the higher index in the first row
            if f < s:
                try:
                    # Remove the interval information from
                    interval_first.pop((s, f))
                except KeyError:
                    # The keys have already been removed
                    continue

        points = []
        lengths = []
        parent_pair = []

        for k, v in interval_first.items():
            for p in v:
                # Length of interval
                l = np.sqrt(np.sum((p[:, 1] - p[:, 0]) ** 2))
                lengths.append(l)
                # Points
                points.append(p)
                parent_pair.append(k)

        # Set of pairs of neighboring fracutres in the parent set.
        # Need not be unique (think long parallel
        # fratures, with a smaller inbetween at the middle of the larger ones).
        # The first row will be strictly set of the second row.
        self.pairs = np.array([p for p in parent_pair]).T
        # For each pair, the length of their interval
        self.interval_lengths = np.cumsum(lengths)
        # Endpoints of the overlapping interval, on the first of the paired fracutres.
        self.interval_points = points

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

        Returns:
            np.array, 2 X num_pairs: Pairs of indices of fractures that form
                neighboring segments.
            np.array, 2 x num_pairs: Coordinate of an endpoint of the overlapping
                segment on the first fracture in the neighbor pair.
            np.array, 2 x num_pairs: Coordinate of an endpoint of the overlapping
                segment on the second fracture in the neighbor pair.

        """

        """ TECHNICAL COMMENT, TO BE PRESERVED FOR NOW
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
        # Uniqueness is enforced afterwards
        for fi in range(self.parent.num_frac):
            # Start and end_points of this fracture
            start, end = self.parent.get_points(fi)

            # Find the neighbors
            # loc_pairs is the index of the neighboring fracture,
            # loc_isect_first is the intersection point on the main fracture
            # loc_isect_sec is the intersection on the neighbor
            loc_pairs, _, loc_isect_first, loc_isect_sec, *rest = self._find_neighbors(
                vec, start, end
            )
            # The ray hit nothing
            if len(loc_pairs) == 0:
                continue
            # We may get up to four hits: One on each side for start and endpoint of the
            # main fracture.
            # Store intersection points
            point_first = np.hstack((point_first, loc_isect_first))
            point_second = np.hstack((point_second, loc_isect_sec))
            # Register the new pairing between the main fracture and its neighbor
            for pi in range(len(loc_pairs)):
                parent_pairs.append((fi, loc_pairs[pi]))

        pair_array = np.array([p for p in parent_pairs]).T

        return pair_array, point_first, point_second

    def _get_search_vector(self):
        """ Obtain a search vector with direction similar to the mean orientation
        of the network, and length equal to the size of the domain.

        Returns:
            np.array (2x1): Vector

        """
        # We are interested in any intersection in the direction of the specified angle.
        # Create a vector with the right direction, and length equal to the maximum
        # size of the domain.
        _, _, dx, dy = self._decompose_domain()
        length = dx ** 2 + dy ** 2

        angle = self._generate_from_distribution(1000, self.dist_orientation).mean()
        vec = np.vstack((np.cos(angle), np.sin(angle))) * length

        return vec

    def _find_neighbors(self, vec, start, end=None):
        """ For a given point or segment (on a parent fracture), find neigbors
        among the parents, by searching in a fixed direction.

        Parameters:
            vec (np.array, 2x1): Search direction.
            start (np.array, 2x1): Start point of the segment.
            end (np.array, 2x1, optional): End point of a segment.

        Returns:
            list: Earch element is the index of neighboring fracture, or None if not found.
            list: Earch element is the  index of second neighboring fracture (behind the
                first), or None if not found.
            np.array: Point of intersection on the base segment. Will be start or end.
            np.array: Point of intersection on the neighboring fracutre.
            np.array: Point of intersection on the second neighboring fracture.
            list of boolean: If True, the search was in the positive direction of
                the search vector.

            All returned quantities may have up to 2 or 4 (if end is not None) fields,
            corresponding to search from start in positive, then negative direction,
            and optionally search from end in positive and negative direction.

            The intersection point arrays of the second neighbor may contain None
            if no secondary neighbors were found.

        """
        # Terminology in the comments below:
        # The start and end points are the *base points*, they form the
        # *base segment*.
        # The rays shooting from the base points in the specified seacrh
        # directions are the *search rays*.
        # (the search vector is so long that it is in practice a ray in this context).

        # Start and endpoints of the parent fractures.
        start_parent, end_parent = self.parent.get_points()

        # Generate the search rays, specified from the start at the base points
        # along the search vector
        if end is None:
            # No end base point.
            search_ray_start = np.hstack((start, start))
            # Search in both directions
            start_pos = start + vec
            start_neg = start - vec
            # End points of the search rays
            search_ray_end = np.hstack((start_pos, start_neg))
        else:
            # Both base points are available
            search_ray_start = np.hstack((start, start, end, end))

            # Search in both directions
            start_pos = start + vec
            start_neg = start - vec
            end_pos = end + vec
            end_neg = end - vec
            # End points of the serch rays
            search_ray_end = np.hstack((start_pos, start_neg, end_pos, end_neg))

        # Data storage stuctures
        # Neighbors found along the search direction
        first_neighbor = []
        # The second neighbor (behind the neighbor)
        second_neighbor = []
        # intersection point on the base segment. Will be one of the base points.
        isect_self = []
        # Intersection point on the first neighbor
        isect_first = []
        # intersection point on the second neighbor
        isect_second = []
        # Whether the interseciton point was found in the positive or negative
        # direction of the search vector.
        is_pos = []

        # From the nodes of this fracture, shoot segments along the vector on
        # both sides of the fracture.

        # Loop over all search rays, see if they hit other fractures in the set.
        for oi in range(search_ray_start.shape[1]):
            # Start and endpoint of the offshot
            s = search_ray_start[:, oi].reshape((-1, 1))
            e = search_ray_end[:, oi].reshape((-1, 1))
            # Compute distance between this point and all other segments in the network
            d, cp, cg_seg = pp.distances.segment_segment_set(
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
                first_neighbor.append(hit[first_constraint])
                # Store the intersection point of the first and second
                # fracture
                isect_self.append(s)
                isect_first.append(cp[:, hit[first_constraint]])

                # Store information on the secondary neighbor, if it exists
                if hit.size > 2:
                    second_constraint = np.argsort(dist_start)[2]
                    second_neighbor.append(hit[second_constraint])
                    isect_second.append(cp[:, hit[second_constraint]])
                else:
                    second_neighbor.append(None)
                    isect_second.append(None)
                # The search rays were defined by alternating between adding and
                # subtracting the search vector
                is_pos.append(oi % 2 == 0)

        # Utility function to convert a list to a numpy array, accounting for
        # various sizes of the list
        def arr_to_np(a):
            b = np.array([p for p in a]).T
            if b.ndim == 3:
                b = b.squeeze()
            if b.size == 2:
                return b.reshape((-1, 1))
            else:
                return b

        # Data conversion
        isect_self = arr_to_np(isect_self)
        isect_first = arr_to_np(isect_first)
        isect_second = arr_to_np(isect_second)

        return (
            first_neighbor,
            second_neighbor,
            isect_self,
            isect_first,
            isect_second,
            is_pos,
        )
