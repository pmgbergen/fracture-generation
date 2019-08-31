#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various utility functions for analysis of fracture networks. For the moment, 2d
only.
"""
import numpy as np
import networkx as nx
import time
import scipy.sparse.linalg as spla

import porepy as pp
import pdb
from fracture_generation import segment_pixelation


def permeability_upscaling(
    network, data, mesh_args, directions, do_viz=True, tracer_transport=False
):
    """ Compute bulk permeabilities for a 2d domain with a fracture network.

    The function sets up a flow field in the specified directions, and calculates
    an upscaled permeability of corresponding to the calculated flow configuration.

    Parameters:
        network (pp.FractureNetwork2d): Network, with domain.
        data (dictionary): Data to be specified.
        mesh_args (dictionray): Parameters for meshing. See FractureNetwork2d.mesh()
            for details.
        directions (np.array): Directions to upscale permeabilities.
            Indicated by 0 (x-direction) and or 1 (y-direction).

    Returns:
        np.array, dim directions.size: Upscaled permeability in each of the directions.
        sim_info: Various information on the time spent on this simulation.

    """

    directions = np.asarray(directions)
    upscaled_perm = np.zeros(directions.size)

    sim_info = {}

    for di, direct in enumerate(directions):

        tic = time.time()

        gb = network.mesh(tol=network.tol, mesh_args=mesh_args)
        print(gb)
        toc = time.time()
        if di == 0:
            sim_info["num_cells_2d"] = gb.grids_of_dimension(2)[0].num_cells
            sim_info["time_for_meshing"] = toc - tic

        gb = _setup_simulation_flow(gb, data, direct)

        pressure_kw = "flow"

        mpfa = pp.Mpfa(pressure_kw)
        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {"pressure": {"cells": 1}}
            d[pp.DISCRETIZATION] = {"pressure": {"diffusive": mpfa}}


        coupler = pp.RobinCoupling(pressure_kw, mpfa)
        for e, d in gb.edges():
            g1, g2 = gb.nodes_of_edge(e)
            d[pp.PRIMARY_VARIABLES] = {"mortar_darcy_flux": {"cells": 1}}
            d[pp.COUPLING_DISCRETIZATION] = {
                "lambda": {
                    g1: ("pressure", "diffusive"),
                    g2: ("pressure", "diffusive"),
                    e: ("mortar_darcy_flux", coupler),
                }
            }

        assembler = pp.Assembler(gb)
        assembler.discretize()
        # Discretize
        tic = time.time()
        A, b = assembler.assemble_matrix_rhs()
        if di == 0:
            toc = time.time()
            sim_info["Time_for_assembly"] = toc - tic
        tic = time.time()
        p = spla.spsolve(A, b)
        if di == 0:
            toc = time.time()
            sim_info["Time_for_pressure_solve"] = toc - tic

        assembler.distribute_variable(p)

        # Post processing to recover flux
        tot_pressure = 0
        tot_area = 0
        tot_inlet_flux = 0
        for g, d in gb:
            # Inlet faces of this grid
            inlet = d.get("inlet_faces", None)
            # If no inlet, no need to do anything
            if inlet is None or inlet.size == 0:
                continue

            # Compute flux field in the grid
            # Internal flux
            flux = d[pp.DISCRETIZATION_MATRICES][pressure_kw]["flux"] * d[pp.STATE]["pressure"]
            # Contribution from the boundary
            bound_flux_discr = d[pp.DISCRETIZATION_MATRICES][pressure_kw]["bound_flux"]
            flux += (
                bound_flux_discr
                * d["parameters"][pressure_kw]["bc_values"]
            )
            # Add contribution from all neighboring lower-dimensional interfaces
            for e, d_e in gb.edges_of_node(g):
                mg = d_e["mortar_grid"]
                if mg.dim == g.dim -1:
                    flux += bound_flux_discr * mg.mortar_to_master_int() * d_e[pp.STATE]["mortar_darcy_flux"]

            # Add contribution to total inlet flux
            tot_inlet_flux += flux[inlet].sum()

            # Next, find the pressure at the inlet faces
            # The pressure is calculated as the values in the neighboring cells,
            # plus an offset from the incell-variations
            pressure_cell = (
                d[pp.DISCRETIZATION_MATRICES][pressure_kw]["bound_pressure_cell"]
                * d[pp.STATE]["pressure"]
            )
            pressure_flux = (
                d[pp.DISCRETIZATION_MATRICES][pressure_kw]["bound_pressure_face"]
                * d["parameters"][pressure_kw]["bc_values"]
            )
            inlet_pressure = pressure_cell[inlet] + pressure_flux[inlet]
            # Scale the pressure at the face with the face length.
            # Also include an aperture scaling here: For the highest dimensional grid, this
            # will be as scaling with 1.
            aperture = d[pp.PARAMETERS][pressure_kw]["aperture"][0]
            tot_pressure += np.sum(inlet_pressure * g.face_areas[inlet] * aperture)
            # Add to the toal outlet area
            tot_area += g.face_areas[inlet].sum() * aperture
            # Also compute the cross sectional area of the domain
            if g.dim == gb.dim_max():
                # Extension of the domain in the active direction
                dx = g.nodes[direct].max() - g.nodes[direct].min()

        # Mean pressure at the inlet, which will also be mean pressure difference
        # over the domain
        mean_pressure = tot_pressure / tot_area

        # The upscaled permeability
        upscaled_perm[di] = -(tot_inlet_flux / tot_area) * dx / mean_pressure
        # End of post processing for permeability upscaling

        if tracer_transport:

            _setup_simulation_tracer(gb, data, direct)

            tracer_variable = "temperature"

            temperature_kw = "transport"
            mortar_variable = "lambda_adv"


            # Identifier of the two terms of the equation
            adv = "advection"

            advection_term = "advection"
            mass_term = "mass"
            advection_coupling_term = "advection_coupling"

            adv_discr = pp.Upwind(temperature_kw)

            adv_coupling = pp.UpwindCoupling(temperature_kw)

            mass_discretization = pp.MassMatrix(temperature_kw)

            for g, d in gb:
                d[pp.PRIMARY_VARIABLES] = {tracer_variable: {"cells": 1}}
                d[pp.DISCRETIZATION] = {
                    tracer_variable: {
                        advection_term: adv_discr,
                        mass_term: mass_discretization,
                    }
                }

            for e, d in gb.edges():
                g1, g2 = gb.nodes_of_edge(e)
                d[pp.PRIMARY_VARIABLES] = {mortar_variable: {"cells": 1}}
                d[pp.COUPLING_DISCRETIZATION] = {
                    advection_coupling_term: {
                        g1: (tracer_variable, adv),
                        g2: (tracer_variable, adv),
                        e: (mortar_variable, adv_coupling),
                    }
                }

            pp.fvutils.compute_darcy_flux(gb, keyword_store=temperature_kw ,
                                          lam_name="mortar_darcy_flux")

            A, b, block_dof, full_dof = assembler.assemble_matrix_rhs(
                gb,
                active_variables=[tracer_variable, mortar_variable],
                add_matrices=False,
            )

            advection_coupling_term += (
                "_" + mortar_variable + "_" + tracer_variable + "_" + tracer_variable
            )

            mass_term += "_" + tracer_variable
            advection_term += "_" + tracer_variable

            lhs = A[mass_term] + data["time_step"] * (
                A[advection_term] + A[advection_coupling_term]
            )
            rhs_source_adv = data["time_step"] * (
                b[advection_term] + b[advection_coupling_term]
            )

            IEsolver = spla.factorized(lhs)

            save_every = 10
            n_steps = int(np.round(data["t_max"] / data["time_step"]))

            # Initial condition
            tracer = np.zeros(rhs_source_adv.size)
            assembler.distribute_variable(
                gb,
                tracer,
                block_dof,
                full_dof,
                variable_names=[tracer_variable, mortar_variable],
            )

            # Exporter
            exporter = pp.Exporter(gb, name=tracer_variable, folder="viz_tmp")
            export_fields = [tracer_variable]

            for i in range(n_steps):

                if np.isclose(i % save_every, 0):
                    # Export existing solution (final export is taken care of below)
                    assembler.distribute_variable(
                        gb,
                        tracer,
                        block_dof,
                        full_dof,
                        variable_names=[tracer_variable, mortar_variable],
                    )
                    exporter.write_vtk(export_fields, time_step=int(i // save_every))
                tracer = IEsolver(A[mass_term] * tracer + rhs_source_adv)

            exporter.write_vtk(export_fields, time_step=(n_steps // save_every))
            time_steps = np.arange(
                0, data["t_max"] + data["time_step"], save_every * data["time_step"]
            )
            exporter.write_pvd(time_steps)


    return upscaled_perm, sim_info


def _setup_simulation_flow(gb, data, direction):

    min_coord = gb.bounding_box()[0][direction]
    max_coord = gb.bounding_box()[1][direction]

    dx = max_coord - min_coord

    for g, d in gb:

        if g.dim == gb.dim_max():
            kxx = np.ones(g.num_cells)
        else:
            kxx = np.ones(g.num_cells) * data["fracture_perm"]

        a = data["aperture"]
        a = np.power(a, gb.dim_max() - g.dim) * np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(kxx * a)

        specified_parameters = {"second_order_tensor": perm, "aperture": a}

        bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if bound_faces.size > 0:
            # Outflow faces
            hit_out = np.where(
                np.abs(g.face_centers[direction, bound_faces] - max_coord) < 1e-8 * dx
            )[0]
            hit_in = np.where(
                np.abs(g.face_centers[direction, bound_faces] - min_coord) < 1e-8 * dx
            )[0]
            # Dirichlet conditions on outlets only
            bound_type = np.array(["neu"] * bound_faces.size)
            bound_type[hit_out] = "dir"

            bound = pp.BoundaryCondition(g, bound_faces.ravel("F"), bound_type)

            # Default is homogeneous conditions on all boundary faces
            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces] = 0
            # The influx is proportional to the
            # Use aperture scaling for lower-dimensional faces; for max_dim
            # the aperture is set to 1.
            influx = g.face_areas[bound_faces[hit_in]] * a[0]
            bc_val[bound_faces[hit_in]] = -influx
            specified_parameters.update({"bc": bound, "bc_values": bc_val})

            # Store the inlet faces
            d["inlet_faces"] = bound_faces[hit_in]

        pp.initialize_default_data(g, d, "flow", specified_parameters)

    g_2d = gb.grids_of_dimension(2)[0]
    d_2d = gb.node_props(g_2d)
    perm = d_2d[pp.PARAMETERS]['flow']['second_order_tensor']

    for e, d in gb.edges():
        gl, _ = gb.nodes_of_edge(e)
        dl = gb.node_props(gl)
        mg = d["mortar_grid"]
        aperture = dl[pp.PARAMETERS]["flow"]["aperture"][0]
        kn = data["fracture_perm"] / aperture
        d[pp.PARAMETERS] = pp.Parameters(mg, ["flow"], [{"normal_diffusivity": kn}])
        d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

    return gb


def _setup_simulation_tracer(gb, data, direction):
    min_coord = gb.bounding_box()[0][direction]
    max_coord = gb.bounding_box()[1][direction]

    parameter_keyword = "transport"

    for g, d in gb:
        param = d[pp.PARAMETERS]
        transport_parameter_dictionary = {}
        param.update_dictionaries(parameter_keyword, transport_parameter_dictionary)
        param.set_from_other(parameter_keyword, "flow", ["aperture"])
        d[pp.DISCRETIZATION_MATRICES][parameter_keyword] = {}

        unity = np.ones(g.num_cells)

        if g.dim == gb.dim_max():
            porosity = 0.2 * unity
        else:
            porosity = 0.8 * unity

        specified_parameters = {"mass_weight": porosity,}

        bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if bound_faces.size > 0:
            hit_out = np.where(
                np.abs(g.face_centers[direction, bound_faces] - max_coord) < 1e-8
            )[0]
            hit_in = np.where(
                np.abs(g.face_centers[direction, bound_faces] - min_coord) < 1e-8
            )[0]
            bound_type = np.array(["neu"] * bound_faces.size)
            bound_type[hit_out] = "dir"
            bound_type[hit_in] = "dir"
            bound = pp.BoundaryCondition(g, bound_faces.ravel("F"), bound_type)
            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces[hit_out]] = 0
            bc_val[bound_faces[hit_in]] = 1

            specified_parameters.update({"bc": bound, "bc_values": bc_val})

            d["inlet_faces"] = bound_faces[hit_in]
        else:
            bc = pp.BoundaryCondition(g)
            specified_parameters.update({"bc": bc, "bc_values": np.zeros(g.num_faces)})

        pp.initialize_data(g, d, parameter_keyword, specified_parameters)

    for e, d in gb.edges():

        d[pp.PARAMETERS].update_dictionaries(parameter_keyword, {})
        d[pp.DISCRETIZATION_MATRICES][parameter_keyword] = {}

    return gb


def connectivity_field(network, num_boxes):
    """ Compute the connectivity field associated with a fracture network.

    The method assumes that all points of the network are contained within the
    network's domain. This can be ensured by invoking network.constrain_to_domain().

    The method is motivated by the paper
        Connectivity field: A measure for characterizing fracture networks,
        by Alghalandis et al. Mathematical Geosciences 2015.

    Parameters:
        network (FractureNetwork2d): fracture network to be analyzed
        num_boxes (np.array, size 2): Number of bins to split the domain into in
            the x and y-direction, respectively.

    Returns:
        np.array, size num_boxes: For each cell in the Cartesian division of the
            domain, the number of other cells the cell is connected to

    """

    # Partition the domain
    _, _, dx, _ = cartesian_partition(
        network.domain, num_x=num_boxes[0], num_y=num_boxes[1]
    )
    # Graph representation of the network
    graph, split_network = network.as_graph()

    num_clusters = len([sg for sg in nx.connected_components(graph)])

    # Field to store the presence of a network
    is_connected = np.zeros((num_clusters, num_boxes[0], num_boxes[1]))

    # Loop over all connected subgraphs of the network, identify connected
    # components
    for gi, sub_graph in enumerate(nx.connected_components(graph)):
        sg = graph.subgraph(sub_graph)
        loc_edges = np.array([[e[0], e[1]] for e in sg.edges()]).T
        # Use a pixelation algorithm to project fractures onto a Cartesian representation
        # of the domain
        pixelated = segment_pixelation.pixelate(
            split_network.pts, loc_edges, num_boxes, dx
        )
        is_connected[gi] = pixelated

    connectivity_field = np.zeros(num_boxes)
    for i in range(num_boxes[0]):
        for j in range(num_boxes[1]):
            hit = np.where(is_connected[:, i, j] > 0)[0]
            connectivity_field[i, j] = np.sum(is_connected[hit].sum(axis=0) > 0)

    # Binary division of connected and non-connected components
    return connectivity_field


def cartesian_partition(domain, num_x, num_y=None):
    """ Define a Cartesian partitioning of a domain.

    The domain could be 1d or 2d.

    Parameters:
        domain (dictionary): The domain in which the fracture set is defined.
            Should contain keys 'xmin', 'xmax', for 2d also 'ymin', 'ymax',
            each of which maps to a double giving the range of the domain.
        num_x (int): Number of bins in the x-direction
        num_y (int, optional): Number of bins in the y-direction. Only if the
            domain is 2d.

    Returns:
        double: Minimum x-coordinate of the domain
        double (optional): Minimum y-coordinate of the domain. Only if the domain
            is 2d.
        double: Spacing of the cells in the x-direction.
        double (optional): Spacing of the cells in the y-direction. Only if the
            domain is 2d.

    """
    x0 = domain["xmin"]
    dx = (domain["xmax"] - domain["xmin"]) / num_x

    if "ymin" in domain.keys() and "ymax" in domain.keys():
        y0 = domain["ymin"]
        dy = (domain["ymax"] - domain["ymin"]) / num_y
        return x0, y0, dx, dy
    else:
        return x0, dx


def compute_topology(networks):
    """ Compute the node and branch topology of a list of fracture network.

    The node topology is defined by counting the number of I-nodes (end nodes),
    T-nodes (where one fractures terminates in another) and X-nodes (standard
    intersection).

    The branch topology counts the number of fracture branches that are isolated
    (both endpoints are I-nodes), branches where one end is an I-node, one is
    connected (T or X), and branches where both endpoints are T or X. These types
    are denoted II, IC and CC, respectively.

    Parameters:
        networks (list of FractureSets): Each list element is a fracture network.

    Returns:
        dictionary with keys 'i', 'y' and 'x'. The values are lists with length
            len(networks), counting the number of i, y and x-nodes for each network.
        dictionary with keys 'ii', 'ic' and 'cc'. The values are lists with length
            len(networks), counting the number of ii, ic and cc branches for each network.

    """

    # Storage arrays
    num_i, num_y, num_x = [], [], []
    num_i_i, num_i_c, num_c_c = [], [], []

    # Loop over all networks, first compute node topology, then branches
    for n in networks:
        node_types = analyze_intersections_of_sets(n, tol=n.tol)
        i = node_types["i_nodes"]
        y = node_types["y_nodes"]
        x = node_types["x_nodes"]

        # Count nodes on the domain boundary - these will be subtracted from
        # the i-nodes
        p = n.pts
        d = n.domain
        tol = n.tol
        num_bound = np.logical_and.reduce(
            (
                np.abs(p[0] - d["xmin"]) < tol,
                np.abs(p[0] - d["xmax"]) < tol,
                np.abs(p[1] - d["ymin"]) < tol,
                np.abs(p[1] - d["ymax"]) < tol,
            )
        ).sum()
        # Store information
        num_i.append((i.sum() - num_bound).astype(np.int))
        num_y.append(y.sum().astype(np.int))
        num_x.append(x.sum().astype(np.int))

        # Branch topology
        # Find the number of T-intersections that ends in each fracture.
        # This is different from the field y_nodes, which considers endpoints
        # of the fracture itself.
        a = node_types["arrests"]
        # Isolated branches have two i-nodes, no other nodes
        ii = (
            np.logical_and.reduce((i == 2, y == 0, x == 0, a == 0)).sum().astype(np.int)
        )
        # ic branches either have one i-node and one y-node (the end point of
        # the fracture must be etiher i or y - x and a cannot be the end),
        # Or two i-nodes together with at least one cross. In the latter case
        # there will be two ic-branches
        ic = np.logical_and.reduce((i == 1, y == 1, x == 0, a == 0)).sum().astype(
            np.int
        ) + 2 * np.logical_and(i == 2, x + a > 0).sum().astype(np.int)
        # CC-branches are formed between y, x and a-nodes. Their number on
        # each fracture will be the number of such nodes minus one.
        cc = np.maximum(y + x + a - 1, 0).sum()

        num_i_i.append(ii)
        num_i_c.append(ic)
        num_c_c.append(cc)

    node_topology = {"i": num_i, "y": num_y, "x": num_x}
    branch_topology = {"ii": num_i_i, "ic": num_i_c, "cc": num_c_c}
    return node_topology, branch_topology


def analyze_intersections_of_sets(set_1, set_2=None, tol=1e-4):
    """ Count the number of node types (I, X, Y) per fracture in one or two
    fracture sets.

    The method finds, for each fracture, how many of the nodes are end-nodes,
    how many of the end-nodes abut to other fractures, and also how many other
    fractures crosses the main one in the form of an X or T intersection,
    respectively.

    Note that the fracture sets are treated as if they contain a single
    family, independent of any family tags in set_x.edges[2].

    To consider only intersections between fractures in different sets (e.g.
    disregard all intersections between fractures in the same family), run
    this function first with two input sets, then separately with a single set
    and take the difference.

    Parameters:
        set_1 (FractureSet): First set of fractures. Will be treated as a
            single family, independent of whether there are different family
            tags in set_1.edges[2].
        set_1 (FractureSet, optional): First set of fractures. Will be treated
            as a single family, independent of whether there are different
            family tags in set_1.edges[2]. If not provided,
        tol (double, optional): Tolerance used in computations to find
            intersections between fractures. Defaults to 1e-4.

    Returns:
        dictionary with keywords i_nodes, y_nodes, x_nodes, arrests. For each
            fracture in the set:
                i_nodes gives the number of the end-nodes of the fracture which
                    are i-nodes
                y_nodes gives the number of the end-nodes of the fracture which
                    terminate in another fracture
                x_nodes gives the number of X-intersections along the fracture
                arrests gives the number of fractures that terminates as a
                    Y-node in this fracture

        If two fracture sets are submitted, two such dictionaries will be
        returned, reporting on the fractures in the first and second set,
        respectively.

    """

    pts_1 = set_1.pts
    num_fracs_1 = set_1.edges.shape[1]

    num_pts_1 = pts_1.shape[1]

    # If a second set is present, also focus on the nodes in the intersections
    # between the two sets
    if set_2 is None:
        # The nodes are a sigle set
        pts = pts_1
        edges = np.vstack((set_1.edges[:2], np.arange(num_fracs_1, dtype=np.int)))

    else:
        # Assign famility based on the two sets, override whatever families
        # were assigned originally
        edges_1 = np.vstack((set_1.edges[:2], np.arange(num_fracs_1, dtype=np.int)))
        pts_2 = set_2.pts
        pts = np.hstack((pts_1, pts_2))

        num_fracs_2 = set_2.edges.shape[1]
        edges_2 = np.vstack((set_2.edges[:2], np.arange(num_fracs_2, dtype=np.int)))

        # The second set will have its points offset by the number of points
        # in the first set, and its edge numbering by the number of fractures
        # in the first set
        edges_2[:2] += num_pts_1
        edges_2[2] += num_fracs_1
        edges = np.hstack((edges_1, edges_2))

    num_fracs = edges.shape[1]

    _, e_split = pp.cg.remove_edge_crossings2(pts, edges, tol=tol)

    # Find which of the split edges belong to family_1 and 2
    family_1 = np.isin(e_split[2], np.arange(num_fracs_1))
    if set_2 is not None:
        family_2 = np.isin(e_split[2], num_fracs_1 + np.arange(num_fracs_2))
    else:
        family_2 = np.logical_not(family_1)
    assert np.all(family_1 + family_2 == 1)

    # Story the fracture id of the split edges
    frac_id_split = e_split[2].copy()

    # Assign family identities to the split edges
    e_split[2, family_1] = 0
    e_split[2, family_2] = 1

    # For each fracture, identify its endpoints in terms of indices in the new
    # split nodes.
    end_pts = np.zeros((2, num_fracs))

    all_points_of_edge = np.empty(num_fracs, dtype=np.object)

    # Loop over all fractures
    for fi in range(num_fracs):
        # Find all split edges associated with the fracture, and its points
        loc_edges = frac_id_split == fi
        loc_pts = e_split[:2, loc_edges].ravel()

        # The endpoint ooccurs only once in this list
        loc_end_points = np.where(np.bincount(loc_pts) == 1)[0]
        assert loc_end_points.size == 2

        end_pts[0, fi] = loc_end_points[0]
        end_pts[1, fi] = loc_end_points[1]

        # Also store all nodes of this edge, including intersection points
        all_points_of_edge[fi] = np.unique(loc_pts)

    i_n, l_n, y_n_c, y_n_f, x_n = count_node_types_between_families(e_split)

    num_i_nodes = np.zeros(num_fracs)
    num_y_nodes = np.zeros(num_fracs)
    num_x_nodes = np.zeros(num_fracs)
    num_arrests = np.zeros(num_fracs)

    for fi in range(num_fracs):
        if set_2 is None:
            row = 0
            col = 0
        else:
            is_set_1 = fi < num_fracs_1
            if is_set_1:
                row = 0
                col = 1
            else:
                row = 1
                col = 0

        # Number of the endnodes that are y-nodes
        num_y_nodes[fi] = np.sum(np.isin(end_pts[:, fi], y_n_c[row, col]))

        # The number of I-nodes are 2 - the number of Y-nodes
        num_i_nodes[fi] = 2 - num_y_nodes[fi]

        # Number of nodes identified as x-nodes for this edge
        num_x_nodes[fi] = np.sum(np.isin(all_points_of_edge[fi], x_n[row, col]))

        # The number of fractures that have this edge as the constraint in the
        # T-node. This is are all nodes that are not end-nodes (2), and not
        # X-nodes
        num_arrests[fi] = all_points_of_edge[fi].size - num_x_nodes[fi] - 2

    if set_2 is None:
        results = {
            "i_nodes": num_i_nodes,
            "y_nodes": num_y_nodes,
            "x_nodes": num_x_nodes,
            "arrests": num_arrests,
        }
        return results
    else:
        results_set_1 = {
            "i_nodes": num_i_nodes[:num_fracs_1],
            "y_nodes": num_y_nodes[:num_fracs_1],
            "x_nodes": num_x_nodes[:num_fracs_1],
            "arrests": num_arrests[:num_fracs_1],
        }
        results_set_2 = {
            "i_nodes": num_i_nodes[num_fracs_1:],
            "y_nodes": num_y_nodes[num_fracs_1:],
            "x_nodes": num_x_nodes[num_fracs_1:],
            "arrests": num_arrests[num_fracs_1:],
        }
        return results_set_1, results_set_2


def count_node_types_between_families(e):
    """ Count the number of nodes (I, L, Y, X) between different fracture
    families.

    The fracutres are defined by their end-points (endpoints of branches should
    alse be fine).

    Parameters:
        e (np.array, 2 or 3 x n_frac): First two rows represent endpoints of
            fractures or branches. The third (optional) gives the family of
            each fracture. If this is not specified, the fractures are assumed
            to come from the same family.

    Returns:

        ** NB: For all returned matrices the family numbers are sorted, and
        rows and columns are defined accordingly.

        np.array (num_families x num_families): Each element contains a numpy
            array with the indexes of all I-connections for the relevnant
            network. The main diagonal describes the i-nodes of the set
            considered by itself.
        np.array (num_families x num_families): Each element contains a numpy
            array with the indexes of all L-connections for the relevnant
            networks. The main diagonal contains L-connection within the nework
            itself, off-diagonal elements represent the meeting between two
            different families. The elements [i, j] and [j, i] will be
            identical.
        np.array (num_families x num_families): Each element contains a numpy
            array with the indexes of all Y-connections that were constrained
            by the other family. On the main diagonal, these are all the
            fractures. On the off-diagonal elments, element [i, j] contains
            all nodes where family i was constrained by family j.
        np.array (num_families x num_families): Each element contains a numpy
            array with the indexes of all Y-connections that were not
            constrained by the other family. On the main diagonal, these are
            all the fractures. On the off-diagonal elments, element [i, j]
            contains all nodes where family j was constrained by family i.
        np.array (num_families x num_families): Each element contains a numpy
            array with the indexes of all X-connections for the relevnant
            networks. The main diagonal contains X-connection within the nework
            itself, off-diagonal elements represent the meeting between two
            different families. The elements [i, j] and [j, i] will be
            identical.

    """

    if e.shape[0] > 2:
        num_families = np.unique(e[2]).size
    else:
        num_families = 1
        e = np.vstack((e, np.zeros(e.shape[1], dtype=np.int)))

    # Nodes occuring only once. Hanging.
    i_nodes = np.empty((num_families, num_families), dtype=np.object)
    # Nodes occuring twice, defining an L-intersection, or equivalently the
    # meeting of two branches of a fracture
    l_nodes = np.empty_like(i_nodes)
    # Nodes in a Y-connection (or T-) that occurs twice. That is, the fracture
    # was not arrested by the crossing fracture.
    y_nodes_full = np.empty_like(i_nodes)
    # Nodes in a Y-connection (or T-) that occurs once. That is, the fracture
    # was arrested by the crossing fracture.
    y_nodes_constrained = np.empty_like(i_nodes)
    # Nodes in an X-connection.
    x_nodes = np.empty_like(i_nodes)

    max_p_ind = e[:2].max()

    # Ensure that all vectors are of the same size. Not sure if this is always
    # necessary, since we're doing an np.where later, but clearly this is useful
    def bincount(hit):
        tmp = np.bincount(e[:2, hit].ravel())
        num_occ = np.zeros(max_p_ind + 1, dtype=np.int)
        num_occ[: tmp.size] = tmp
        return num_occ

    # First do each family by itself
    families = np.sort(np.unique(e[2]))
    for i in families:
        hit = np.where(e[2] == i)[0]
        num_occ = bincount(hit)

        if np.any(num_occ > 4):
            raise ValueError("Not ready for more than two fractures meeting")

        i_nodes[i, i] = np.where(num_occ == 1)[0]
        l_nodes[i, i] = np.where(num_occ == 2)[0]
        y_nodes_full[i, i] = np.where(num_occ == 3)[0]
        y_nodes_constrained[i, i] = np.where(num_occ == 3)[0]
        x_nodes[i, i] = np.where(num_occ == 4)[0]

    # Next, compare two families

    for i in families:
        for j in families:
            if i == j:
                continue

            hit_i = np.where(e[2] == i)[0]
            num_occ_i = bincount(hit_i)
            hit_j = np.where(e[2] == j)[0]
            num_occ_j = bincount(hit_j)

            # I-nodes are not interesting in this setting (they will be
            # covered by the single-family case)

            hit_i_i = np.where(np.logical_and(num_occ_i == 1, num_occ_j == 0))[0]
            i_nodes[i, j] = hit_i_i
            hit_i_j = np.where(np.logical_and(num_occ_i == 0, num_occ_j == 1))[0]
            i_nodes[j, i] = hit_i_j

            # L-nodes between different families
            hit_l = np.where(np.logical_and(num_occ_i == 1, num_occ_j == 1))[0]
            l_nodes[i, j] = hit_l
            l_nodes[j, i] = hit_l

            # Two types of Y-nodes between different families
            hit_y = np.where(np.logical_and(num_occ_i == 1, num_occ_j == 2))[0]
            y_nodes_constrained[i, j] = hit_y
            y_nodes_full[j, i] = hit_y

            hit_y = np.where(np.logical_and(num_occ_i == 2, num_occ_j == 1))[0]
            y_nodes_constrained[j, i] = hit_y
            y_nodes_full[i, j] = hit_y

            hit_x = np.where(np.logical_and(num_occ_i == 2, num_occ_j == 2))[0]
            x_nodes[i, j] = hit_x
            x_nodes[j, i] = hit_x

    return i_nodes, l_nodes, y_nodes_constrained, y_nodes_full, x_nodes
