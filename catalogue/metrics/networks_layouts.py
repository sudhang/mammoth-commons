# Plot the network:
"""
******
Layout
******

Node positioning algorithms for graph drawing.

For `random_layout()` the possible resulting shape
is a square of side [0, scale] (default: [0, 1])
Changing `center` shifts the layout by that amount.

For the other layout routines, the extent is
[center - scale, center + scale] (default: [-1, 1]).

Warning: Most layout routines have only been tested in 2-dimensions.

"""

import networkx as nx
from networkx.utils import np_random_state

__all__ = [
    "bipartite_layout",
    "circular_layout",
    "forceatlas2_layout",
    "kamada_kawai_layout",
    "random_layout",
    "rescale_layout",
    "rescale_layout_dict",
    "shell_layout",
    "spring_layout",
    "spectral_layout",
    "planar_layout",
    "fruchterman_reingold_layout",
    "spiral_layout",
    "multipartite_layout",
    "bfs_layout",
    "arf_layout",
]


def _process_params(G, center, dim):
    # Some boilerplate code.
    import numpy as np

    if not isinstance(G, nx.Graph):
        empty_graph = nx.Graph()
        empty_graph.add_nodes_from(G)
        G = empty_graph

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    if len(center) != dim:
        msg = "length of center coordinates must match dimension of layout"
        raise ValueError(msg)

    return G, center


@np_random_state("seed")
def forceatlas2_layout(
    G,
    pos=None,
    *,
    max_iter=100,
    jitter_tolerance=1.0,
    scaling_ratio=2.0,
    gravity=1.0,
    distributed_action=False,
    strong_gravity=False,
    node_mass=None,
    node_size=None,
    weight=None,
    dissuade_hubs=False,
    linlog=False,
    seed=None,
    dim=2,
):
    """Position nodes using the ForceAtlas2 force-directed layout algorithm.

    This function applies the ForceAtlas2 layout algorithm [1]_ to a NetworkX graph,
    positioning the nodes in a way that visually represents the structure of the graph.
    The algorithm uses physical simulation to minimize the energy of the system,
    resulting in a more readable layout.

    Parameters
    ----------
    G : nx.Graph
        A NetworkX graph to be laid out.
    pos : dict or None, optional
        Initial positions of the nodes. If None, random initial positions are used.
    max_iter : int (default: 100)
        Number of iterations for the layout optimization.
    jitter_tolerance : float (default: 1.0)
        Controls the tolerance for adjusting the speed of layout generation.
    scaling_ratio : float (default: 2.0)
        Determines the scaling of attraction and repulsion forces.
    distributed_attraction : bool (default: False)
        Distributes the attraction force evenly among nodes.
    strong_gravity : bool (default: False)
        Applies a strong gravitational pull towards the center.
    node_mass : dict or None, optional
        Maps nodes to their masses, influencing the attraction to other nodes.
    node_size : dict or None, optional
        Maps nodes to their sizes, preventing crowding by creating a halo effect.
    dissuade_hubs : bool (default: False)
        Prevents the clustering of hub nodes.
    linlog : bool (default: False)
        Uses logarithmic attraction instead of linear.
    seed : int, RandomState instance or None  optional (default=None)
        Used only for the initial positions in the algorithm.
        Set the random state for deterministic node layouts.
        If int, `seed` is the seed used by the random number generator,
        if numpy.random.RandomState instance, `seed` is the random
        number generator,
        if None, the random number generator is the RandomState instance used
        by numpy.random.
    dim : int (default: 2)
        Sets the dimensions for the layout. Ignored if `pos` is provided.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.florentine_families_graph()
    >>> pos = nx.forceatlas2_layout(G)
    >>> nx.draw(G, pos=pos)

    References
    ----------
    .. [1] Jacomy, M., Venturini, T., Heymann, S., & Bastian, M. (2014).
           ForceAtlas2, a continuous graph layout algorithm for handy network
           visualization designed for the Gephi software. PloS one, 9(6), e98679.
           https://doi.org/10.1371/journal.pone.0098679
    """
    import numpy as np

    if len(G) == 0:
        return {}
    # parse optional pos positions
    if pos is None:
        pos = nx.random_layout(G, dim=dim, seed=seed)
        pos_arr = np.array(list(pos.values()))
    else:
        # set default node interval within the initial pos values
        pos_init = np.array(list(pos.values()))
        max_pos = pos_init.max(axis=0)
        min_pos = pos_init.min(axis=0)
        dim = max_pos.size
        pos_arr = min_pos + seed.rand(len(G), dim) * (max_pos - min_pos)
        for idx, node in enumerate(G):
            if node in pos:
                pos_arr[idx] = pos[node].copy()

    mass = np.zeros(len(G))
    size = np.zeros(len(G))

    # Only adjust for size when the users specifies size other than default (1)
    adjust_sizes = False
    if node_size is None:
        node_size = {}
    else:
        adjust_sizes = True

    if node_mass is None:
        node_mass = {}

    for idx, node in enumerate(G):
        mass[idx] = G.degree(node) + 1
        size[idx] = G.degree(node)

    n = len(G)
    gravities = np.zeros((n, dim))
    attraction = np.zeros((n, dim))
    repulsion = np.zeros((n, dim))
    A = nx.to_numpy_array(G, weight="weight")

    def estimate_factor(n, swing, traction, speed, speed_efficiency, jitter_tolerance):
        """Computes the scaling factor for the force in the ForceAtlas2 layout algorithm.

        This   helper  function   adjusts   the  speed   and
        efficiency  of the  layout generation  based on  the
        current state of  the system, such as  the number of
        nodes, current swing, and traction forces.

        Parameters
        ----------
        n : int
            Number of nodes in the graph.
        swing : float
            The current swing, representing the oscillation of the nodes.
        traction : float
            The current traction force, representing the attraction between nodes.
        speed : float
            The current speed of the layout generation.
        speed_efficiency : float
            The efficiency of the current speed, influencing how fast the layout converges.
        jitter_tolerance : float
            The tolerance for jitter, affecting how much speed adjustment is allowed.

        Returns
        -------
        tuple
            A tuple containing the updated speed and speed efficiency.

        Notes
        -----
        This function is a part of the ForceAtlas2 layout algorithm and is used to dynamically adjust the
        layout parameters to achieve an optimal and stable visualization.

        """
        import numpy as np

        # estimate jitter
        opt_jitter = 0.05 * np.sqrt(n)
        min_jitter = np.sqrt(opt_jitter)
        max_jitter = 10
        min_speed_efficiency = 0.05

        other = min(max_jitter, opt_jitter * traction / n**2)
        jitter = jitter_tolerance * max(min_jitter, other)

        if swing / traction > 2.0:
            if speed_efficiency > min_speed_efficiency:
                speed_efficiency *= 0.5
            jitter = max(jitter, jitter_tolerance)
        if swing == 0:
            target_speed = np.inf
        else:
            target_speed = jitter * speed_efficiency * traction / swing

        if swing > jitter * traction:
            if speed_efficiency > min_speed_efficiency:
                speed_efficiency *= 0.7
        elif speed < 1000:
            speed_efficiency *= 1.3

        max_rise = 0.5
        speed = speed + min(target_speed - speed, max_rise * speed)
        return speed, speed_efficiency

    speed = 1
    speed_efficiency = 1
    swing = 1
    traction = 1
    for _ in range(max_iter):
        # compute pairwise difference
        diff = pos_arr[:, None] - pos_arr[None]
        # compute pairwise distance
        distance = np.linalg.norm(diff, axis=-1)

        # linear attraction
        if linlog:
            attraction = -np.log(1 + distance) / distance
            np.fill_diagonal(attraction, 0)
            attraction = np.einsum("ij, ij -> ij", attraction, A)
            attraction = np.einsum("ijk, ij -> ik", diff, attraction)

        else:
            attraction = -np.einsum("ijk, ij -> ik", diff, A)

        if distributed_action:
            attraction /= mass[:, None]

        # repulsion
        tmp = mass[:, None] @ mass[None]
        if adjust_sizes:
            distance += -size[:, None] - size[None]

        d2 = distance**2
        # remove self-interaction
        np.fill_diagonal(tmp, 0)
        np.fill_diagonal(d2, 1)
        factor = (tmp / d2) * scaling_ratio
        repulsion = np.einsum("ijk, ij -> ik", diff, factor)

        # gravity
        gravities = (
            -gravity
            * mass[:, None]
            * pos_arr
            / np.linalg.norm(pos_arr, axis=-1)[:, None]
        )

        if strong_gravity:
            gravities *= np.linalg.norm(pos_arr, axis=-1)[:, None]
        # total forces
        update = attraction + repulsion + gravities

        # compute total swing and traction
        swing += (mass * np.linalg.norm(pos_arr - update, axis=-1)).sum()
        traction += (0.5 * mass * np.linalg.norm(pos_arr + update, axis=-1)).sum()

        speed, speed_efficiency = estimate_factor(
            n,
            swing,
            traction,
            speed,
            speed_efficiency,
            jitter_tolerance,
        )

        # update pos
        if adjust_sizes:
            swinging = mass * np.linalg.norm(update, axis=-1)
            factor = 0.1 * speed / (1 + np.sqrt(speed * swinging))
            df = np.linalg.norm(update, axis=-1)
            factor = np.minimum(factor * df, 10.0 * np.ones(df.shape)) / df
        else:
            swinging = mass * np.linalg.norm(update, axis=-1)
            factor = speed / (1 + np.sqrt(speed * swinging))

        pos_arr += update * factor[:, None]
        if abs((update * factor[:, None]).sum()) < 1e-10:
            break

    return dict(zip(G, pos_arr))
