import tensorflow as tf
import numpy as np  # maybe it is better to use tensorflow only
from scipy.optimize import minimize
import heapq
from collections import namedtuple
from copy import deepcopy
import logging
parent_logger = logging.getLogger(__name__)


Coordinates = namedtuple('Coordinates', ['x', 'y', 'z'])
AtomDescriptor = namedtuple('AtomDescriptor', ['radius', 'channels'])


class Atom:
    """Structure for holding coordinates and atom types."""

    def __init__(self, coordinates, atom_type):
        self.coordinates = coordinates
        self.atom_type = atom_type

    def __str__(self):
        return "A(" + str(self.coordinates[0]) + " " + str(self.coordinates[1]) + " " + \
               str(self.coordinates[2]) + ' ' + str(self.atom_type) + ")"


class Molecule:
    """Struct for holding atoms."""

    def __init__(self, atoms=None):
        if atoms:
            self.atoms = atoms
        else:
            self.atoms = []

    def add_atom(self, atom):
        self.atoms.append(atom)

    def __len__(self):
        return len(self.atoms)

    def __str__(self):
        result = ""
        for atom in self.atoms[:-1]:
            result += atom.__str__() + ", "
        result += self.atoms[-1].__str__()
        return result


class MoleculeHeap:
    """Heap queue for molecules.

    Saves top-k high priority molecules. Each molecule added with its priority.
    """
    def __init__(self, max_size):
        self.heap = []
        self.total_entries = 0
        self.max_size = max_size

    def add(self, molecule, priority):
        """ Add molecule. Heap contain high priority molecules.
        Since, standard python heapq module is min-heap, use minus norm as priority.
        """
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, (priority, self.total_entries, molecule))
            self.total_entries += 1
        else:
            heapq.heappushpop(self.heap, (priority, self.total_entries, molecule))
            self.total_entries += 1

    def get_items(self):
        pairs = self.get_pairs()
        return [molecule for (_, molecule) in pairs]

    def get_pairs(self):
        pairs = [(-norm, molecule) for (norm, _id, molecule) in self.heap]
        pairs = sorted(pairs, key=lambda pair: pair[0])
        return pairs

    def __len__(self):
        return len(self.heap)


class Fitter:
    """Transforms density tensors into structs.
    Haven't been tested for non-square tensors."""

    def __init__(self, molecule_to_tensor_func, atom_type_to_descriptor, verbose=2):
        """Set parameters."""

        # Number of best molecules to hold.
        self.best_max = 10

        # Maximal number of atoms we try to add each dfs step before the optimization.
        # Reduce value to speed up conversion. Set to 1 to optimize right after adding
        # an atom.
        self.max_atom_by_dfs_step = 4

        # Maximal number of best molecules that go to the next dfs step. Reduce value
        # to speed up conversion.
        self.max_molecules_by_dfs_step = 10

        # Minimal density for atom detection. Increase value to speed up conversion.
        # Read Fitter._get_possible_atom_types on how to use it.
        self.threshold = 0.5

        # Maximal number of atoms in molecule.
        self.max_atoms = 20

        # Function, that takes Molecule object and return its density tensor.
        self.molecule_to_tensor = molecule_to_tensor_func

        # Grid resolution
        self.voxel_size = 0.5

        # Dict from atom type to AtomDescriptors
        self.atom_type_to_descriptor = atom_type_to_descriptor

        # Turn on optimization by atom coordinates
        self.optimization_is_on = True

        # Maximum number of optimization iterations
        self.optimization_maxiter = 100

        # Tolerance of the optimizer
        self.optimization_tolerance = 0.0001

        if verbose == 3:
            parent_logger.setLevel(logging.DEBUG)
        elif verbose == 2:
            parent_logger.setLevel(logging.INFO)
        elif verbose == 1:
            parent_logger.setLevel(logging.ERROR)
        else:
            raise ValueError('Allowed verbose levels: [1, 2, 3], but ' + str(verbose) + ' provided.')
        parent_logger.debug("Fitter is initialized.")

        # Change it for precise logging control
        # logging.getLogger(__name__ + ".dfs").setLevel(logging.DEBUG)
        # logging.getLogger(__name__ + ".add_atoms").setLevel(logging.DEBUG)
        # logging.getLogger(__name__ + ".get_max_indices").setLevel(logging.DEBUG)
        # logging.getLogger(__name__ + ".index_to_coordinates").setLevel(logging.DEBUG)
        # logging.getLogger(__name__ + ".optimize_molecule").setLevel(logging.DEBUG)
        # logging.getLogger(__name__ + ".get_possible_atom_types").setLevel(logging.DEBUG)
        # logging.getLogger(__name__ + ".dfs_select_good_molecules").setLevel(logging.DEBUG)

    def tensor_to_molecule(self, tensor, **kwargs):
        """Transform density tensor into Molecule object.

        Arguments:
        tensor -- 4D tf.tensor [x, y, z, c], where c -- channels.
        """
        best_molecules = MoleculeHeap(self.best_max)
        current_molecule = Molecule()
        try:
            self._dfs(tensor, current_molecule, best_molecules, **kwargs)
        except KeyboardInterrupt:
            pass
        return best_molecules.get_pairs()

    def _dfs(self, original_tensor, current_molecule, best_molecules, **kwargs):
        """DFS step for the fitting procedure.

        Each step tries to add up to self.max_atom_by_step atoms into maximal density points.

        Arguments:
        original_tensor -- original 4D tf.tensor [x, y, z, c], where c -- channels.
        current_molecule -- Molecule object to add atoms to.
        best_molecules -- heap with top molecules.
        """
        logger = logging.getLogger(__name__ + '.dfs')

        # Tensor of the difference between the original tensor and the density tensor of the current_molecule
        current_tensor = original_tensor - self.molecule_to_tensor(current_molecule)
        current_norm = tf.norm(current_tensor)
        logger.debug("Current norm:" + str(float(current_norm)))
        if best_molecules.heap:
            logger.info("DFS step. Depth: " + str(len(current_molecule)) + " Best norm: " + str(-best_molecules.heap[-1][0]))

        new_molecules = self._add_atoms(current_molecule, current_tensor)

        # Molecules for the next dfs step
        good_molecules = []
        for molecule in new_molecules:
            if self.optimization_is_on:
                molecule, norm = self._optimize_molecule(molecule, original_tensor)
            else:
                norm = tf.norm(original_tensor - self.molecule_to_tensor(molecule))
                logger.debug("Molecule: " + str(molecule) + " Norm: " + str(norm))
            if norm < current_norm:
                good_molecules.append((molecule, norm))

        # Additional filtering of molecules
        good_molecules = self._dfs_select_good_molecules(good_molecules, best_molecules)
        for molecule in good_molecules:
            self._dfs(original_tensor, molecule, best_molecules, **kwargs)

    def _add_atoms(self, current_molecule, current_tensor):
        """Creates molecules by adding up to self.max_atoms_per_step atoms into the current molecule.

        For example, if self.max_atoms_per_step is 2, it will create molecules M with an atom at the top-1
        density point and then will add an atom at the top-2 point to the copy of created molecules M.

        Arguments:
        current_molecule -- Molecule object to add atoms to.
        current_tensor -- Tensor of the difference between the original tensor and the density tensor of the current_molecule
        """
        logger = logging.getLogger(__name__ + '.add_atoms')

        # Indices of maximal density points
        max_indices = self._get_max_indices(current_tensor)

        molecules = []
        last_layer_molecules = [current_molecule]
        for i, index in enumerate(max_indices):
            new_molecules = []
            new_atom_coordinates = self._index_to_coordinates(index, current_tensor.shape[:-1])
            possible_atom_types = self._get_possible_atom_types(current_tensor[index])
            for atom_type in possible_atom_types:
                new_atom = Atom(new_atom_coordinates, atom_type)
                for molecule in last_layer_molecules:
                    new_molecule = deepcopy(molecule)
                    new_molecule.add_atom(new_atom)
                    new_molecules.append(new_molecule)
            last_layer_molecules = new_molecules
            molecules.extend(new_molecules)
            logger.debug("For index " + str(i) + " added molecules: " + str(len(new_molecules)) + ", " + str([str(molecule) for molecule in new_molecules]))
        return molecules

    def _get_max_indices(self, tensor):
        """Return self.max_atoms_by_step points of maximal value in tensor.

        Haven't been tested with non-square tensors."""
        logger = logging.getLogger(__name__ + ".get_max_indices")

        # Since we are looking for different xyz coordinates, select only maximum by channel axis
        max_channels = tf.math.reduce_max(tensor, axis=-1)

        # Make tensor flat
        max_channels = tf.reshape(max_channels, [-1])

        _, top_k_indices = tf.math.top_k(max_channels, k=self.max_atom_by_dfs_step)
        logger.debug("Top k flat indices:" + str(top_k_indices))

        # For each maximal voxel, we will select voxels with the same density by random.
        # If we have one such voxel, we will select it. But if there are many, it will increase sparseness of new atoms.
        # It's especially important for the first step, since there are may be many 1.0 voxels.
        result_indices = []
        for max_index in top_k_indices:
            same_good_indices = tf.where(max_channels == max_channels[max_index])
            same_good_indices = tf.random.shuffle(same_good_indices)
            for index in same_good_indices:
                if int(index) not in result_indices:
                    # Turn index back to 3D
                    result_indices.append(np.unravel_index(int(index), tensor.shape[:-1]))
                    break
        logger.debug("Result indices: " + str(result_indices))
        return result_indices


    def _index_to_coordinates(self, index, tensor_shape):
        """Converts tensor indices into grid coordinates."""
        logger = logging.getLogger(__name__ + ".index_to_coordinates")
        xyz = (np.array(index) - np.array(tensor_shape) / 2) * self.voxel_size
        logger.debug("Index: " + str(index) + " converted to " + str(xyz))
        return Coordinates(*xyz)

    def _get_possible_atom_types(self, channels):
        """Returns list of possible atoms at point with such channels.

        An atom considered to be possible, if its channels are subset of the channels argument.
        It is because it is possible to has some channel density from the neighboor atom, but it's not possible
        to has low density of channel if atom with such channel is presented.
        """
        logger = logging.getLogger(__name__ + ".get_possible_atom_types")
        logger.debug("Input channels: " + str(channels))

        current_channels = set()
        for i, channel in enumerate(channels):
            if channel > self.threshold:
                current_channels.add(i)
        logger.debug("Current channels: " + str(current_channels))

        possible_atom_types = []
        for atom_type, (_, atom_channels) in self.atom_type_to_descriptor.items():
            if atom_channels.issubset(current_channels):
                possible_atom_types.append(atom_type)
        logger.debug("Possible atom types: " + str(possible_atom_types))

        return possible_atom_types

    def _optimize_molecule(self, molecule, original_tensor, **kwargs):
        """Optimize coordinates of atoms in the molecule, so its density is looks like the original tensor.
        """
        logger = logging.getLogger(__name__ + ".optimize_molecule")

        def norm_of_difference(coordinates, *args):
            """ Target function to minimize.
            Gets a tuple (original_tensor, molecule, mol_to_ten) as *args.

            coordinates -- 3*n vector of molecule's atoms coordinates.
            original_tensor -- original density tensor.
            molecule -- molecule to optimize
            mol_to_ten -- function converting molecule into tensor
            """
            # logger.debug("Coor: " + str(coordinates))
            original_tensor, molecule, molecule_to_tensor_func = args
            molecule = deepcopy(molecule)
            for i, atom in enumerate(molecule.atoms):
                new_coordinates = Coordinates(coordinates[3 * i + 0],
                                              coordinates[3 * i + 1],
                                              coordinates[3 * i + 2])
                atom.coordinates = new_coordinates
            result_tensor = molecule_to_tensor_func(molecule)
            norm = float(tf.norm(original_tensor - result_tensor))
            # logger.debug("Target f: " + str(norm))
            return norm

        # Bonds, so fitted atoms won't leave the tensor
        x_bond, y_bond, z_bond = (np.array(original_tensor.shape[:-1]) / 2) * self.voxel_size

        coordinates = np.zeros(3 * len(molecule))
        bounds = []
        for i, atom in enumerate(molecule.atoms):
            coordinates[3 * i + 0] = atom.coordinates[0]
            bounds.append((-x_bond, x_bond))

            coordinates[3 * i + 1] = atom.coordinates[1]
            bounds.append((-y_bond, y_bond))

            coordinates[3 * i + 2] = atom.coordinates[2]
            bounds.append((-z_bond, z_bond))

        logger.debug("Initial coordinates: " + str(coordinates))
        logger.debug("Initial bounds: " + str(bounds))
        current_norm = norm_of_difference(coordinates, original_tensor, molecule, self.molecule_to_tensor)
        logger.debug("Initial norm: " + str(current_norm))

        # bounds are supported by Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and trust-constr methods.
        result = minimize(fun=norm_of_difference,
                          x0=coordinates,
                          args=(original_tensor, molecule, self.molecule_to_tensor),
                          bounds=bounds,
                          method='L-BFGS-B',
                          tol=self.optimization_tolerance,
                          options={'maxiter' : self.optimization_maxiter,
                                   'gtol': 1e-09,
                                   'eps': 1e-4})

        for i, atom in enumerate(molecule.atoms):
            new_coordinates = Coordinates(result.x[3 * i + 0],
                                          result.x[3 * i + 1],
                                          result.x[3 * i + 2])
            atom.coordinates = new_coordinates
        logger.debug("Final norm: " + str(result.fun))
        logger.debug("Total iterations: " + str(result.nit))
        logger.debug("Message: " + str(result.message))

        return molecule, result.fun

    def _dfs_select_good_molecules(self, good_molecules, best_molecules):
        """Select molecules for the next round of dfs.

        At the moment it selects top self.max_molecules_by_dfs_step molecules.

        Arguments:
        good_molecules -- list of pairs (molecule, norm of its tensor)
        """
        logger = logging.getLogger(__name__ + ".dfs_select_good_molecules")

        new_good_molecules = MoleculeHeap(self.max_molecules_by_dfs_step)
        for molecule, norm in good_molecules:
            best_molecules.add(molecule, -norm)
            if len(molecule) < self.max_atoms:
                new_good_molecules.add(molecule, -norm)
        logger.debug("Good molecules: " + str([str(molecule) for molecule in new_good_molecules.get_items()]))
        logger.debug("Norms of good molecules: " + str([norm for (norm, _) in new_good_molecules.get_pairs()]))
        logger.debug("Best molecules: " + str([str(molecule) for molecule in best_molecules.get_items()]))
        return new_good_molecules.get_items()


