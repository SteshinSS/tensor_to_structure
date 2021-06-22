import tensorflow as tf
import numpy as np  # maybe it is better to use tensorflow only
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
        return [molecule for (priority, _id, molecule) in self.heap]

    def get_pairs(self):
        return [(-norm, molecule) for (norm, _id, molecule) in self.heap]

    def __len__(self):
        return len(self.heap)


class Fitter:
    """Transforms density tensors into structs.
    Haven't been tested for non-square tensors."""

    def __init__(self, molecule_to_tensor_func, atom_type_to_descriptor, verbose=3):
        """Set parameters."""

        # Number of best molecules we hold
        self.best_max = 10

        # Maximal number of atoms we try to add each bfs step
        self.max_atom_by_step = 2

        # Maximal numbers of best molecules that go to the next bfs step
        self.max_molecules_by_step = 3

        # Minimal density for atom detection
        self.threshold = 0.3

        # Maximal number of atoms in molecule
        self.max_atoms = 30

        # Function, that takes Molecule object and return its density tensor.
        self.molecule_to_tensor = molecule_to_tensor_func

        # Grid resolution
        self.voxel_size = 0.5

        # Dict from atom type to AtomDescriptors
        self.atom_type_to_descriptor = atom_type_to_descriptor

        if verbose == 3:
            parent_logger.setLevel(logging.DEBUG)
        parent_logger.debug("Fitter is initialized.")

        logging.getLogger(__name__ + ".bfs").setLevel(logging.INFO)
        logging.getLogger(__name__ + ".add_atoms").setLevel(logging.CRITICAL)
        logging.getLogger(__name__ + ".get_max_indices").setLevel(logging.CRITICAL)
        logging.getLogger(__name__ + ".index_to_coordinates").setLevel(logging.CRITICAL)
        logging.getLogger(__name__ + ".get_possible_atom_types").setLevel(logging.CRITICAL)
        logging.getLogger(__name__ + ".select_good_molecules").setLevel(logging.CRITICAL)

    def tensor_to_molecule(self, tensor, **kwargs):
        """Transform density tensor into struct.

        Arguments:
        tensor -- 4D tf.tensor [x, y, z, c], where c -- channels.
        """
        best_molecules = MoleculeHeap(self.best_max)
        current_molecule = Molecule()
        self._bfs(tensor, current_molecule, best_molecules, **kwargs)
        return best_molecules.get_pairs()

    def _bfs(self, original_tensor, current_molecule, best_molecules, **kwargs):
        """BFS step for the fitting procedure.

        Each step tries to add up to self.max_atom_by_step atoms into maximal density points.

        Arguments:
        tensor -- original 4D tf.tensor [x, y, z, c], where c -- channels.
        current_molecule -- Molecule object to add atoms to.
        best_molecules -- heap with top molecules.
        """
        logger = logging.getLogger(__name__ + '.bfs')
        logger.info("BFS step. Depth:" + str(len(current_molecule)))

        # Tensor of the difference between the original tensor and the density tensor of the current_molecule
        current_tensor = original_tensor - self.molecule_to_tensor(current_molecule)
        current_norm = tf.norm(current_tensor)
        logger.info("Current norm:" + str(current_norm))

        new_molecules = self._add_atoms(current_molecule, current_tensor)
        logger.debug("New molecules: " + str(len(new_molecules)) + ", " + str([str(molecule) for molecule in new_molecules]))

        # Molecules for the next bfs step
        good_molecules = []
        for molecule in new_molecules:
            #molecule, norm = self._optimize_molecule(molecule, original_tensor)
            norm = tf.norm(original_tensor - self.molecule_to_tensor(molecule))
            logger.debug("Molecule: " + str(molecule) + " Norm: " + str(norm))
            if norm < current_norm:
                good_molecules.append((molecule, norm))
        logger.debug("Molecules with lower norm: " + str(len(good_molecules)) + ", " + str([str(molecule[0]) + str(molecule[1]) for molecule in good_molecules]))

        good_molecules = self._select_good_molecules(good_molecules, best_molecules)
        logger.debug("Molecules for the next bfs step: " + str(len(good_molecules)) + ", " + str([str(molecule) for molecule in good_molecules]))
        for molecule in good_molecules:
            self._bfs(original_tensor, molecule, best_molecules, **kwargs)

    def _add_atoms(self, current_molecule, current_tensor):
        """Creates molecules by adding up to self.max_atoms_per_step atoms into the current molecule.

        For example, if self.max_atoms_per_step is 2, it will create molecules with an atom at the top-1
        density point, then add an atom at the top-2 point to the copy of them.

        Arguments:
        current_molecule -- Molecule object to add atoms to.
        current_tensor -- Tensor of the difference between the original tensor and the density tensor of the current_molecule
        """
        logger = logging.getLogger(__name__ + '.add_atoms')

        # Indices of maximal density points
        max_indices = self._get_max_indices(current_tensor)
        logger.debug("Max indices: " + str(len(max_indices)) + ", " + str(max_indices))

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

        # Haven't been tested for non-square shape
        x_dim, y_dim, z_dim = max_channels.shape

        max_channels = tf.reshape(max_channels, [-1])
        _, top_k_indices = tf.math.top_k(max_channels, k=self.max_atom_by_step)
        logger.debug("Top k indices:" + str(len(top_k_indices)) + ", " + str(top_k_indices))

        # For each maximal voxel, we will select voxels with the same density by random.
        # If we have one such voxel, we will select it. But if there are many, it will increase sparseness of new atoms.
        # It's especially important for the first step, since there are many 1.0 voxels.
        result_indices = []
        for max_index in top_k_indices:
            same_good_indices = tf.where(max_channels == max_channels[max_index])
            same_good_indices = tf.random.shuffle(same_good_indices)
            for index in same_good_indices:
                if int(index) not in result_indices:
                    result_indices.append(np.unravel_index(int(index), tensor.shape[:-1]))
                    break
        logger.debug("Result indices: " + str(len(result_indices)) + ", " + str(result_indices))
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
        current_channels = set()
        for i, channel in enumerate(channels):
            if channel > self.threshold:
                current_channels.add(i)
        logger.debug("Current channels: " + str(len(current_channels)) + ", " + str(current_channels))

        possible_atom_types = []
        for atom_type, (_, atom_channels) in self.atom_type_to_descriptor.items():
            if atom_channels.issubset(current_channels):
                possible_atom_types.append(atom_type)
        logger.debug("Possible atom types: " + str(len(possible_atom_types)) + ", " + str(possible_atom_types))

        return possible_atom_types


    def _optimize_molecule(self, molecule, tensor):
        """Optimize coordinates of atoms in the molecule, so its density is looks like the original tensor.
        """
        pass

    def _select_good_molecules(self, good_molecules, best_molecules):
        """Select molecules for the next round of bfs.

        Arguments:
        good_molecules -- list of pairs (molecule, norm of its tensor)
        """
        logger = logging.getLogger(__name__ + ".select_good_molecules")
        new_good_molecules = MoleculeHeap(self.max_molecules_by_step)
        for molecule, norm in good_molecules:
            best_molecules.add(molecule, -norm)
            if len(molecule) < self.max_atoms:
                new_good_molecules.add(molecule, -norm)
        logger.debug("Good molecules: " + str(len(new_good_molecules)) + ", " + str([str(molecule) for molecule in new_good_molecules.get_items()]))
        logger.debug("Best molecules: " + str(len(best_molecules)) + ", " + str([str(molecule) for molecule in best_molecules.get_items()]))
        return new_good_molecules.get_items()


