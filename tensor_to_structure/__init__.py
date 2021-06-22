import tensorflow as tf
import heapq
from collections import namedtuple
from copy import deepcopy


Coordinates = namedtuple('Coordinates', ['x', 'y', 'z'])
AtomDescriptor = namedtuple('AtomDescriptor', ['radius', 'channels'])


class Atom:
    """Structure for holding coordinates and atom types."""

    def __init__(self, coordinates, atom_type):
        self.coordinates = coordinates
        self.atom_type = atom_type


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


class Fitter:
    """Transforms density tensors into structs.
    Haven't been tested for non-square tensors."""

    def __init__(self, molecule_to_tensor_func, atom_type_to_descriptor):
        """Set parameters."""

        # Number of best molecules we hold
        self.best_max = 10

        # Maximal number of atoms we try to add each bfs step
        self.max_atom_by_step = 2

        # Maximal numbers of best molecules that go to the next bfs step
        self.max_molecules_by_step = 15

        # Minimal density for atom detection
        self.threshold = 0.3

        # Maximal number of atoms in molecule
        self.max_atoms = 20

        # Function, that takes Molecule object and return its density tensor.
        self.molecule_to_tensor = molecule_to_tensor_func

        # Grid resolution
        self.grid_resolution = 0.5

        # Dict from atom type to AtomDescriptors
        self.atom_type_to_descriptor = atom_type_to_descriptor

    def tensor_to_molecule(self, tensor, **kwargs):
        """Transform density tensor into struct.

        Arguments:
        tensor -- 4D tf.tensor [x, y, z, c], where c -- channels.
        """
        best_molecules = MoleculeHeap(self.best_max)
        current_molecule = Molecule()
        self._bfs(tensor, current_molecule, best_molecules, **kwargs)
        return best_molecules

    def _bfs(self, original_tensor, current_molecule, best_molecules, **kwargs):
        """BFS step for the fitting procedure.

        Each step tries to add up to self.max_atom_by_step atoms into maximal density points.

        Arguments:
        tensor -- original 4D tf.tensor [x, y, z, c], where c -- channels.
        current_molecule -- Molecule object to add atoms to.
        best_molecules -- heap with top molecules.
        """

        # Tensor of the difference between the original tensor and the density tensor of the current_molecule
        current_tensor = self.molecule_to_tensor(current_molecule) - original_tensor
        current_norm = tf.norm(current_tensor)

        new_molecules = self._add_atoms(current_molecule, current_tensor)

        # Molecules for the next bfs step
        good_molecules = []
        for molecule in new_molecules:
            molecule, norm = self._optimize_molecule(molecule, original_tensor)
            if norm < current_norm:
                good_molecules.append((molecule, norm))

        good_molecules = self._select_good_molecules(good_molecules, best_molecules)
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

        # Indices of maximal density points
        max_indices = self._get_max_indices(current_tensor)

        molecules = []
        last_layer_molecules = [current_molecule]
        for index in max_indices:
            new_molecules = []
            new_atom_coordinates = self._index_to_coordinates(index, current_tensor.shape[:-1])
            possible_atom_types = self._get_possible_atom_types(current_tensor[index])
            for atom_type in possible_atom_types:
                new_atom = Atom(new_atom_coordinates, atom_type)
                for molecule in last_new_molecules:
                    new_molecule = deepcopy(molecule)
                    new_molecule.add_atom(new_atom)
                    new_molecules.append(new_molecule)
            last_new_molecules = new_molecules
            new_molecules.extend(new_molecules)
        return molecules

    def _get_max_indices(self, tensor):
        """Return self.max_atoms_by_step points of maximal value in tensor.

        Haven't been tested with non-square tensors."""

        # Since we are looking for different xyz coordinates, select only maximum by channel axis
        max_channels = tf.math.reduce_max(tensor, axis=-1)

        # Haven't been tested for non-square shape
        x_dim, y_dim, z_dim = max_channels.shape

        max_channels = tf.reshape(max_channels, [-1])
        _, top_k_indices = tf.math.top_k(max_channels, k=self.max_atom_by_step)

        # For each maximal voxel, we will select voxels with the same density by random.
        # If we have one such voxel, we will select it. But if there are many, it will increase sparseness of new atoms.
        # It's especially important for the first step, since there are many 1.0 voxels.
        flat_indices = []
        for max_index in top_k_indices:
            same_good_indices = tf.where(max_channels == max_channels[max_index])
            same_good_indices = tf.random.shuffle(same_good_indices)
            for index in same_good_indices:
                if index not in flat_indices:
                    flat_indices.append(index)
                    break

        # Return flat indices into 3D
        result_indices = []
        for index in flat_indices:
            index = int(index)
            idx_z = index % z_dim
            index //= z_dim
            idx_y = index % y_dim
            index //= y_dim
            idx_x = index
            result_indices.append((idx_x, idx_y, idx_z))
        return result_indices

    def _index_to_coordinates(self, index, tensor_shape):
        """Converts tensor indices into grid coordinates."""
        xyz = index - (tensor_shape / 2)
        return Coordinates(*xyz)

    def _get_possible_atom_types(self, channels):
        """Returns list of possible atoms at point with such channels.

        An atom considered to be possible, if its channels are subset of the channels argument.
        It is because it is possible to has some channel density from the neighboor atom, but it's not possible
        to has low density of channel if atom with such channel is presented.
        """
        current_channels = set()
        for i, channel in enumerate(channels):
            if channel:
                current_channels.add(i)

        possible_atom_types = []
        for atom_type, (_, atom_channels) in self.atom_type_to_descriptor:
            if current_channels.issubset(atom_channels):
                possible_atom_types.append(atom_type)

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
        new_good_molecules = MoleculeHeap(self.max_molecules_by_step)
        for molecule in good_molecules:
            new_tensor = self.molecule_to_tensor(molecule)
            new_norm = tf.norm(new_tensor)
            best_molecules.add(molecule, -new_norm)
            new_good_molecules.add(molecule, -new_norm)
        return new_good_molecules.get_items()


