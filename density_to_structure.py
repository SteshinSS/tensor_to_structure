import tensorflow as tf
import numpy as np
import heapq
from collections import namedtuple
from copy import deepcopy


Coordinates = namedtuple('Coordinates', ['x', 'y', 'z'])


class Atom:
    """Structure for holding coordinates and atom types."""

    def __init__(self, coordinates, atom_type):
        self.coordinate = coordinates
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


def molecule_to_tensor(molecule, **kwargs):
    """Example function for conversion struct to tensor.

    molecule -- Molecule object
    """
    from rdkit import Chem
    from moleculekit.smallmol.smallmol import SmallMol
    from moleculekit.tools.voxeldescriptors import getVoxelDescriptors

    molecule = Chem.AddHs(molecule)
    Chem.AllChem.EmbedMolecule(molecule)
    Chem.AllChem.MMFFOptimizeMolecule(molecule)
    molecule = SmallMol(molecule)
    vox, center, shape = getVoxelDescriptors(molecule, voxelsize=0.5, buffer=1, boxsize=[20, 20, 20], center=[0,0,0])
    vox = vox.reshape((40, 40, 40, 8))
    return vox



class Fitter():
    """Transforms density tensors into structs."""

    def __init__(self):
        """Set parameters."""

        # Number of best molecules we hold
        self.best_max = 10

        # Maximal number of atoms we try to add each bfs step
        self.max_atom_by_step = 2

        # Minimal density for atom detection
        self.threshold = 0.3

        # Maximal number of atoms in molecule
        self.max_atoms = 20

        # Function, that takes Molecule object and return its density tensor.
        self.molecule_to_tensor = molecule_to_tensor

    def tensor_to_molecule(self, tensor, **kwargs):
        """Transform density tensor into struct.

        Arguments:
        tensor -- 4D tf.tensor [x, y, z, c], where c -- channels.
        """

        best_molecules = []
        current_molecule = Molecule()
        self._bfs(tensor, current_molecule, best_molecules)
        return best_molecules

    def _bfs(self, tensor, current_molecule, best_molecules, **kwargs):
        """BFS step for the fitting procedure.

        Each step tries to add up to self.max_atom_by_step atoms into maximal density points.

        Arguments:
        tensor -- original 4D tf.tensor [x, y, z, c], where c -- channels.
        current_molecule -- Molecule object to add atoms to.
        best_molecules -- heap with top self.best_max molecules.
        """

        # Tensor of the difference between the original tensor and the density tensor of the current_molecule
        current_tensor = self.molecule_to_tensor(current_molecule) - tensor
        current_norm = tf.norm(current_tensor)

        new_molecules = self._add_atoms(current_molecule, current_tensor)

        # Molecules for the next bfs step
        good_molecules = []
        for molecule in new_molecules:
            molecule, norm = self._optimize_molecule(molecule, tensor)
            if norm < current_norm:
                good_molecules.append((molecule, norm))

        good_molecules = self._select_best_molecules(good_molecules, best_molecules)

    def _get_max_indexes(self, tensor):
        """Return self.max_atoms_per_step points of maximal value in tensor."""
        pass

    def _add_atoms(self, current_molecule, current_tensor):
        """Creates molecules by adding up to self.max_atoms_per_step atoms into the current molecule.

        For example, if self.max_atoms_per_step is 2, it will create molecules with an atom at the top-1
        density point, then add an atom at the top-2 point to the copy of them.

        Arguments:
        current_molecule -- Molecule object to add atoms to.
        current_tensor -- Tensor of the difference between the original tensor and the density tensor of the current_molecule
        """

        # Indexes of maximal density points
        max_indexes = self._get_max_indexes(current_tensor)

        molecules = []
        last_layer_molecules = [current_molecule]
        for index in max_indexes:
            new_molecules = []
            new_atom_coordinates = self._index_to_coordinates(index)
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

    def _index_to_coordinates(self, index):
        """Converts tensor indexes into grid coordinates."""
        pass

    def _get_possible_atom_types(self, channels):
        """Returns list of possible atoms at point with such channels.

        An atom considered to be possible, if its channels are subset of the channels argument.
        It is because it is possible to has some channel density from the neighboor atom, but it's not possible
        to has low density of channel if atom with such channel is presented.
        """
        pass

    def _optimize_molecule(self, molecule, tensor):
        """Optimize coordinates of atoms in the molecule, so its density is looks like the original tensor.
        """
        pass

    def _select_good_molecules(self, good_molecules):
        """Select molecules for the next round of bfs.

        Arguments:
        good_molecules -- list of pairs (molecule, norm of its tensor)
        """
        pass


