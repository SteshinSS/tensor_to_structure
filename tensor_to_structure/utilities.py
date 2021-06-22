import numpy as np
from tensor_to_structure import AtomDescriptor, Molecule, Atom
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt

atom_type_to_descriptor = {
    1: AtomDescriptor(0, set([])),
    6: AtomDescriptor(1.4, {3, 4}),
    7: AtomDescriptor(1.35, {2, 4}),
    8: AtomDescriptor(1.3, {2, 3, 4}),
    9: AtomDescriptor(1.25, {1, 4}),
    15: AtomDescriptor(1.6, {1, 3, 4}),
    16: AtomDescriptor(1.6, {1, 2, 4}),
    17: AtomDescriptor(1.58, {1, 2, 3, 4}),
    35: AtomDescriptor(1.7, {0, 4}),
    53: AtomDescriptor(2, {0, 1, 4}),
}


def molecule_to_tensor(molecule, **kwargs):
    """Example function for conversion struct to tensor.

    molecule -- Molecule object.
    """
    total_channels = 5
    molecule_channels = np.zeros((len(molecule), total_channels))
    for i, atom in enumerate(molecule.atoms):
        radius, atom_channels = atom_type_to_descriptor[atom.atom_type]
        for channel in atom_channels:
            molecule_channels[i][channel] = radius

    coords = np.zeros((len(molecule), 3, 1))
    for i, atom in enumerate(molecule.atoms):
        coords[i][0][0], coords[i][1][0], coords[i][2][0] = atom.coordinates

    vox, center, shape = getVoxelDescriptors(None,
                                             voxelsize=0.5,
                                             buffer=1,
                                             boxsize=[20, 20, 20],
                                             center=[0, 0, 0],
                                             userchannels=molecule_channels,
                                             usercoords=coords)
    vox = vox.reshape((40, 40, 40, total_channels))
    return vox


def rdkit_to_molecule(rdkit_molecule):
    rdkit_molecule = Chem.AddHs(rdkit_molecule)
    Chem.AllChem.EmbedMolecule(rdkit_molecule)
    Chem.AllChem.UFFOptimizeMolecule(rdkit_molecule)
    conformer = rdkit_molecule.GetConformers()[0]
    rdkit_positions = conformer.GetPositions()

    atoms = []
    for i, rd_atom in enumerate(rdkit_molecule.GetAtoms()):
        atoms.append(Atom(rdkit_positions[i], rd_atom.GetAtomicNum()))
    return Molecule(atoms)


def molecule_to_rdkit(molecule):
    rdkit_mol = Chem.rdchem.RWMol()
    for atom in molecule.atoms:
        rdkit_mol.AddAtom(Chem.rdchem.Atom(atom.atom_type))
    Chem.AllChem.EmbedMolecule(rdkit_mol)
    for i in range(len(rdkit_mol.GetAtoms())):
        rdkit_mol.GetConformer(0).SetAtomPosition(i, molecule.atoms[i].coordinates)
    return rdkit_mol


def rdkit_to_tensor(molecule):
    from rdkit import Chem
    from moleculekit.smallmol.smallmol import SmallMol
    from moleculekit.tools.voxeldescriptors import getVoxelDescriptors

    molecule = Chem.AddHs(molecule)
    Chem.AllChem.EmbedMolecule(molecule)
    Chem.AllChem.MMFFOptimizeMolecule(molecule)
    molecule = SmallMol(molecule)
    vox, center, shape = getVoxelDescriptors(molecule, voxelsize=0.5, buffer=1, boxsize=[20, 20, 20], center=[0, 0, 0])
    vox = vox.reshape((40, 40, 40, 8))
    return vox


def plot_voxels(voxels, threshold=1.0):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(projection='3d')
    ax.voxels(voxels[:, :, :, -1] == threshold, facecolors='red')
