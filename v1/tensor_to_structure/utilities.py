from tensor_to_structure import Molecule, Atom
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from mayavi import mlab


def rdkit_to_molecule(rdkit_molecule):
    """Converts rdkit molecule to Molecule object."""
    rdkit_molecule = Chem.AddHs(rdkit_molecule)
    Chem.AllChem.EmbedMolecule(rdkit_molecule)
    Chem.AllChem.UFFOptimizeMolecule(rdkit_molecule)
    conformer = rdkit_molecule.GetConformers()[0]
    rdkit_positions = conformer.GetPositions()

    atoms = []
    for i, rd_atom in enumerate(rdkit_molecule.GetAtoms()):
        if rd_atom.GetAtomicNum() != 1:
            atoms.append(Atom(rdkit_positions[i], rd_atom.GetAtomicNum()))
    return Molecule(atoms)


def molecule_to_rdkit(molecule):
    """Converts Molecule object to rdkit Mol."""
    rdkit_mol = Chem.rdchem.RWMol()
    for atom in molecule.atoms:
        rdkit_mol.AddAtom(Chem.rdchem.Atom(atom.atom_type))
    Chem.AllChem.EmbedMolecule(rdkit_mol)
    for i in range(len(rdkit_mol.GetAtoms())):
        rdkit_mol.GetConformer(0).SetAtomPosition(i, molecule.atoms[i].coordinates)
    return rdkit_mol


def plot_voxels(voxels, threshold=1.0, channel=-1):
    """Plot voxels with density >= than threshold."""
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(projection='3d')
    ax.voxels(voxels[:, :, :, channel] >= threshold, facecolors='red')


def plot_molecule(molecule, channel=-1, zoom=2.3):
    """Plot density tensor. Don't forget to run %gui qt"""
    molecule = molecule[:, :, :, channel]
    fig = mlab.figure(size=(500, 500))
    mlab.pipeline.volume(mlab.pipeline.scalar_field(molecule), vmin=0.01)
    fig.scene.camera.zoom(zoom)


def plot_side_by_side(before, after, channel=-1, zoom=2.3):
    """Plot two density tensors. Don't forget to run %gui qt"""
    before = before[:, :, :, channel]
    after = after[:, :, :, channel]
    vmin = 0.01
    fig_before = mlab.figure('Before', size=(500, 500))
    mlab.pipeline.volume(mlab.pipeline.scalar_field(before), vmin=vmin)

    fig_after = mlab.figure('After', size=(500, 500))
    mlab.pipeline.volume(mlab.pipeline.scalar_field(after), vmin=vmin)

    fig_before.scene.camera.zoom(zoom)
    fig_after.scene.camera.zoom(zoom)
    mlab.sync_camera(fig_before, fig_after)
    mlab.sync_camera(fig_after, fig_before)


def close_mlab():
    """Close all mayavi windows."""
    mlab.close(all=True)

