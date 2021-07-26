from . import Molecule, Atom, rdkit_to_molecule
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import numpy as np


def draw_bic(results):
    """Plot bic curve for tensor_to_structure() results."""
    x = range(0, len(results))
    y = []
    best_x = 0
    best_y = None
    for i, result in enumerate(results):
        bic = result[1]['bic']
        if bic is None:
            raise RuntimeError('No bic found. Ensure you run tensor_to_structure with calculate_bic=True.')
        if best_y is None:
            best_y = bic
            best_x = i
        else:
            if best_y > bic:
                best_y = bic
                best_x = i
        y.append(bic)
    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    plt.ylabel('Bic')
    plt.xlabel('Total atoms')
    plt.title('Best bic is ' + str(best_y) + ' at ' + str(best_x) + ' atoms')
    plt.show()


def test_get_distances(lhs, rhs):
    """Example function for bond distances."""
    # https://www.chegg.com/homework-help/using-bond-lengths-table-92-p-340-assuming-ideal-geometry-ca-chapter-10-problem-95p-solution-9780077340230-exc
    pair = set([lhs, rhs])

    if pair == set([6, 6]):  # C C
        return [1.27, 1.44, 1.64]
    elif pair == set([6, 7]):  # C N
        return [1.21, 1.37, 1.57]
    elif pair == set([6, 8]):  # C O
        return [1.18, 1.33, 1.53]
    elif pair == set([7, 7]):  # N N
        return [1.16, 1.34, 1.56]
    elif pair == set([7, 8]):  # N O
        return [1.13, 1.32, 1.52]
    elif pair == set([8, 8]):  # O O
        return [0, 1.34, 1.58]
    else:
        raise RuntimeError('Unknown distances for pair ' + str(pair))


_norm_const = np.sqrt(2*np.pi)


def test_voxelizer(molecule, descriptors):
    """Example function for voxelization."""
    all_channels = set()
    for descriptor in descriptors.values():
        for channel in descriptor[1]:
            all_channels.add(channel)

    result = np.zeros((40, 40, 40, len(all_channels)))
    for atom in molecule.atoms:
        std = descriptors[atom.atom_type][0]
        channels = descriptors[atom.atom_type][1]

        coordinates = atom.coordinates
        atom_tensor = np.zeros(result.shape[:-1])
        for idx in np.ndindex(result.shape[:-1]):
            atom_tensor[idx] = (((np.array(idx) - 20) * 0.5 - coordinates) ** 2).sum()
        atom_tensor = np.exp(-1.0 * atom_tensor * std)

        # atom_tensor = np.exp(-atom_tensor/(2.0 * std * std)) / (_norm_const * std)
        for channel in channels:
            result[:, :, :, channel] += atom_tensor
    return result


def get_similarity_density_norm(lhs, rhs, molecule_to_tensor):
    """Create density tensors and return norm of difference between tensors.
    lhs, rhs -- either Molecule objects or rdkit molecules.
    """
    if not isinstance(lhs, Molecule):
        lhs = rdkit_to_molecule(lhs)
    lhs_tensor = molecule_to_tensor(lhs)

    if not isinstance(rhs, Molecule):
        rhs = rdkit_to_molecule(rhs)
    rhs_tensor = molecule_to_tensor(rhs)
    return np.linalg.norm(lhs_tensor - rhs_tensor)


def plot_voxels(voxels, threshold=1.0, channel=-1):
    """Plot voxels with density >= than threshold."""
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(projection='3d')
    ax.voxels(voxels[:, :, :, channel] >= threshold, facecolors='red')


def plot_molecule(molecule, channel=0, zoom=2.3):
    """Plot density tensor. Don't forget to run %gui qt"""
    from mayavi import mlab
    molecule = molecule[:, :, :, channel]
    fig = mlab.figure(size=(500, 500))
    mlab.pipeline.volume(mlab.pipeline.scalar_field(molecule), vmin=0.01)
    fig.scene.camera.zoom(zoom)


def plot_side_by_side(before, after, channel=0, zoom=2.3):
    """Plot two density tensors. Don't forget to run %gui qt"""
    from mayavi import mlab
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
    from mayavi import mlab
    mlab.close(all=True)


def plot_coordinates(lhs, rhs, colors=None):
    """Plot atoms of Molecule objects in 3D plane.
    Do
        %matplotlib notebook
    before."""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    if colors is None:
        colors = dict({
            6: 'tab:gray',
            7: 'tab:brown',
            8: 'tab:blue',
            9: 'tab:orange',
            16: 'tab:olive',
            17: 'tab:green',

        })
    for atom in lhs.atoms:
        coordinates = atom.coordinates
        ax.scatter(coordinates[0], coordinates[1], coordinates[2], marker='o', color=colors[atom.atom_type])
    for atom in rhs.atoms:
        coordinates = atom.coordinates
        ax.scatter(coordinates[0], coordinates[1], coordinates[2], marker='s', color=colors[atom.atom_type])
    plt.show()