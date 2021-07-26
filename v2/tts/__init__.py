import numpy as np
from collections import namedtuple
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import logging


Coordinates = namedtuple('Coordinates', ['x', 'y', 'z'])
AtomDescriptor = namedtuple('AtomDescriptor', ['std', 'channels'])


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


def get_best_rms(lhs, rhs):
    """Align molecules and return RMS distance."""
    params = AllChem.AdjustQueryParameters()
    params.makeAtomsGeneric = True
    params.makeBondsGeneric = True
    lhs_generic = AllChem.AdjustQueryProperties(lhs, params)
    rhs_generic = AllChem.AdjustQueryProperties(rhs, params)
    return AllChem.GetBestRMS(lhs_generic, rhs_generic)


def get_ecfp_similarity(lhs, rhs, radius=None):
    """Calculate Morgan fingerprints and return Tanimoto similarity.

    radius -- iterable with radius of Morgan fingerprints. Default: [3, 4, 5]. """
    similarity = []

    if radius is None:
        radius = range(2, 5)

    for i in radius:
        lhs_fingerprint = AllChem.GetMorganFingerprint(lhs, i)
        rhs_fingerprint = AllChem.GetMorganFingerprint(rhs, i)
        similarity.append(DataStructs.DiceSimilarity(lhs_fingerprint, rhs_fingerprint))
    return similarity


def construct_molecule(tensor, coordinates, descriptors, threshold=0.7, voxel_step=0.5):
    """Construct Molecule object from coordinates.

    tensor -- original density tensor.
    coordinates -- atom coordinates.
    descriptors -- map from atom type to atom descriptor.
    voxel_step -- size of tensor's grid in A.
    threshold -- minimal value of channel to be considered."""
    logger = logging.getLogger(__name__)
    molecule = Molecule()
    for xyz in coordinates:
        idx = np.around(xyz).astype(int)
        channels = set()
        values = tensor[idx[0], idx[1], idx[2]]
        for channel, value in enumerate(values):
            if value > threshold:
                channels.add(channel)

        best_atom = None
        best_value = 0.0
        for atom_type, (_, atom_channels) in descriptors.items():
            if atom_channels.issubset(channels):
                current_value = 0.0
                for c in atom_channels:
                    current_value += values[c]
                if current_value > best_value:
                    best_atom = atom_type
                    best_value = current_value

        logger.debug("Molecule at " + str(xyz) + " has channel: " + str(values) + ". Selected atom type: " + str(best_atom))
        xyz = (xyz - np.array(tensor.shape[:-1]) / 2) * voxel_step
        atom = Atom(xyz, best_atom)
        molecule.add_atom(atom)
    return molecule


def sample_tensor(tensor, trials=50, channel=0):
    """Sample points from tensor.

    tensor -- tensor to sample from.
    trials -- number of trials in Bernoulli sampling.
    channel -- channel of the tensor."""
    points = []
    shape = tensor.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                prob = tensor[i, j, k, channel]
                result = np.random.binomial(trials, prob)
                for _ in range(result):
                    points.append(np.array((i, j, k)))
    return np.stack(points)


def tensor_to_structure(tensor,
                        descriptors,
                        voxel_step,
                        min_atoms=1,
                        max_atoms=10,
                        trials=70,
                        max_iter=500,
                        n_init=10,
                        tol=0.001,
                        reg_covar=1e-6,
                        init_params='kmeans',
                        threshold=0.1,
                        bayesian=False,
                        calculate_bic=False,
                        verbose=0):
    """
    Converts density tensor to Molecule object without bonds.
    Returns list with pairs. i'th pair contain fitted molecule with min_atoms + i atoms
    and info dict.

    Usage:
        results = tensor_to_structure(tensor, descriptors)

        five_atom_molecule, info = results[4][0], results[4][1]
        for key, value in info.items():
            print(key, value)


    Arguments:
        tensor -- 4D density tensor [x, y, z, channels]
        descriptors -- map from atom types to atom descriptors
        voxel_step -- size of tensor's grid in A.
        min_atoms -- minimal number of atoms in the tensor
        max_atoms -- maximal number of atoms in the tensor
        trials -- number of Bernoulli trials for sampling
        threshold -- minimal value of channel to be considered as atom
        verbose -- set > 0 to print info

        EM-algorithm parameters.
        See more: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.fit
            max_iter -- The number of EM iterations to perform.
            n_init -- The number of initializations to perform. The best results are kept.
            tol -- The convergence threshold.
            reg_covar -- Non-negative regularization added to the diagonal of covariance.
            init_params -- The method used to initialize.
            bayesian -- Use BayesianGaussianMixture instead.
            calculate_bic -- return Bayesian information criterion (only with bayesian = False)
    """
    normalized_tensor = tensor / tensor.max()
    sampled_points = sample_tensor(normalized_tensor, trials)

    result = []
    for n_atoms in range(min_atoms, max_atoms):
        if verbose > 0:
            print("Fitting", n_atoms, "atoms...")
        if bayesian:
            gmm = BayesianGaussianMixture(n_components=n_atoms,
                                          covariance_type='spherical',
                                          max_iter=max_iter,
                                          n_init=n_init,
                                          weight_concentration_prior_type='dirichlet_distribution',
                                          covariance_prior=1.0)
            gmm.fit(sampled_points)
        else:
            weights = [1.0 / n_atoms] * n_atoms
            gmm = GaussianMixture(n_components=n_atoms,
                                  covariance_type='spherical',
                                  max_iter=max_iter,
                                  n_init=n_init,
                                  tol=tol,
                                  reg_covar=reg_covar,
                                  init_params=init_params,
                                  weights_init=weights)
            gmm.fit(sampled_points)
        info = dict({
            'coordinates': gmm.means_,
            'weights': gmm.weights_,
            'covariance': gmm.covariances_,
            'bic': None
        })
        if calculate_bic:
            if bayesian:
                raise RuntimeError("Can't calculate bin in bayesian mode. Use bayesian=False.")
            else:
                info['bic'] = gmm.bic(sampled_points)

        molecule = construct_molecule(tensor, gmm.means_, descriptors, threshold=threshold, voxel_step=voxel_step)
        result.append((molecule, info))

    if verbose > 0:
        if calculate_bic:
            best_bic = result[0][1]['bic']
            best_n_atoms = 1
            for i, res in enumerate(result):
                if res[1]['bic'] < best_bic:
                    best_bic = res[1]['bic']
                    best_n_atoms = i
            print('Best bic at ' + str(best_n_atoms + min_atoms))

    return result


def estimate_total_atoms(tensor, radius=1, epsilon=0.1):
    """
    Greedy algorithm to estimate number of atoms in the density tensor.
    Select voxels with density: (density < 1.0 + epsilon) & (density > 1.0 - epsilon)
    and delete random spheres of certain radius until there is none such voxels."""
    n_atoms = 0

    # Take occupancy channel
    tensor = tensor[:, :, :, 0]
    tensor = (tensor > 1.0 - epsilon) & (tensor < 1.0 + epsilon)

    # Indices of voxels with values around 1.0
    idx = np.where(tensor)
    while idx[0].shape[0] > 0:

        # Select random voxel
        total_idx = idx[0].shape[0]
        atom_idx = np.random.randint(0, total_idx)
        atom_center = np.array((idx[0][atom_idx], idx[1][atom_idx], idx[2][atom_idx]))

        # Remove neighbour voxels
        for i in range(total_idx):
            true_index = np.array((idx[0][i], idx[1][i], idx[2][i]))
            if np.linalg.norm(true_index - atom_center) < radius:
                tensor[true_index] = False
        tensor = (tensor > 1.0 - epsilon) & (tensor < 1.0 + epsilon)
        idx = np.where(tensor)
        n_atoms += 1
    return n_atoms


def get_closest_atom(rdkit_mol, distance_matrix, atom_idx, valences):
    """Return closest atom with free valence.

    Arguments:
        rdkit_mol -- molecule in rdkit format
        distance_matrix -- matrix of atom distances
        atom_idx -- index of atom to search closest to.
        valences -- map from atom type to maximum valence.
    """
    closest_atom = None
    closest_atom_idx = None
    closest_distance = None
    for i, other_atom in enumerate(rdkit_mol.GetAtoms()):
        # If it is not the same atom and its degree is not maximal
        if (i != atom_idx) and other_atom.GetDegree() < valences[other_atom.GetAtomicNum()]:
            # If there are no bond yet and it is close
            if (rdkit_mol.GetBondBetweenAtoms(i, atom_idx) is None) and (
                    closest_distance is None or closest_distance > distance_matrix[i, atom_idx]):
                closest_atom = other_atom
                closest_atom_idx = i
                closest_distance = distance_matrix[i, atom_idx]
    return closest_atom, closest_atom_idx, closest_distance


def find_best_bond(bond_distances, closest_distance):
    """Return type of bond. Return None if atoms are too far.

    bond_distances -- list of maximal distances:
        [triple, double, single]
    closest_distance -- distance between atoms.
    """
    min_idx = None
    for i, distance in enumerate(bond_distances):
        if distance > closest_distance:
            min_idx = i

    if min_idx == 0:
        return Chem.rdchem.BondType.TRIPLE
    elif min_idx == 1:
        return Chem.rdchem.BondType.DOUBLE
    elif min_idx == 2:
        return Chem.rdchem.BondType.SINGLE
    else:
        return None


def molecule_to_rdkit(molecule):
    """Converts Molecule object to rdkit Mol."""
    rdkit_mol = Chem.rdchem.RWMol()
    for atom in molecule.atoms:
        rdkit_mol.AddAtom(Chem.rdchem.Atom(atom.atom_type))

    conformer = Chem.AllChem.Conformer(len(molecule))
    conformer.Set3D(True)
    for i, atom in enumerate(molecule.atoms):
        conformer.SetAtomPosition(i, atom.coordinates)

    rdkit_mol.AddConformer(conformer)
    return rdkit_mol


def get_bonds(molecule, get_distances, valences):
    """Set bonds to Molecule object. Return rdkit molecule.

    Arguments:
        molecule -- Molecule object
        get_distances -- function f(lhs, rhs). Takes atom types and return maximal bond distances:
            [triple_bond, double_bond, single_bond]
        valences -- map from atom type to maximal valence.
    """
    rdkit_mol = molecule_to_rdkit(molecule)

    distance_matrix = Chem.rdmolops.Get3DDistanceMatrix(rdkit_mol)
    for i, atom in enumerate(rdkit_mol.GetAtoms()):
        # While the valence is not maximal
        while atom.GetDegree() < valences[atom.GetAtomicNum()]:
            # Find the closest atom
            closest_atom, closest_atom_idx, closest_distance = get_closest_atom(rdkit_mol,
                                                                                distance_matrix,
                                                                                i,
                                                                                valences)
            # No atoms with free valence
            if closest_atom is None:
                break

            # Find best bond
            bond_distances = get_distances(atom.GetAtomicNum(), closest_atom.GetAtomicNum())
            bond = find_best_bond(bond_distances, closest_distance)

            # The closest atom is too far
            if bond is None:
                break

            rdkit_mol.AddBond(i, closest_atom_idx, bond)

    Chem.SanitizeMol(rdkit_mol)
    return rdkit_mol


def rdkit_to_molecule(rdkit_molecule):
    """Converts rdkit molecule to Molecule object."""
    conformer = rdkit_molecule.GetConformers()[0]
    rdkit_positions = conformer.GetPositions()

    atoms = []
    for i, rd_atom in enumerate(rdkit_molecule.GetAtoms()):
        if rd_atom.GetAtomicNum() != 1:
            atoms.append(Atom(rdkit_positions[i], rd_atom.GetAtomicNum()))
    return Molecule(atoms)


if __name__=='__main__':
    import tts.utils
    print('Running smoke tests...')
    molecule = Chem.MolFromSmiles('CCC')
    molecule = Chem.AddHs(molecule)
    Chem.AllChem.EmbedMolecule(molecule)
    Chem.AllChem.UFFOptimizeMolecule(molecule)
    molecule = Chem.RemoveHs(molecule)

    my_molecule = rdkit_to_molecule(molecule)
    descriptors = {6: AtomDescriptor(1.0, {0})}
    original_tensor = tts.utils.test_voxelizer(my_molecule, descriptors)
    results = tensor_to_structure(original_tensor,
                                  descriptors,
                                  voxel_step=0.5,
                                  min_atoms=3,
                                  max_atoms=4,
                                  )

    maximal_valence = {6: 4}
    def get_distance_c(lhs, rhs):
        return [1.27, 1.44, 1.64]

    result = get_bonds(results[0][0], get_distance_c, maximal_valence)
    rms = get_best_rms(result, molecule)
    ecpf = get_ecfp_similarity(result, molecule)
    print('RMS:', rms, 'ECPF:', ecpf)
    assert rms < 0.5
    assert ecpf[0] > 0.9
    assert ecpf[1] > 0.9
    assert ecpf[2] > 0.9
    print('Smoke test successfully passed')
