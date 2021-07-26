# tensor_to_structure/v2
## What is it
This is python library for inferring molecular structure by its density tensor with normaly distributed densities. See main [README.md](https://github.com/SteshinSS/tensor_to_structure/blob/main/README.md).

## How to use it
Drop tensor_to_structure folder in your working directory, do:

    import tts
    import tts.utils

And read the tutorial.

## Documentation
The tutorial show the default usage. Here is cheat-sheet:

    tts.Coordinates -- namedtuple with coordinates
    tts.AtomDescriptor -- namedtuple with (atom_radius, set(channels))
    tts.Atom -- Structure for holding coordinates and atom types
    tts.Molecule -- Structure for holding list of atoms
    
    tts.tensor_to_structure() -- main function
    tts.get_bonds() -- set bonds for estimated structure
	tts.rdkit_to_molecule() -- get a Molecule object out of rdkit Mol
	tts.molecule_to_rdkit() -- get rdkit Mol out of a Molecule object
	tts.estimate_total_atoms() -- quick and dirty estimation of total number of atoms in tensor
	tts.get_best_rms() -- return rms metric
	tts.get_ecfp_similarity() -- return tanimoto metric

	tts_utilities.plot_molecule() -- plot molecule by mayavi
	tts_utilities.plot_side_by_side() -- plot two molecules together
	tts_utilities.close_mlab() -- close all mayavi windows
	tts_utilities.draw_bic() -- plot bic curve for tensor_to_structure() result
	tts_utilities.get_similarity_density_norm() -- calculate norm of difference between two tensors
	tts_utilites.plot_coordinates() -- plot coordinates in 3D plane
	

Read doc strings and the code if bogged. It is not that big.

## Inspiration
This code was inspired by [liGAN implementation](https://github.com/mattragoza/liGAN). 
    
