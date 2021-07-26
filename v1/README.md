# tensor_to_structure/v1
## What is it
This is general python library for inferring molecular structure by its density tensor. See main [README.md](https://github.com/SteshinSS/tensor_to_structure/blob/main/README.md).

## How to use it
Drop tensor_to_structure folder in your working directory, do:

    import tensor_to_structure as tts
    import tensor_to_structure.utilities as tts_utilities

And read the tutorial.

## Documentation
The tutorial show the default usage. Here is cheat-sheet:

    tts.Coordinates -- namedtuple with coordinates
    tts.AtomDescriptor -- namedtuple with (atom_radius, set(channels))
    tts.Atom -- Structure for holding coordinates and atom types
    tts.Molecule -- Structure for holding list of atoms
    tts.Fitter -- Class for transforming density tensors to molecules
	    tts.Fitter.tensor_to_molecule() -- main function
	  
	tts_utilities.rdkit_to_molecule() -- get a Molecule object out of rdkit Mol
	tts_utilities.molecule_to_rdkit() -- get rdkit Mol out of a Molecule object
	tts_utilities.plot_molecule() -- plot molecule by mayavi
	tts_utilities.plot_side_by_side() -- plot two molecules together

Read doc strings and the code if bogged. It is not that big.

## Inspiration
This code was inspired by [liGAN implementation](https://github.com/mattragoza/liGAN). 
    
