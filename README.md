# tensor_to_structure
## What is it
There are different ways to represent a chemical molecule in digital formats, like SMILES for 1D text formulas or 2D graphs. There is also a way to capture 3D nature of molecules by density tensors representations[1]. A molecule is placed on a virtual grid, and the density of atoms is calculated within the grid's nodes. Thus space is divided into 3D voxels -- analogous to 2D pixels. 

While there is software for generating voxels (like [libmolgrid](https://gnina.github.io/libmolgrid/) and [moleculekit](https://software.acellera.com/docs/latest/moleculekit/tutorials/voxelization_tutorial.html)), the inverse problem of inferring molecular structure by its density tensor has no analytic solution. There were attempts to solve it by neural networks[1] or optimization procedure[3], but it works in a complex pipeline only. The *tensor_to_structure* is the python library that solves the problem of inferring molecular structure by its density tensor.

## How is it works
The algorithm is computationally heavy, so I prepared two versions. The *tensor_to_structure/v1* is a general-purpose library with minimal assumptions about the problem. The *tensor_to_structure/v2* library was developed for a gaussian mixture model. Use the second only if you model an atom's density by a normal distribution, otherwise use *tensor_to_structure/v1*.

**Warning**: It didn't intend to be published. There are tutorials, and the code is documented, but I shared it for myself only and won't maintain it.

## Literature

 1. Ragoza, M., Hochuli, J., Idrobo, E., Sunseri, J., & Koes, D. R. (2017). Protein–ligand scoring with convolutional neural networks. _Journal of chemical information and modeling_, _57_(4), 942-957.
 2. Ragoza, M., Masuda, T., & Koes, D. R. (2020). Learning a Continuous Representation of 3D Molecular Structures with Deep Generative Models. _arXiv preprint arXiv:2010.08687_.
