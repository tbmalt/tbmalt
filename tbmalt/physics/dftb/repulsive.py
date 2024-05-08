from itertools import combinations_with_replacement
from typing import List, Optional, Dict, Tuple, Union
import torch
import numpy as np
from ase.build import molecule
from tbmalt import Geometry
from tbmalt.io import skf

Tensor = torch.Tensor

torch.set_default_dtype(torch.float64)

class RepulsiveSplineFeed():
    r"""Repulsive Feed using splines for DFTB calculations. Data is derived from a skf file.

    This feed uses splines to calculate the repulsive energy of a Geometry in the way it is defined for DFTB.

    Arguments:
        spline_data: Dictionary containing the the tuples of atomic number pairs as keys and the corresponding spline data as values.
    """

    def __init__(self, spline_data: Dict[Tuple, Tensor]):
        self.spline_data = {frozenset(interaction_pairs):data for interaction_pairs,data in spline_data.items()}

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by `RepulsiveSplineFeed` object."""
        return list(self.spline_data.values())[0].grid.dtype

    @property
    def device(self) -> torch.device:
        """The device on which the `RepulsiveSplineFeed` object resides."""
        return list(self.spline_data.values())[0].grid.device

    def __call__(self, geo: Union[Geometry, Tensor]) -> Tensor:
        r"""Calculate the repulsive energy of a Geometry.

        Arguments:
            geo: Geometry object(s) for which the repulsive energy should be calculated. Either a single Geometry object or a batch of Geometry objects.

        Returns:
            Erep: The repulsive energy of the Geometry object(s).
        """
        if geo.atomic_numbers.dim() == 1: #this means it is not a batch
            batch_size = 1
        else:
            batch_size = geo.atomic_numbers.size(dim=0)

        indxs = torch.tensor(range(geo.atomic_numbers.size(dim=-1)), device=self.device)
        indx_pairs = torch.combinations(indxs)
        
        Erep = torch.zeros((batch_size), device=self.device, dtype=self.dtype)
        for indx_pair in indx_pairs:
            atomnum1 = geo.atomic_numbers[..., indx_pair[0]].reshape((batch_size, ))
            atomnum2 = geo.atomic_numbers[..., indx_pair[1]].reshape((batch_size, ))

            distance = geo.distances[..., indx_pair[0], indx_pair[1]].reshape((batch_size, ))

            for batch_indx in range(batch_size):
                if atomnum1[batch_indx] == 0 or atomnum2[batch_indx] == 0:
                    continue
                Erep[batch_indx] += self._repulsive_calc(distance[batch_indx], atomnum1[batch_indx], atomnum2[batch_indx])

        return Erep

    def _repulsive_calc(self, distance: Tensor, atomnum1: Union[Tensor, int], atomnum2: Union[Tensor, int]) -> Tensor:
        """Calculate the repulsive energy contribution between two atoms.

        Arguments:
            distance: The distance between the two atoms.
            atomnum1: The atomic number of the first atom.
            atomnum2: The atomic number of the second atom.

        returns:
            Erep: The repulsive energy contribution between the two atoms.
        """
        spline = self.spline_data[frozenset((int(atomnum1), int(atomnum2)))]
        tail_start = spline.grid[-1]
        exp_head_cutoff = spline.grid[0]

        if distance < spline.cutoff:
            if distance > tail_start:
                return self._tail(distance, tail_start, spline.tail_coef)
            elif distance > exp_head_cutoff:
                for ind in range(len(spline.grid)):
                    if distance < spline.grid[ind]:
                        return self._spline(distance, spline.grid[ind-1], spline.spline_coef[ind-1])
            else:
                return self._exponential_head(distance, spline.exp_coef)
        return 0
   
    @classmethod
    def _exponential_head(cls, distance: Tensor, coeffs: Tensor) -> Tensor:
        r"""Exponential head calculation of the repulsive spline. 

        Arguments:
            distance: The distance between the two atoms.
            coeffs: The coefficients of the exponential head.

        Returns:
            energy: The energy value of the exponential head.
                The energy is calculated as :math:`\exp(-coeffs[0] \cdot r + coeffs[1]) + coeffs[2]`.
        """
        a1 = coeffs[0].clone()
        a2 = coeffs[1].clone()
        a3 = coeffs[2].clone()

        return torch.exp(-a1*distance + a2) + a3

    @classmethod 
    def _spline(cls, distance: Tensor, start: Tensor, coeffs: Tensor) -> Tensor:
        r"""3rd order polynomial Spline calculation of the repulsive spline.

        Arguments:
            distance: The distance between the two atoms.
            start: The start of the spline segment.
            coeffs: The coefficients of the polynomial.

        Returns:
            energy: The energy value of the spline segment.
                The energy is calculated as :math:`coeffs[0] + coeffs[1]*(distance - start) + coeffs[2]*(distance - start)^2 + coeffs[3]*(distance - start)^3`.
        """
        rDiff = distance - start
        energy = coeffs[0] + coeffs[1]*rDiff + coeffs[2]*rDiff**2 + coeffs[3]*rDiff**3
        return energy

    @classmethod 
    def _tail(cls, distance: Tensor, start: Tensor, coeffs: Tensor) -> Tensor:
        r"""5th order polynomial trailing tail calculation of the repulsive spline.

        Arguments:
            distance: The distance between the two atoms.
            start: The start of the trailing tail segment.
            coeffs: The coefficients of the polynomial.
        
        Returns:
            energy: The energy value of the tail.
                The energy is calculated as :math:`coeffs[0] + coeffs[1]*(distance - start) + coeffs[2]*(distance - start)^2 + coeffs[3]*(distance - start)^3 + coeffs[4]*(distance - start)^4 + coeffs[5]*(distance - start)^5`.
        """
        rDiff = distance - start
        energy = coeffs[0] + coeffs[1]*rDiff + coeffs[2]*rDiff**2 + coeffs[3]*rDiff**3 + coeffs[4]*rDiff**4 + coeffs[5]*rDiff**5
        return energy



    @classmethod
    def from_database(cls, path: str, species: List[int], dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> 'RepulsiveSplineFeed':
        r"""Instantiate instance from a HDF5 database of Slater-Koster files.

        Instantiate a `RepulsiveSplineFeed` instance from a HDF5 database for the specified elements.

        Arguments:
            path: Path to the HDF5 file from which the repulsive interaction data should be taken.
            species: List of atomic numbers for which the repulsive spline data should be read.
            device: Device on which the feed object and its contents resides.

        Returns:
            repulsive_feed: A `RepulsiveSplineFeed` instance.

        Notes:
            This method will not instantiate `SkFeed` instances directly
            from human readable skf files, or a directory thereof. Thus, any
            such files must first be converted into their binary equivalent.
            This reduces overhead & file format error instabilities. The code
            block provide below shows how this can be done:

            >>> from tbmalt.io.skf import Skf
            >>> Zs = ['H', 'C', 'Au', 'S']
            >>> for file in [f'{i}-{j}.skf' for i in Zs for j in Zs]:
            >>>     Skf.read(file).write('my_skf.hdf5')
        """
        interaction_pairs = combinations_with_replacement(species, r=2)
        return cls({interaction_pair: skf.Skf.read(path, interaction_pair, device=device, dtype=dtype).r_spline for interaction_pair in interaction_pairs})

