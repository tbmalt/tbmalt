from itertools import combinations_with_replacement
from typing import List
import torch
import numpy as np
from ase.build import molecule
from tbmalt import Geometry
from tbmalt.io import skf

torch.set_default_dtype(torch.float64)

class RepulsiveSplineFeed():

    def __init__(self, spline_data):
        self.spline_data = {frozenset(interaction_pairs):data for interaction_pairs,data in spline_data.items()}

    def __call__(self, geo):
        if geo.atomic_numbers.dim() == 1: #this means its not a batch
            batch_size = 1
        else:
            batch_size = geo.atomic_numbers.size(dim=0)

        indxs = torch.tensor(range(geo.atomic_numbers.size(dim=-1)))
        indx_pairs = torch.combinations(indxs)
        
        Erep = torch.zeros((batch_size))
        for indx_pair in indx_pairs:
            atomnum1 = geo.atomic_numbers[..., indx_pair[0]].reshape((batch_size, ))
            atomnum2 = geo.atomic_numbers[..., indx_pair[1]].reshape((batch_size, ))

            distance = geo.distances[..., indx_pair[0], indx_pair[1]].reshape((batch_size, ))

            for batch_indx in range(batch_size):
                if atomnum1[batch_indx] == 0 or atomnum2[batch_indx] == 0:
                    continue
                Erep[batch_indx] += self._repulsive_calc(distance[batch_indx], atomnum1[batch_indx], atomnum2[batch_indx])

        return Erep

    def _repulsive_calc(self, distance, atomnum1, atomnum2):
        spline = self.spline_data[frozenset((int(atomnum1), int(atomnum2)))]
        tail_start = spline.grid[-1]
        exp_head_cutoff = spline.grid[0]

        if distance < spline.cutoff:
            if distance > tail_start:
                return self._tail(distance, tail_start, spline.tail_coef)
            elif distance > exp_head_cutoff:
                for ind in range(len(spline.grid)):
                    if distance < spline.grid[ind]:
                        return self._spline(distance, spline.grid[ind], spline.spline_coef[ind])
            else:
                return self._exponential_head(distance, spline.exp_coef)
        return 0
   
    @classmethod
    def _exponential_head(cls, distance, coeffs):
        a1 = coeffs[0].clone()
        a2 = coeffs[1].clone()
        a3 = coeffs[2].clone()

        return torch.exp(-a1*distance + a2) + a3

#For some reason this spline method is less accurate than the ones below for spline and tail spline
#    @classmethod 
#    def _spline(cls, distance, start, coeffs):
#        energy = coeffs[0].clone()
#        rDiff = distance - start
#        for coeff in coeffs[1:]:
#            energy += coeff*rDiff
#            rDiff *= rDiff
#        return energy

    @classmethod 
    def _spline(cls, distance, start, coeffs):
        rDiff = distance - start
        energy = coeffs[0] + coeffs[1]*rDiff + coeffs[2]*rDiff**2 + coeffs[3]*rDiff**3
        return energy

    @classmethod 
    def _tail(cls, distance, start, coeffs):
        rDiff = distance - start
        energy = coeffs[0] + coeffs[1]*rDiff + coeffs[2]*rDiff**2 + coeffs[3]*rDiff**3 + coeffs[4]*rDiff**4 + coeffs[5]*rDiff**5
        return energy



    @classmethod
    def from_database(cls, path: str, species: List[int], **kwargs) -> 'RepulsiveSplineFeed':
        interaction_pairs = combinations_with_replacement(species, r=2)
        return cls({interaction_pair: skf.Skf.read(path, interaction_pair).r_spline for interaction_pair in interaction_pairs})

