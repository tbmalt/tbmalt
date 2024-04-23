from itertools import combinations_with_replacement
from typing import List
import torch
import numpy as np
from ase.build import molecule
from tbmalt import Geometry
from tbmalt.io import skf

torch.set_default_dtype(torch.float64)

debugprint = 280

class RepulsiveSplineFeed():
    """Repulsive Feed using splines. Data is derived from a skf file.

    """

    def __init__(self, spline_data):
        self.spline_data = {frozenset(interaction_pairs):data for interaction_pairs,data in spline_data.items()}

    def __call__(self, geo):
        print('Called RepulsiveSplineFeed for geometry: \n', geo)
        print('#'*debugprint)
        print('#'*debugprint)
        if geo.atomic_numbers.dim() == 1: #this means its not a batch
            batch_size = 1
        else:
            batch_size = geo.atomic_numbers.size(dim=0)

        indxs = torch.tensor(range(geo.atomic_numbers.size(dim=-1)))
        indx_pairs = torch.combinations(indxs)
        
        Erep = torch.zeros((batch_size))
        print('Initial Erep: ', Erep)
        print('-'*debugprint)
        for indx_pair in indx_pairs:
            atomnum1 = geo.atomic_numbers[..., indx_pair[0]].reshape((batch_size, ))
            atomnum2 = geo.atomic_numbers[..., indx_pair[1]].reshape((batch_size, ))
            print('calculation for atomnum1: ', atomnum1, ' and atomnum2: ', atomnum2)

            distance = geo.distances[..., indx_pair[0], indx_pair[1]].reshape((batch_size, ))
            print('Distance: ', distance)
            print('-'*debugprint)

            for batch_indx in range(batch_size):
                if atomnum1[batch_indx] == 0 or atomnum2[batch_indx] == 0:
                    continue
                Erep[batch_indx] += self._repulsive_calc(distance[batch_indx], atomnum1[batch_indx], atomnum2[batch_indx])
            print('Erep after pair: ', indx_pair, ' \n\tErep: ', Erep)
            print('#'*debugprint)

        return Erep

    def _repulsive_calc(self, distance, atomnum1, atomnum2):
        print('Called _repulsive_calc for \n\tatomnum1: ', atomnum1, '\n\tatomnum2: ', atomnum2, '\n\tdistance: ', distance)
        print('#'*debugprint)
        spline = self.spline_data[frozenset((int(atomnum1), int(atomnum2)))]
        #print('Spline data read: \n', spline)
        tail_start = spline.grid[-1]
        print('Tail start: ', tail_start)
        exp_head_cutoff = spline.grid[0]
        print('Exp head cutoff: ', exp_head_cutoff)
        print('-'*debugprint)

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
    def _exponential_head(cls, distance, coeffs):
        print('Called _exponential_head for \n\tdistance: ', distance, '\n\tcoeffs: ', coeffs)
        a1 = coeffs[0].clone()
        a2 = coeffs[1].clone()
        a3 = coeffs[2].clone()
        print('cloned coeffs values: \n')
        print('a1: ', a1, ' a2: ', a2, ' a3: ', a3)

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
        print('Called _spline for \n\tdistance: ', distance, ' \n\tstart: ', start, '\n\tcoeffs: ', coeffs)
        rDiff = distance - start
        print('   rDiff: ', rDiff)
        energy = coeffs[0] + coeffs[1]*rDiff + coeffs[2]*rDiff**2 + coeffs[3]*rDiff**3
        print('   Energy: ', energy)
        return energy

    @classmethod 
    def _tail(cls, distance, start, coeffs):
        print('Called _tail for \n\tdistance: ', distance, '\n\tstart: ', start, '\n\tcoeffs: ', coeffs)
        rDiff = distance - start
        print('   rDiff: ', rDiff)
        energy = coeffs[0] + coeffs[1]*rDiff + coeffs[2]*rDiff**2 + coeffs[3]*rDiff**3 + coeffs[4]*rDiff**4 + coeffs[5]*rDiff**5
        print('   Energy: ', energy)
        return energy



    @classmethod
    def from_database(cls, path: str, species: List[int], **kwargs) -> 'RepulsiveSplineFeed':
        interaction_pairs = combinations_with_replacement(species, r=2)
        return cls({interaction_pair: skf.Skf.read(path, interaction_pair).r_spline for interaction_pair in interaction_pairs})

