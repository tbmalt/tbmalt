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
        indxs = torch.tensor(range(len(geo.atomic_numbers)))
        indx_pairs = torch.combinations(indxs)
        
        Erep = 0
        for indx_pair in indx_pairs:
            print('###################################')
            print('indx_pair:')
            print(indx_pair)
            atomic_numbers = (int(geo.atomic_numbers[indx_pair[0]]),
                              int(geo.atomic_numbers[indx_pair[1]]))
            print('atomic_numbers:')
            print(atomic_numbers)
            distance = geo.distances[indx_pair[0], indx_pair[1]]
            print('distance')
            print(distance)

            Erep += self._repulsive_calc(distance, self.spline_data[frozenset(atomic_numbers)])
            print('Erep:')
            print(Erep)

        return Erep

    def _repulsive_calc(self, distance, spline):
        tail_start = spline.grid[-1]
        exp_head_cutoff = spline.grid[0]

        if distance < spline.cutoff:
            if distance > tail_start:
                return self._spline(distance, tail_start, spline.tail_coef)
            elif distance > exp_head_cutoff:
                for ind in range(len(spline.grid)):
                    if distance < spline.grid[ind]:
                        return self._spline(distance, spline.grid[ind - 1], spline.spline_coef[ind])
            else:
                return self._exponential_head(distance, spline.exp_coef)
        return 0
   
    @classmethod
    def _exponential_head(cls, distance, coeffs):
        a1 = coeffs[0]
        a2 = coeffs[1]
        a3 = coeffs[2]

        return np.exp(-a1*distance + a2) + a3

    @classmethod 
    def _spline(cls, distance, start, coeffs):
        energy = coeffs[0]
        rDiff = r - start
        for coeff in coeffs[1:]:
            energy += coeff*rDiff
            rDiff *= rDiff
        return energy

class RepulsiveSplineFeed_batch():

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
        print('----------------')
        print('Erep:')
        print(Erep)


        for indx_pair in indx_pairs:
            print('###################################')
            print('indx_pair:')
            print(indx_pair)
            print('----------------')
            atomnum1 = geo.atomic_numbers[..., indx_pair[0]].reshape((batch_size, ))
            atomnum2 = geo.atomic_numbers[..., indx_pair[1]].reshape((batch_size, ))
            print('atomic_numbers:')
            print('first:')
            print(atomnum1)
            print('second:')
            print(atomnum2)

            distance = geo.distances[..., indx_pair[0], indx_pair[1]].reshape((batch_size, ))
            print('----------------')
            print('distance')
            print(distance)

            for batch_indx in range(batch_size):
                if atomnum1[batch_indx] == 0 or atomnum2[batch_indx] == 0:
                    continue
                Erep[batch_indx] += self._repulsive_calc(distance[batch_indx], atomnum1[batch_indx], atomnum2[batch_indx])

            print('----------------')
            print('Erep:')
            print(Erep)

        return Erep

    def _repulsive_calc(self, distance, atomnum1, atomnum2):
        print(atomnum1)
        print(distance)
        spline = self.spline_data[frozenset((int(atomnum1), int(atomnum2)))]
        tail_start = spline.grid[-1]
        exp_head_cutoff = spline.grid[0]

        if distance < spline.cutoff:
            if distance > tail_start:
                return self._spline(distance, tail_start, spline.tail_coef)
            elif distance > exp_head_cutoff:
                for ind in range(len(spline.grid)):
                    if distance < spline.grid[ind]:
                        return self._spline(distance, spline.grid[ind - 1], spline.spline_coef[ind])
            else:
                return self._exponential_head(distance, spline.exp_coef)
        return 0
   
    @classmethod
    def _exponential_head(cls, distance, coeffs):
        a1 = coeffs[0]
        a2 = coeffs[1]
        a3 = coeffs[2]

        return np.exp(-a1*distance + a2) + a3

    @classmethod 
    def _spline(cls, distance, start, coeffs):
        energy = coeffs[0]
        rDiff = distance - start
        for coeff in coeffs[1:]:
            energy += coeff*rDiff
            rDiff *= rDiff
        return energy

    @classmethod
    def from_database(cls, path: str, species: List[int], **kwargs) -> 'RepulsiveSplineFeed_batch':
        interaction_pairs = combinations_with_replacement(species, r=2)
        return cls({interaction_pair: skf.Skf.read(path, interaction_pair).r_spline for interaction_pair in interaction_pairs})


geometry_batch = Geometry.from_ase_atoms([molecule('H2O'), molecule('CH4')])
geometry_single = Geometry.from_ase_atoms(molecule('CH4'))

spline_1 = skf.Skf.RSpline(
        grid = torch.Tensor([10, 12.0, 14.0]),
        cutoff = torch.Tensor([16.0]),
        spline_coef = torch.Tensor([[2.1, 5.3, 7.6],
                                    [3.2, 8.9, 1.0],
                                    [7.2, 10.2, 0.6]]),
        exp_coef = torch.Tensor([1.0, 0.8, 1.2]),
        tail_coef = torch.Tensor([2.0, 5.3, 6.7, 1.3, 0.8]),
        )

spline_6 = skf.Skf.RSpline(
        grid = torch.Tensor([10, 12.0, 14.0]),
        cutoff = torch.Tensor([16.0]),
        spline_coef = torch.Tensor([[2.1, 5.3, 7.6],
                                    [3.2, 8.9, 1.0],
                                    [7.2, 10.2, 0.6]]),
        exp_coef = torch.Tensor([1.0, 0.8, 1.2]),
        tail_coef = torch.Tensor([2.0, 5.3, 6.7, 1.3, 0.8]),
        )

spline_8 = skf.Skf.RSpline(
        grid = torch.Tensor([10, 12.0, 14.0]),
        cutoff = torch.Tensor([16.0]),
        spline_coef = torch.Tensor([[2.1, 5.3, 7.6],
                                    [3.2, 8.9, 1.0],
                                    [7.2, 10.2, 0.6]]),
        exp_coef = torch.Tensor([1.0, 0.8, 1.2]),
        tail_coef = torch.Tensor([2.0, 5.3, 6.7, 1.3, 0.8]),
        )

spline_data = {(1, 1): spline_1, (6, 1): spline_6, (8, 1): spline_8}

#feed = RepulsiveSplineFeed_batch(spline_data) 
#print(feed(geometry_single))
#print(feed(geometry_batch))

atoms = geometry_batch.atomic_numbers
print(atoms)

atoms = [8, 6, 1]
print(atoms)

feed = RepulsiveSplineFeed_batch.from_database('skf/auorg-1-1.hdf5', species=atoms)
print(feed(geometry_batch))

                                                                         
