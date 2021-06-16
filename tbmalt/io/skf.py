# -*- coding: utf-8 -*-
"""Load Slater-Koster Tables."""
import os
from typing import Union, List, Dict, Tuple
import torch
import h5py
from h5py import Group
from torch import Tensor
from tbmalt.structures.geometry import batch_chemical_symbols
from tbmalt.data.sk import int_s, int_p, int_d, int_f, hdf_suffix, skf_suffix


class Skf:
    """Get data from Skf file or binary file for single element pair.

    Arguments:
        element_pair: Single element pair, it can be chemical name pair or
            element number pair.
        hamiltonian: Hamiltonian data.
        overlap: Overlap data.
        repulsive: A dictionary contains all the repulsive data.
        onsite: On-site data if element_pair is homo.
        U: Hubbert U data if element_pair is homo.

    Keyword Args:
        hs_grid: Distance grid points for Hamiltonian or overlap, this will
            be None if both hamiltonian and overlap are not returned.
        r_poly: The polynomial coefficients.
        g_step: Grid distance between `hs_grid`.
        n_grids: Number of grid points of Hamiltonian or overlap.
        mass: Mass of the chemical element if homo.
        occupations: Orbital occupations if homo.
        device: Device type used in this object.
        dtype: Tensor dtype used in this object.

    Attributes:
        element_pair: Single element pair, it can be chemical name pair or
            element number pair.
        homo: If the element_pair is homo or hetero.
        hamiltonian: Hamiltonian data.
        overlap: Overlap data.
        repulsive: All the repulsive keys from input repulsive dictionary.
        onsite: On-site data if element_pair is homo.
        U: Hubbert U data if element_pair is homo.ielement
        kwargs: All the keys from input kwargs dictionary.

    Notes:
        The class `Skf` tests on the mio, pbc and auorg type files, please be
        careful when read other type SKF files.

    Examples:
        Evaluating load mio SKF files with defined path to skf:

        >>> from tbmalt.io.skf import Skf
        >>> from tbmalt.data.sk import int_d
        >>> path = '../../tests/unittests/data/slko/mio/H-H.skf'
        >>> skf = Skf.read(path, torch.tensor([1, 1]))
        >>> print(skf.g_step)
        >>> 0.02
        >>> print(skf.hamiltonian.shape)
        >>> torch.Size([500, 10])

    """

    def __init__(self, element_pair: Union[Tensor, List[str]],
                 hamiltonian: Tensor = None, overlap: Tensor = None,
                 repulsive: dict = None, onsite: Tensor = None,
                 U: Tensor = None, **kwargs):
        self.element_pair = element_pair
        self.homo = self.element_pair[0] == self.element_pair[1]

        # HS is controlled by read_hamiltonian, read_overlap
        self.hamiltonian = hamiltonian
        self.overlap = overlap

        # set all repulsive attributes
        if repulsive is not None:
            for irep in repulsive.keys():
                setattr(self, irep, repulsive[irep])

        # If in hetero element_pair, U and onsite will be None
        self.onsite, self.U = onsite, U

        # all the rest parameters in kwargs dict
        for iarg in kwargs:
            setattr(self, iarg, kwargs[iarg])

    @classmethod
    def read(cls, path_to_file: str, element_pair: Tensor,
             mask_hs: bool = False, max_l: Dict[int, int] = None,
             interactions: List[Tuple[int, int, int]] = None,
             **kwargs) -> 'Skf':
        """Read different type SKF files according to interactions.

        To minimize the data memory, the Hamiltonian and overlap beyond the
        minimal basis will never be uesed, therefore selective Hamiltonian
        and overlap are applied and controlled by `with_mask`. If `with_mask`
        is True, either `max_l` or `interactions` should be assigned.

        Arguments:
            path_to_file: Joint path to SKF files or binary hdf with SKF data.
            element_pair: Single element number pair.
            mask_hs: If use mask to generate only the used Hamiltonian or
                overlap, the default is False.
            interactions: A list of orbital interactions, which is determined
                by the maximum of quantum number ℓ of each element pair.

        Returns:
            Skf: All attributes of object `Skf`.

        """
        # Check if the joint path to the SKF file exists
        assert os.path.isfile(path_to_file), '%s does not exist' % path_to_file

        # Get the type of the input SKF file with given joint path
        this_suffix = path_to_file.split('.')[-1]
        assert this_suffix in hdf_suffix + skf_suffix, 'suffix of ' + \
            '%s is not in %s or %s' % (path_to_file, hdf_suffix, skf_suffix)

        file_type = 'from_hdf' if this_suffix in hdf_suffix else 'from_skf'

        if mask_hs:
            assert max_l is not None or interactions is not None, 'mask_hs' + \
                ' is True, one of max_l and interactions should be offered'
            interactions = interactions if interactions is not None else \
                _interaction[max([max_l[int(iele)] for iele in element_pair])]

        return getattr(Skf, file_type)(
            path_to_file, element_pair, mask_hs, interactions, **kwargs)

    @classmethod
    def from_skf(cls, path_to_skf: str, element_pair: Tensor, mask_hs: bool,
                 interactions: List[Tuple[int, int, int]], **kwargs) -> 'Skf':
        """Read a skf file and return an `Skf` instance.

        File names should follow the naming convention X-Y.skf where X and
        Y are the names of chemical symbols.

        Arguments:
            path: Path to the target skf file.
            element_pair: Current element number pair.
            mask_hs: If use mask to generate only the used Hamiltonian or
                overlap, the default is False.
            interactions: A list of orbital interactions, which is determined
                by the maximum of quantum number ℓ of each element pair.

        Keyword Args:
            read_hamiltonian: If read Hamiltonian data.
            read_overlap: If read overlap data.
            read_repulsive: If read repulsive data.
            read_onsite: If read onsite data.
            read_U: If read Hubbert U.
            read_other_params: If read the rest of parameters.
            device: A device type used in this classmethod.
            dtype: A data dtype used in this classmethod.

        Returns:
            Skf: Return the arguments in `Skf` object.

        """
        read_hamiltonian = kwargs.get('read_hamiltonian', True)
        read_overlap = kwargs.get('read_overlap', True)
        read_repulsive = kwargs.get('read_repulsive', True)
        device = kwargs.get('device', torch.device('cpu'))
        dtype = kwargs.get('dtype', torch.get_default_dtype())

        homo = element_pair[0] == element_pair[1]

        file = open(path_to_skf, 'r').read()
        lines = file.split('\n')

        # @ will be in first line for extended format with f orbital
        if '@' in lines[0]:
            raise NotImplementedError('extended format not implemented')

        # check if asterisk in SKF files
        is_asterisk = '*' in lines[2] or '*' in lines[2 + homo]

        # 0th line, grid distance and grid points number
        g_step, n_grids = lines[0].replace(',', ' ').split()[:2]
        g_step, n_grids = float(g_step), int(n_grids)

        # read on-site energies, spe, Hubbard U and occupations
        if homo:
            if is_asterisk:
                homo_ln = _asterisk_to_repeat_tensor(_asterisk(
                    lines[1].replace(',', ' ')), dtype, device)
            else:
                homo_ln = torch.tensor(_lmf(lines[1])).to(device).to(dtype)
            n = int((len(homo_ln) - 1) / 3)  # -> Number of shells specified
            onsite, spe, U, occupations = homo_ln.split_with_sizes([n, 1, n, n])

        # return None for on-site, spe, U and occupations if hetero
        else:
            onsite, spe, U, occupations = None, None, None, None

        # read 1 + homo line data, which contains mass, rcut and cutoff
        if is_asterisk:
            mass, *r_poly, rcut = _asterisk_to_repeat_tensor(_asterisk(
                lines[1 + homo].replace(',', '')), dtype, device)[:10]
        else:
            mass, *r_poly, rcut = torch.tensor(
                _lmf(lines[1 + homo]), dtype=dtype, device=device)[:10]
        mass, rcut, r_poly = float(mass), float(rcut), torch.stack(r_poly)

        # to avoid error when reading PBC SKF, in PBC, lines of HS < n_grids
        rep_line = lines.index('Spline') + 1
        line_hs_end = min(rep_line - 1, n_grids + 2 + homo)

        # read the hamiltonian and overlap tables with '*' in each line
        if is_asterisk and read_hamiltonian + read_overlap:
            h_data, s_data = torch.stack([_asterisk_to_repeat_tensor(
                _asterisk(ii.replace(',', ' ')), dtype, device)
                for ii in lines[2 + homo: line_hs_end]]).chunk(2, 1)
        elif not is_asterisk and read_hamiltonian + read_overlap:
            h_data, s_data = torch.tensor(
                [_lmf(ii) for ii in lines[2 + homo: line_hs_end]],
                dtype=dtype, device=device).chunk(2, 1)

        if read_hamiltonian + read_overlap:
            # get the mask for Hamiltonian (mask_h) and overlap (mask_s)
            len_hs = h_data.shape[1]
            mask_h = _get_hs_mask(len_hs, read_hamiltonian, mask_hs, interactions)
            mask_s = _get_hs_mask(len_hs, read_overlap, mask_hs, interactions)
            hs_grid = torch.arange(1, len(h_data) + 1, device=device) * g_step
            h_data, s_data = h_data[..., mask_h], s_data[..., mask_s]
        else:
            h_data, s_data, hs_grid = None, None, None

        kwd = {'hs_grid': hs_grid, 'r_poly': r_poly, 'n_grids': n_grids,
               'g_step': g_step, 'rcut': rcut, 'device': device, 'dtype': dtype}

        if homo:  # Passed only if homo case
            kwd.update({'mass': mass, 'occupations': occupations, 'spe': spe})

        # Check if there is a spline representation
        if read_repulsive:
            r_int, r_cutoff = lines[rep_line].split()  # -> 1st line
            r_int, r_cutoff = int(r_int), float(r_cutoff)
            r_a123 = torch.tensor(_lmf(lines[rep_line + 1])).to(device).to(dtype)
            r_tab = torch.tensor([_lmf(line) for line in lines[
                rep_line + 2: rep_line + 1 + r_int]]).to(device).to(dtype)
            rep = r_tab[:, 2:]  # -> repulsive tables
            r_grid = torch.tensor([*r_tab[:, 0], r_tab[-1, 1]]).to(device).to(dtype)
            r_long_tab = torch.tensor(
                _lmf(lines[rep_line + 1 + r_int])).to(device).to(dtype)
            r_long_grid = r_long_tab[:2]  # last line start, end
            r_c_0to5 = r_long_tab[2:]  # last line values

        rep = {'r_int': r_int, 'r_table': rep, 'r_long_grid': r_long_grid,
               'r_grid': r_grid, 'r_a123': r_a123, 'r_c_0to5': r_c_0to5,
               'r_cutoff': r_cutoff} if read_repulsive else None

        return cls(element_pair, h_data, s_data, rep, onsite, U, **kwd)

    @classmethod
    def from_hdf(cls, path: str, element_pair: Union[Tensor, str],
                 mask_hs: bool, interactions: List[Tuple[int, int, int]],
                 **kwargs) -> 'Skf':
        """Generate integral from h5py binary data.

        Arguments:
            path: Path to the target binary file.
            element_pair: Current element number pair.
            mask_hs: If use mask to generate only the used Hamiltonian or
                overlap, the default is False.
            interactions: A list of orbital interactions, which is determined
                by the maximum of quantum number ℓ of each element pair.

        Keyword Args:
            read_hamiltonian: If read Hamiltonian data.
            read_overlap: If read overlap data.
            read_repulsive: If read repulsive data.
            read_onsite: If read onsite data.
            read_U: If read Hubbert U.
            read_other_params: If read the rest of parameters.
            device: Device type used in this classmethod.
            dtype: Tensor dtype used in this classmethod.

        Returns:
            Skf: Return the arguments in `Skf` object.

        """
        read_hamiltonian = kwargs.get('read_hamiltonian', True)
        read_overlap = kwargs.get('read_overlap', True)
        read_repulsive = kwargs.get('read_repulsive', True)
        read_onsite = kwargs.get('read_onsite', True)
        read_U = kwargs.get('read_U', True)
        read_other_params = kwargs.get('read_other_params', True)
        device = kwargs.get('device', torch.device('cpu'))
        dtype = kwargs.get('dtype', torch.get_default_dtype())
        homo = element_pair[0] == element_pair[1]

        # get the group name with chemical element name
        if isinstance(element_pair, Tensor):
            element_name_pair = batch_chemical_symbols(element_pair)
        this_name = element_name_pair[0] + '-' + element_name_pair[1]

        with h5py.File(path, 'r') as f:

            if read_hamiltonian:
                maskh = _get_hs_mask(
                    f[this_name + '/hamiltonian'][()].shape[1],
                    read_hamiltonian, mask_hs, interactions)
                h_data = torch.from_numpy(f[this_name + '/hamiltonian'][()][
                    ..., maskh]).to(device).to(dtype)
            else:
                h_data = None

            if read_overlap:
                masks = _get_hs_mask(f[this_name + '/overlap'][()].shape[1],
                                     read_overlap, mask_hs, interactions)
                s_data = torch.from_numpy(f[this_name + '/overlap'][()][
                    ..., masks]).to(device).to(dtype)
            else:
                s_data = None

            if read_repulsive:
                rep = {ipr: torch.from_numpy(f[
                    this_name + '/' + ipr][()]).to(device).to(dtype) for ipr
                    in ['r_table', 'r_grid', 'r_a123', 'r_long_grid', 'r_c_0to5']}
                rep.update({'r_cutoff': f[this_name + '/r_cutoff'][()]})
                rep.update({'r_int': f[this_name + '/r_int'][()]})
            else:
                rep = None

            onsite = torch.from_numpy(f[this_name + '/onsite'][()]).to(
                device).to(dtype) if read_onsite and homo else None

            U = torch.from_numpy(f[this_name + '/U'][()]).to(device).to(
                dtype) if read_U and homo else None

            if read_other_params:
                kwd = {ipara: f[this_name + '/' + ipara][()] for ipara in
                       ['rcut', 'g_step', 'n_grids']}
                kwd.update({'hs_grid': torch.from_numpy(
                    f[this_name + '/hs_grid'][()]).to(device).to(dtype)})

                # update homo element_pair
                if homo:
                    kwd.update({'occupations': torch.from_numpy(
                        f[this_name + '/occupations'][()]).to(device).to(dtype)})
                    kwd.update({'mass': f[this_name + '/mass'][()]})
            else:
                kwd = {}  # create empty dict
            kwd.update({'device': device, 'dtype': dtype})

        return cls(element_pair, h_data, s_data, rep, onsite, U, **kwd)

    @staticmethod
    def to_hdf(target: Union[str, Group], skf: object,
               read_hamiltonian: bool = True, read_overlap: bool = True,
               read_repulsive: bool = True, read_onsite: bool = True,
               read_U: bool = True, read_other_params: bool = True,
               mode: str = 'a'):
        """Write standard Slater-Koster data to hdf type.

        Arguments:
            target: The string will be the name of the target to be written,
                the Group type will be the opened `File` object in h5py.
            skf: Object with Slater-Koster raw data.
            element_pair: Tensor type element number pairs or string type
                chemical element names.
            read_hamiltonian: If write Hamiltonian data.
            read_overlap: If write overlap data.
            read_repulsive: If write repulsive data.
            mode: Mode to write data.

        """
        if isinstance(target, str):
            # Create a HDF5 database and save the feed to it
            target = h5py.File(target, mode=mode)

        # get the group name with chemical element name
        if isinstance(skf.element_pair, Tensor):
            chemical_name_pair = batch_chemical_symbols(skf.element_pair)
        else:
            chemical_name_pair = skf.element_pair
        g_name = chemical_name_pair[0] + '-' + chemical_name_pair[1]

        # If element pair is not in target, create new group
        g = target[g_name] if g_name in target.keys() else \
            target.create_group(g_name)

        # write hamiltonian, overlap, onsite and U, currently gradient is not
        # included in skf object, detach is not necessary, cpu() will avoid
        # error if data type is cuda
        if read_hamiltonian:
            g.create_dataset('hamiltonian', data=skf.hamiltonian.cpu())

        if read_overlap:
            g.create_dataset('overlap', data=skf.overlap.cpu())

        if read_onsite and skf.homo:
            g.create_dataset('onsite', data=skf.onsite.cpu())

        if read_U and skf.homo:
            g.create_dataset('U', data=skf.U.cpu())

        if read_repulsive:
            g.create_dataset('repulsive', data=read_repulsive)
            g.create_dataset('r_int', data=skf.r_int)
            g.create_dataset('r_cutoff', data=skf.r_cutoff)
            for ipr in ['r_table', 'r_grid', 'r_a123', 'r_long_grid', 'r_c_0to5']:
                g.create_dataset(ipr, data=getattr(skf, ipr).cpu())

        if read_other_params:
            for ipr in ['g_step', 'n_grids', 'rcut']:
                g.create_dataset(ipr, data=getattr(skf, ipr))

            for ipr in ['hs_grid', 'r_poly']:
                g.create_dataset(ipr, data=getattr(skf, ipr).cpu())


            if skf.homo:
                g.create_dataset('mass', data=skf.mass)
                g.create_dataset('occupations', data=skf.occupations.cpu())


class CompressionRadii:
    pass


def _get_hs_mask(n_interaction: int, read_ski: bool, mask_hs: bool,
                 interactions: List[Tuple[int, int, int]]) -> List[bool]:
    """Return the mask for Hamiltonian or overlap.

    The read_ski determines if read Hamiltonian or overlap, if False, the
    mask will return False for all interactions. If read_ski is True and
    mask_hs is False, the mask will return True for all interactions, it
    suggests that all Hamiltonian or overlap will be returned. If read_ski is
    True and mask_hs is True, the mask will be applied to select the
    interactions.

    Arguments:
        n_interaction: The total number of Hamiltonian or overlap interactions.
        read_ski: If read Hamiltonian or overlap.
        mask_hs: If use mask to generate only the used Hamiltonian or overlap.
        interactions: A list of orbital interactions, which is determined
            by the maximum of quantum number ℓ of each element pair.

    Returns:
        mask: Mask the interactions in Hamiltonian or overlap.

    """
    if mask_hs and read_ski:
        # In normal SKF, the total Hamiltonian and overlap in each line
        # is 20, in extended format, it will be 40. The number of interactions
        # will be the half.
        assert n_interaction in (10, 20), 'Number of interactions ' + \
            'should be 10 or 20, but get %d' % n_interaction
        assert len(interactions) <= n_interaction, 'number of interactions' + \
            ' is more than the total interactions size: %d' % n_interaction

        sk_i = _interaction[2] if n_interaction == 10 else _interaction[3]

        return [True if interaction in interactions else False
                for interaction in sk_i]

    # do not use mask, return True to read all Hamiltonian or overlap
    elif not mask_hs and read_ski:
        return [True] * n_interaction

    # do not read Hamiltonian or overlap, return False
    elif not read_ski:
        return [False] * n_interaction


def _asterisk_to_repeat_tensor(
        xx: list, dtype=torch.float64, device=torch.device('cpu')) -> Tensor:
    """Transfer data with asterisk to Tensor."""
    return torch.cat([torch.tensor([float(ii.split('*')[1])]).repeat(
        int(ii.split('*')[0])) if '*' in ii else torch.tensor([float(ii)])
        for ii in xx]).to(device).to(dtype)


# alias for common code structure to deal with SK tables each line
_lmf = lambda xx: list(map(float, xx.split()))
_asterisk = lambda xx: list(map(str.strip, xx.split()))

# default interactions for each max quantum number, to make sure the order of
# interactions will be always fixed
_interaction = {0: int_s, 1: int_p, 2: int_d, 3: int_f}
