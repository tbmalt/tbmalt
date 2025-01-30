# -*- coding: utf-8 -*-
"""Methods for reading from and writing to skf and associated files."""
import re
import warnings
from dataclasses import dataclass
from os.path import isfile, split, splitext, basename, isdir, exists, join
from time import time
from typing import List, Tuple, Union, Dict, Sequence, Any, Optional
from itertools import product
import glob

import h5py
import numpy as np
import torch
from h5py import Group
from torch import Tensor

from tbmalt.common.maths import triangular_root, tetrahedral_root
from tbmalt.data import atomic_numbers, chemical_symbols
from tbmalt.common.batch import pack

OptTens = Optional[Tensor]
SkDict = Dict[Tuple[int, int], Tensor]


class Skf:
    r"""Slater-Koster file parser.

    This class handles the parsing of DFTB+ skf formatted Slater-Koster files,
    and their binary analogs. Data can be read from and saved to files using
    the `read` & `write` methods. Reading a file will return an `Skf` instance
    holding all stored data.

    Arguments:
         atom_pair: Atomic numbers of the elements associated with the
            interaction.
         hamiltonian: Dictionary keyed by azimuthal-number-pairs (ℓ₁, ℓ₂) and
            valued by m×d Hamiltonian integral tensors; where m and d iterate
            over bond-order (σ, π, etc.) and distances respectively.
         overlap: Dictionary storing the overlap integrals in a manner akin to
            ``hamiltonian``.
         grid: Distances at which the ``hamiltonian`` & ``overlap`` elements
            were evaluated.
         r_spline: A :class:`.RSpline` object detailing the repulsive
            spline. [DEFAULT=None]
         r_poly: A :class:`.RPoly` object detailing the repulsive
            polynomial. [DEFAULT=None]
         on_sites: On site terms, homo-atomic systems only. [DEFAULT=None]
         hubbard_us: Hubbard U terms, homo-atomic systems only. [DEFAULT=None]
         mass: Atomic mass, homo-atomic systems only. [DEFAULT=None]
         occupations: Occupations of the orbitals, homo-atomic systems only.
            [DEFAULT=None]

    Examples:
        Examples of reading and writing.

        >>> import urllib, tarfile
        >>> from os.path import join
        >>> from tbmalt.io.skf import Skf
        >>> link = 'https://github.com/dftbparams/auorg/releases/download/v1.1.0/auorg-1-1.tar.xz'
        >>> taraug = urllib.request.urlretrieve(link, path := join('./auorg-1-1.tar.xz'))
        >>> tartmp = tarfile.open(path)
        >>> tartmp.extractall('./')
        >>> cc = Skf.from_skf('./auorg-1-1/C-C.skf')
        >>> print(cc.hamiltonian.keys())
        dict_keys([(0, 0), (0, 1), (1, 1)])

        use auorg-1-1 to generate binary h5 and read the binary file

        >>> cc.write('tmp.h5')
        >>> cch5 = Skf.read('tmp.h5')
        >>> print(cch5.hamiltonian.keys())
        dict_keys([(0, 0), (0, 1), (1, 1)])

    Attributes:
        atomic: True if the system contains atomic data, only relevant to the
            homo-atomic cases.

    .. _Notes:
    Notes:
        HOMO atomic systems commonly, but not always, include additional
        "atomic" data; namely atomic mass, on-site terms, occupations, and
        the Hubbard-U values. These can be optionally specified using the
        ``mass``, ``on_sites``, ``occupations``, and ``hubbard_us`` attributes
        respectively. However, these attributes are mutually inclusive, i.e.
        either all are specified or none are. Furthermore, values contained
        within such tensors should be ordered from the lowest azimuthal number
        to highest, where applicable.

        Further information regarding the skf file format specification can be
        found in the document: "`Format of the v1.0 Slater-Koster Files`_".

    Warnings:
        This may fail to parse files which do not strictly adhere to the skf
        file format. Some skf files, such as those from the "pbc" parameter
        set, contain non-trivial errors in them, e.g. incorrectly specified
        number of grid points. Such files require fixing before they can be
        read in.

        The ``atom_pair`` argument is order sensitive, i.e. [6, 7] ≠ [7, 6].
        For example, the p-orbital of the s-p-σ interaction would be located
        on N when ``atom_pair`` is [6, 7] but on C when it is [7, 6].

    Raises:
        ValueError: if some but not all atomic attributes are specified. See
            the :ref:`Notes` section for more details.

    .. _Format of the v1.0 Slater-Koster Files:
        https://dftb.org/fileadmin/DFTB/public/misc/slakoformat.pdf

    """

    # Used to reorder hamiltonian and overlap data read in from skf files.
    _sorter = [9, 8, 7, 5, 6, 3, 4, 0, 1, 2]
    _sorter_e = [19, 18, 17, 16, 14, 15, 12, 13, 10, 11,
                 7, 8,  9,  4,  5,  6,  0, 1,  2,  3]

    # Dataclasses for holding the repulsive interaction data.
    @dataclass
    class RPoly:
        """Dataclass container for the repulsive polynomial.

        Arguments:
            cutoff: Cutoff radius of the repulsive interaction.
            coef: The eight polynomial coefficients (c2-c9).
            """
        cutoff: Tensor
        coef: Tensor

    @dataclass
    class RSpline:
        """Dataclass container for the repulsive spline.

        Arguments:
            grid: Distance for the primary spline segments.
            cutoff: Cutoff radius for the tail spline.
            spline_coef: The primary spline's Coefficients (four per segment).
            exp_coef: The exponential expression's coefficients a1, a2 & a3.
            tail_coef: The six coefficients of the terminal tail spline.

        """
        grid: Tensor
        cutoff: Tensor
        spline_coef: Tensor
        exp_coef: Tensor
        tail_coef: Tensor

    # HDF5-SK version number. Updated when introducing a change that would
    # break backwards compatibility with previously created HDF5-skf file.
    version = '0.1'

    def __init__(
            self, atom_pair: Tensor, hamiltonian: SkDict, overlap: SkDict,
            grid: Tensor, r_spline: Optional[RSpline] = None,
            r_poly: Optional[RPoly] = None, hubbard_us: OptTens = None,
            on_sites: OptTens = None, occupations: OptTens = None,
            mass: OptTens = None):

        self.atom_pair = atom_pair

        # SkDict attributes
        self.hamiltonian = hamiltonian
        self.overlap = overlap
        self.grid = grid

        # Ensure grid is uniformly spaced
        if not (grid.diff().diff().abs() < 1E-5).all():
            raise ValueError('Electronic integral grid spacing is not uniform')

        # Repulsive attributes
        self.r_spline = r_spline
        self.r_poly = r_poly

        # Either the system contains atomic information or it does not; it is
        # illogical to have some atomic attributes but not others.
        check = [i is not None for i in [on_sites, hubbard_us,
                                         occupations, mass]]
        if all(check) != any(check):
            raise ValueError(
                'Either all or no atomic attributes must be supplied:'
                '\n\t- on_sites\n\t- hubbard_us\n\t- mass\n\t- occupations')

        # Atomic attributes
        self.atomic: bool = all(check)
        self.on_sites = on_sites
        self.hubbard_us = hubbard_us
        self.mass = mass
        self.occupations = occupations

    @classmethod
    def read(cls, path: str, atom_pair: Optional[Sequence[int]] = None,
             **kwargs) -> 'Skf':
        """Parse Slater-Koster data from skf files and their binary analogs.

        Arguments:
            path: Path to the file that is to be read (.skf or .hdf5).
            atom_pair: Atomic numbers of the element pair. This is only used
                when reading from an HDF5 file with more than one SK entry.
                [DEFAULT=None]

        Keyword Arguments:
            device: Device on which to place tensors. [DEFAULT=None]
            dtype: dtype to be used for floating point tensors. [DEFAULT=None]

        Returns:
            skf: `Skf` object containing all data parsed from the specified
                file.
        """
        if not isfile(path):  # Verify the target file exists
            raise FileNotFoundError(f'Could not find: {path}')

        if 'sk' in splitext(path)[1].lower():  # If path points to an skf file
            # Issue a waring if the user specifies `atom_pair` for an skf file
            if atom_pair is not None:
                warnings.warn('"atom_pair" argument is only used when reading'
                              'from HDF5 files with multiple SK entries.')

            return cls.from_skf(path, **kwargs)

        with h5py.File(path, 'r') as db:  # Otherwise must be a hdf5 database
            # If atom_pair is specified use this to identify the target
            if atom_pair is not None:
                name = '-'.join([chemical_symbols[int(i)] for i in atom_pair])
            else:
                # Otherwise scan for valid entries: if only 1 SK entry exists
                # then assume it's the target; if multiple entries exist then
                # it's impossible to know which the user wanted.
                e = '[A-Z][a-z]*'
                entries = [k for k in db if re.fullmatch(f'{e}-{e}', k)]
                if len(entries) == 1:
                    name = entries[0]
                else:
                    raise ValueError('Use atom_pair when database have '
                                     f'more than one entry: {basename(path)}')

            return cls.from_hdf5(db[name], **kwargs)

    @classmethod
    def from_skf(cls, path: str, dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None) -> 'Skf':
        """Parse and skf file into an `Skf` instance.

        File names should follow the naming convention X-Y.skf where X & Y are
        the chemical symbols of the associated elements. However, any file
        which **ends** in X.Y will be successfully parsed (where "." is any
        character (including no character)).

        Arguments:
            path: Path to the target skf file.
            device: Device on which to place tensors. [DEFAULT=None]
            dtype: dtype to be used for floating point tensors. [DEFAULT=None]

        Returns:
            skf: Return the arguments in `Skf` object.

        """
        dd = {'dtype': dtype, 'device': device}
        kwargs_in = {}

        # Identify the elements involved according to the file name
        e = '[A-Z][a-z]?'
        try:
            atom_pair = torch.tensor([atomic_numbers[i] for i in re.findall(
                e, re.search(rf'{e}.?{e}(?=.)', split(path)[-1]).group(0))])
        except AttributeError as error:
            raise ValueError(
                'Could not parse element names form file.') from error

        lines = open(path, 'r').readlines()

        # Remove the comment line if present
        lines = lines[1:] if lines[0].startswith('@') else lines

        # 0th line, grid distance and grid points number
        g_step, n_grids = lines[0].replace(',', ' ').split()[:2]
        g_step, n_grids = float(g_step), int(n_grids)

        # In line with file specification only "nGridPoints - 1" points are
        # guaranteed to be present. This is because the grid point at r=0 is
        # implicit rather than explicit. It is worth remarking that some skf
        # files will actually provide nGridPoints rows; in these cases the
        # last row is ignored.
        n_grids = n_grids - 1
        grid = torch.arange(1, n_grids + 1, **dd) * g_step

        # Determine if this is the homo/atomic case (from the file's contents)
        atomic = len(atom_ln := _s2t(_esr(lines[1]), **dd)) in [10, 13]

        # Read in the mass and polynomial repulsion coefficients
        mass, r_poly, r_cut = _s2t(_esr(lines[1 + atomic]),
                                   **dd)[:10].split([1, 8, 1])

        # If polynomial coefficients are valid, create an r_poly object
        if (r_poly != 0).any():
            kwargs_in['r_poly'] = cls.RPoly(r_cut, r_poly)

        # Parse hamiltonian/overlap integrals.
        h_data, s_data = _s2t(_esr('  '.join(
            lines[2 + atomic: 2 + atomic + n_grids])),
            **dd).view(n_grids, -1).chunk(2, 1)

        # H/S tables are reordered so the lowest l comes first, broken up
        # into shell-pair chunks, e.g. ss, sp, sd, pp, etc, before finally
        # being placed into dictionaries.
        count = h_data.shape[-1]
        sort = cls._sorter if count == 10 else cls._sorter_e  # ◂──────┐
        max_l = round(tetrahedral_root(count) - 1)  # ◂─f-orbital catch┘

        # Sort, segmentation and parse the tables into a pair of dictionaries
        l_pairs = torch.triu_indices(max_l+1, max_l+1).T
        h_data, s_data = [{
            tuple(l_pair.tolist()): integral for l_pair, integral in
            #            |   ↓ Sorting ↓   |    ↓ Segmentation by ℓ pair ↓    |
            zip(l_pairs, integrals.T[sort].split((l_pairs[:, 0] + 1).tolist()))
            if not (integral == 0.).all()}  # ← Ignore any dummy interactions
            for integrals in [h_data, s_data]]

        if atomic:  # Parse homo data; on-site/Hubbard-U/occupations. (skip spe)
            n = int((len(atom_ln) - 1) / 3)  # -> Number of shells specified
            occs, hubb_u, _, on_site = atom_ln.flip(0).split([n, n, 1, n])
            # If integrals were culled; atomic data must be too.
            max_l = int(triangular_root(len(h_data)) - 1) + 1
            kwargs_in.update({
                'mass': mass, 'occupations': occs[:max_l],
                'on_sites': on_site[:max_l], 'hubbard_us': hubb_u[:max_l]})

        # Parse repulsive spline (if present)
        if 'Spline\n' in lines:
            ln = lines.index('Spline\n') + 2
            n_int, r_cutoff = lines[ln - 1].split()
            r_tab = _s2t(lines[ln + 1: ln + int(n_int)], **dd).view(-1, 6)
            r_grid = torch.cat((r_tab[:, 0], r_tab[None, -1, 1]))
            kwargs_in['r_spline'] = cls.RSpline(
                # Repulsive grid, cutoff & repulsive spline coefficients.
                r_grid, torch.tensor(float(r_cutoff), **dd), r_tab[:, 2:],
                # The exponential and tail spline's coefficients.
                _s2t(lines[ln], **dd), _s2t(lines[ln + int(n_int)], **dd)[2:])

        return cls(atom_pair, h_data, s_data, grid, **kwargs_in)

    @classmethod
    def from_hdf5(cls, source: Group, dtype: Optional[torch.dtype] = None,
                  device: Optional[torch.device] = None) -> 'Skf':
        """Instantiate a `Skf` instances from an HDF5 group.

        Arguments:
            source: An HDF5 group containing Slater-Koster data.
            device: Device on which to place tensors. [DEFAULT=None]
            dtype: dtype to be used for floating point tensors. [DEFAULT=None]

        Returns:
            skf: The resulting `Skf` object.
        """
        # Make a call to the proxy helper function
        atom_pair, H, S, grid, kwargs = cls._from_hdf5_helper(
            source, dtype, device)
        return cls(atom_pair, H, S, grid, **kwargs)

    def write(self, path: str, overwrite: Optional[bool] = False):
        """Save the Slater-Koster data to a file.

        The target file can be either a skf file or a hdf5 database. Desired
        file format will be inferred from the file's name.

        Arguments:
            path: path to the file in which the data is to be saved.
            overwrite: Existing skf-files/HDF5-groups can only be overwritten
                when ``overwrite`` is True. [DEFAULT=False]

        """
        if 'sk' in splitext(path)[1].lower():  # If path points to an skf file
            if isfile(path) and not overwrite:
                raise FileExistsError(
                    'File already exists; use "overwrite" to permit '
                    'overwriting.')
            self.to_skf(path)

        else:  # Otherwise it must be an HDF5 file
            with h5py.File(path, 'a') as db:  # Create/open the HDF5 file
                name = '-'.join([chemical_symbols[int(i)]
                                 for i in self.atom_pair])
                if name in db:  # If an entry already exists in this database
                    if not overwrite:  # Then raise an exception
                        raise FileExistsError(
                            f'Entry {name} already exists; use "overwrite" '
                            'to permit overwriting.')
                    else:  # Unless told to overwrite it
                        del db[name]
                # Create the HDF5 entry & fill it with data via `to_hdf5`.
                self.to_hdf5(db.create_group(name))

    def to_skf(self, path: str):
        """Writes data to a skf formatted file.

        Arguments:
            path: path specifying the location of the skf file.
        """

        def t2a(t):
            """Converts a torch tensor to numpy array."""
            return t.detach().cpu().numpy()

        def a2s(a, f):
            """Converts a numpy array into a formatted string."""
            # Slow but easy way to convert array to string
            if a.ndim == 1:
                return ''.join(f'{j:{f}}' for j in a)
            else:
                return '\n'.join([a2s(j, f) for j in a])

        # Used for working out array lengths later on
        max_l = max(max(self.hamiltonian)[0], 2)

        # Build the first line defining the integral data's grid.
        # Format: {grid step size} {number of grid points}
        grid_n = len(self.grid)
        grid_step = self.grid.diff()[0]
        # Note that nGridPoints is always one more than the actual number
        # of integral rows provided as the zeroth point is implicit.
        output = f'{grid_step:<12.8f}{grid_n + 1:>5}'

        # Parse the atomic data into a string.
        # Format: {on site terms} {SPE} {hubbard u values} {occupancies}
        if self.atomic:
            # Care must be taken when parsing atomic data ase some elements of
            # these arrays may have been culled at read time.
            homo = np.zeros((max_l + 1) * 3)  # Parse in standard atomic data
            for n, i in enumerate([self.occupations, self.hubbard_us,
                                   self.on_sites]):
                homo[(start := (max_l + 1) * n):start + len(i)] = t2a(i)
            # Add dummy SPE value and reverse the array's order
            homo = np.flip(np.insert(homo, (max_l + 1) * 2, 0.))
            # Finally append the homo data to the output string
            output += '\n' + a2s(homo, '>21.12E')

        # Generate the repulsive polynomial line.
        # Format {mass} {coefficients} {cutoff} {ZEROS}
        coef = np.zeros(7) if self.r_poly is None else t2a(self.r_poly.coef)
        r = np.zeros(1) if self.r_poly is None else t2a(self.r_poly.cutoff)
        mass = t2a(self.mass) if self.atomic else np.zeros(1)
        r_poly_data = np.hstack((mass, coef, r, np.zeros(10)))

        output += '\n' + a2s(r_poly_data, '>21.12E')

        # Build HS data
        ls = range(max_l, -1, -1)
        lps = [i for i in product(ls, ls) if i[0] <= i[1]]
        hs_data = np.hstack([  # Concatenate H & S matrices.
            np.hstack(  # Collate each integral, adding dummy data as needed.
                [t2a(torch.atleast_2d(i.get(l, torch.zeros(l[0], grid_n)))).T
                 for l in lps])
            for i in [self.hamiltonian, self.overlap]])
        output += '\n' + a2s(hs_data, '>21.12E')

        # Append the repulsive spline data, is present.
        if (rs_data := self.r_spline) is not None:
            grid = rs_data.grid
            # Header
            output += '\nSpline'
            # Grid data: {number of grid points} {cutoff}
            output += f'\n{len(grid):<5} {rs_data.cutoff:>12.8f}'
            # Exponential: {coefficients}
            output += '\n' + a2s(rs_data.exp_coef, '>21.12E')
            # Primary spline: {from} {to} {coefficients}
            s_data = t2a(torch.cat((grid[:-1].view(-1, 1),
                                    grid[1:].view(-1, 1),
                                    rs_data.spline_coef), -1))
            output += '\n' + a2s(s_data, '>21.12E')
            # Spline tail: {from} {to} {coefficients}
            tail = t2a(torch.cat((grid[-1:], rs_data.cutoff[None],
                                  rs_data.tail_coef)))
            output += '\n' + a2s(tail, '>21.12E')

        # Write the results to the target file
        open(path, 'w').write(output)

    def to_hdf5(self, target: Group):
        """Saves the `Skf` instance into a target HDF5 Group.

        Arguments:
            target: The hdf5 group to which the skf data should be saved.

        Notes:
            This function does not create its own group as it expects that
            ``target`` is the group into which data should be writen.
        """
        def t2n(t: Tensor) -> np.ndarray:
            """Convert torch tensor to a numpy array."""
            return t.detach().cpu().numpy()

        def add_data(entities: Dict[str, Any], to: str):
            """Create a new group and add multiple datasets to it."""
            to = target.create_group(to)
            for name, data in entities.items():
                # Convert any torch tensor into numpy arrays.
                data = t2n(data) if isinstance(data, Tensor) else data
                to.create_dataset(name, data=data)

        # Attributes
        target.attrs.update(
            {'atoms': t2n(self.atom_pair), 'version': self.version,
             'has_r_poly': self.r_poly is not None, 'is_atomic': self.atomic,
             'has_r_spline': self.r_spline is not None})

        # Convert electronic integral matrices into structured numpy arrays.
        dtype = np.dtype([('%s-%s' % k, np.float64, tuple(v.shape))
                          for k, v in self.hamiltonian.items()])
        h_data, s_data = [np.array(tuple(t2n(i) for i in j.values()), dtype)
                          for j in [self.hamiltonian, self.overlap]]

        # SkDict component
        add_data({'H': h_data, 'count': len(self.grid), 'S': s_data,
                  'step': self.grid.diff()[0]}, 'integrals')

        if (p := self.r_poly) is not None:  # Repulsive polynomial
            add_data({'coef': p.coef, 'cutoff': p.cutoff}, 'r_poly')

        if (s := self.r_spline) is not None:  # Repulsive spline
            add_data(
                {'grid': s.grid, 'cutoff': s.cutoff, 'exp_coef': s.exp_coef,
                 'step': s.grid.diff()[0], 'tail_coef': s.tail_coef,
                 'spline_coef': s.spline_coef, }, 'r_spline')

        if self.atomic:  # Atomic
            add_data(
                {'on_sites': self.on_sites, 'hubbard_us': self.hubbard_us,
                 'occupations': self.occupations, 'mass': self.mass}, 'atomic')

        # Metadata
        add_data({'time_created': time()}, 'metadata')

    def __str__(self) -> str:
        """Returns a string representing the `Skf` object."""
        cls_name = self.__class__.__name__
        name = '-'.join([chemical_symbols[int(i)] for i in self.atom_pair])
        r_spline = 'No' if self.r_spline is None else 'Yes'
        r_poly = 'No' if self.r_poly is None else 'Yes'
        atomic = 'No' if self.atomic is None else 'Yes'
        return f'{cls_name}({name}, r-spline: {r_spline}, r-poly: {r_poly}, ' \
               f'atomic-data: {atomic})'

    def __repr__(self) -> str:
        """Returns a simple string representation of the `Skf` object."""
        cls_name = self.__class__.__name__
        name = '-'.join([chemical_symbols[int(i)] for i in self.atom_pair])
        return f'{cls_name}({name})'

    @classmethod
    def _from_hdf5_helper(
            cls, source: Group, dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None) -> tuple:
        """This abstracts the contents of the `from_hdf5` function.

        This function acts as any other `from_hdf5` would be expected to with
        the exception that it returns the contents of the hdf5 group rather
        than a class instance representing them.

        This abstraction prevents having to duplicate the entire `from_hdf5`
        function in the `VCRSkf` class just to add a single line to allow
        for the compression radii to be extracted.
        """

        def tt(group, name):
            """Convenience function to convert to data into tensors"""
            return torch.tensor(group[name][()], **dd)

        dd = {'dtype': dtype, 'device': device}

        # Check that the version is compatible
        if float(source.attrs['version']) > float(cls.version):
            warnings.warn('HDF5-skf file Version exceeds local code version.')

        kwargs = {}
        atom_pair = torch.tensor(source.attrs['atoms'])

        # Retrieve integral data
        ints = source['integrals']
        # Convert structured numpy arrays into a dictionary of tensors
        H, S = ({
            # Convert name-string > Tuple[int, int] & np.array > torch.tensor
            tuple(map(int, n.split('-'))): torch.tensor(i[n], **dd)
            # Loop over field names (strings like "ℓ₁-ℓ₂") & their numpy arrays
            for n in i.dtype.fields.keys()}
            for i in [ints['H'], ints['S']])
        grid = (torch.arange(0, ints['count'][()], **dd) + 1) * ints['step'][()]

        if source.attrs['has_r_spline']:  # Repulsive spline data
            r = source['r_spline']
            kwargs['r_spline'] = cls.RSpline(
                tt(r, 'grid'), tt(r, 'cutoff'), tt(r, 'spline_coef'),
                tt(r, 'exp_coef'), tt(r, 'tail_coef'))

        if source.attrs['has_r_poly']:  # Repulsive polynomial
            r = source['r_poly']
            kwargs['r_poly'] = cls.RPoly(tt(r, 'cutoff'), tt(r, 'coef'))

        if source.attrs['is_atomic']:  # Atomic data
            a = source['atomic']
            kwargs.update({'on_sites': tt(a, 'on_sites'),
                           'hubbard_us': tt(a, 'hubbard_us'),
                           'mass': tt(a, 'mass'),
                           'occupations': tt(a, 'occupations')})

        return atom_pair, H, S, grid, kwargs


class VCRSkf(Skf):
    """Variable compression radius Slater-Koster file.

    This class handles the parsing of variable compression radius Slater-Koster
    binary files. These are similar to standard `Skf` files but differ in that
    their electronic integral tables iterate over the compression radii of the
    two species, not just over distance.

    Arguments:
        atom_pair: Atomic numbers of the elements associated with the
            interaction.
        hamiltonian: Dictionary keyed by azimuthal-number-pairs (ℓ₁, ℓ₂) and
            valued by m×r1×r2×d Hamiltonian integral tensors; where r1/r2 are
            the compression radii of atoms one/two and m & d iterate over the
            bond-order (σ, π, etc.) and distances respectively.
        overlap: Dictionary storing the overlap integrals in a manner akin to
            ``hamiltonian``.
        grid: Distances at which the ``hamiltonian`` & ``overlap`` elements
            were evaluated.
        compression_radii: Mesh-grid tensor specifying the compression radii.
        r_spline: A :class:`.RSpline` object detailing the repulsive
            spline. [DEFAULT=None]
        r_poly: A :class:`.RPoly` object detailing the repulsive
            polynomial. [DEFAULT=None]
        on_sites: On site terms, homo-atomic systems only. [DEFAULT=None]
        hubbard_us: Hubbard U terms, homo-atomic systems only. [DEFAULT=None]
        mass: Atomic mass, homo-atomic systems only. [DEFAULT=None]
        occupations: Occupations of the orbitals, homo-atomic systems only.
            [DEFAULT=None]

    Examples:
        Examples of reading the h5 file with VCR. Generation data set with VCR
        can be seen in `examples/example_01/example_02_setup.py`.

        >>> import urllib, torch
        >>> from os.path import join
        >>> from tbmalt.io.skf import Skf
        >>> link = 'https://zenodo.org/record/8109578/files/example_dftb_vcr.h5?download=1'
        >>> h5file = urllib.request.urlretrieve(link, path := join('./example_dftb_vcr.h5'))
        >>> skfh5 = Skf.read('./example_dftb_vcr.h5', torch.tensor([6, 6]))
        >>> print(skfh5.hamiltonian.keys())
        dict_keys([(0, 0), (0, 1), (1, 1)])

    Notes:
        Unlike their `Skf` parent class `VCRSkf` files cannot be stored as text
        files, only as HDF5 binaries.

    """
    version = '1.0'

    from_sfk = NotImplementedError('Not applicable')
    to_sfk = NotImplementedError('Not applicable')

    def __init__(self, atom_pair: Tensor, hamiltonian: SkDict, overlap: SkDict,
                 grid: Tensor, compression_radii: Tensor, *args, **kwargs):
        super().__init__(atom_pair, hamiltonian, overlap, grid, *args, **kwargs)
        self.compression_radii = compression_radii

    @classmethod
    def read(cls, path: str, atom_pair: Optional[Sequence[int]] = None,
             **kwargs) -> 'VCRSkf':
        """Load variable compression radii Slater-Koster data from a hdf5 file.

        Arguments:
            path: Path to the hdf5 database file that is to be read.
            atom_pair: Atomic numbers of the element pair whose interactions
                are to be loaded. This argument is mandatory when there is
                more than one element pair present in the database.
                [DEFAULT=None]

        Keyword Arguments:
            device: Device on which to place tensors. [DEFAULT=None]
            dtype: dtype to be used for floating point tensors. [DEFAULT=None]

        Returns:
            vcrskf: `VCRSkf` object containing the requested data.
        """
        with h5py.File(path, 'r') as db:
            # If atom_pair is specified use this to identify the target
            if atom_pair is not None:
                name = '-'.join([chemical_symbols[int(i)] for i in atom_pair])
            else:
                # Otherwise scan for valid entries: if only 1 SK entry exists
                # then assume it's the target; if multiple entries exist then
                # it's impossible to know which the user wanted.
                e = '[A-Z][a-z]*'
                entries = [k for k in db if re.fullmatch(f'{e}-{e}', k)]
                if len(entries) == 1:
                    name = entries[0]
                else:
                    raise ValueError(
                        'atom_pair must be specified when a database has more'
                        f' than one entry: {basename(path)}')

            return cls.from_hdf5(db[name], **kwargs)

    @classmethod
    def from_hdf5(cls, source: Group, dtype: Optional[torch.dtype] = None,
                  device: Optional[torch.device] = None) -> 'VCRSkf':
        """Instantiate a `VCRSkf` instances from an HDF5 group.

        Arguments:
            source: HDF5 group containing variable compression radius Slater-
                Koster data.
            device: Device on which to place tensors. [DEFAULT=None]
            dtype: dtype to be used for floating point tensors. [DEFAULT=None]

        Returns:
            vcrskf: The resulting `VCRSkf` object.
        """
        # Make a call to the proxy helper function to extract most of the data
        atom_pair, H, S, grid, kwargs = cls._from_hdf5_helper(
            source, dtype, device)
        # Extract the compression radii manually
        compression_radii = torch.tensor(
            source['integrals']['compression_radii'][()],
            dtype=dtype, device=device)

        return cls(atom_pair, H, S, grid, compression_radii, **kwargs)

    def write(self, path: str, overwrite: Optional[bool] = False):
        """Save the Slater-Koster data to a database.

        Arguments:
            path: path to the hdf5 in which the data is to be saved.
            overwrite: Existing HDF5-groups can only be overwritten when
                ``overwrite`` is True. [DEFAULT=False]

        """
        with h5py.File(path, 'a') as db:  # Create/open the HDF5 file
            name = '-'.join([chemical_symbols[int(i)]
                             for i in self.atom_pair])
            if name in db:  # If an entry already exists in this database
                if not overwrite:  # Then raise an exception
                    raise FileExistsError(
                        f'Entry {name} already exists; use "overwrite" '
                        'to permit overwriting.')
                else:  # Unless told to overwrite it
                    del db[name]
            # Create the HDF5 entry & fill it with data via `to_hdf5`.
            self.to_hdf5(db.create_group(name))

    def to_hdf5(self, target: Group):
        """Saves the `Skf` instance into a target HDF5 Group.

        Arguments:
            target: The hdf5 group to which the skf data should be saved.

        Notes:
            This function does not create its own group as it expects that
            ``target`` is the group into which data should be writen.
        """
        # Call the super's to_hdf5 method & add the compression radii data.
        super().to_hdf5(target)
        target['integrals'].create_dataset('compression_radii',
                                           data=self.compression_radii)

    @classmethod
    def from_dir(cls, source: str, target: str):
        """Parse multiple skf files into a hdf5 `VCRSkf` instance.

        When run against a `target` directory this method scans for map files
        which indicate the presence of data that can be used to build variable
        compression radii Slater-Koster files instance (`VCRSkf`). There are
        two types of source, the first is 'csv' type, the second is 'skf' type.

        Arguments:
            source: director holding the skf and map files.
            target: path to the hdf5 database in which the results should
                be stored.

        Notes:
            Directories may contain multiple maps & their associated sk-files.
            Map files must be csv formatted & named "skf_map_{X}_{Y}.csv" where
            "X" & "Y" are the elemental symbols of the two relevant species.
            Each line, "i", of the map file specifies the compression radii,
            for species "X" & "Y" respectively, used when making the sk-file
            "{X}_{Y}_{i}.skf". Note that indexing for "i" starts at one not
            zero and that data must be ordered so that it loops over values of
            "X" then over values of "Y", i.e:

                0.1,1.0
                0.1,2.0
                0.2,1.0
                0.2,2.0

            This map file example shows that there are two compression radii
            used for "X" and two for "Y", giving a total of four combinations.
            It is therefore expected that there are four sk-files.
            If the source is 'skf' type, the code will loop over files like:

                "{X}-{Y}-{i}-{j}.skf"
                "H-H-02.00-04.50.skf"
                "H-C-02.00-04.50.skf"

            For homo files, you also need one homo files, such as "H-H.skf" to
            offer homo properties, such as mass, on_sites or hubbard_us.

        """

        e = '[A-Z][a-z]'

        # Validate the path to the source directory
        if not isdir(source):
            raise NotADirectoryError(
                    f'`source` must be a valid directory: ({source})')

        # If the database exists; open it & identify what systems are present.
        # This avoids trying to parse systems that have already been parsed.
        current = []
        if exists(target):
            with h5py.File(target, 'a') as db:
                current += [tuple(k.split('-')) for k in db
                            if re.fullmatch(fr'{e}*-{e}*', k)]

        # Locate any map files present the directory
        all_files = glob.glob(join(source, '*'))
        if is_csv := re.search('csv$', all_files[0]):
            re_s1 = re.compile(fr'(?<=skf_map_){e}?_{e}?(?=.csv)').search
            map_files = {tuple(ele.group(0).split('_')): i for i in all_files
                         if (ele := re_s1(i))}
        else:
            re_s1 = re.compile(fr'(?<=\/){e}?-{e}?-\d+.\d+-\d+.\d+').search
            map_files = {tuple(ele.group(0).split('-')): i for i in all_files
                         if (ele := re_s1(i))}
            map_keys = np.array(list(map_files.keys()))
            atom_pair_names = np.unique(map_keys[..., :2], axis=0)
            re_ho = re.compile(fr'(?<=\/){e}?-{e}?.skf').search
            homo_files = {tuple(ele.group(0)[:-4].split('-')):
                              i for i in all_files if (ele := re_ho(i))}

        # Report systems found to the user [print header]
        print(f'Map files found for {len(map_files)} system(s):\n'
              '\tAtom 1\tAtom 2\tAction')

        # Loop over each system
        if is_csv:
            for pair, path in map_files.items():
                print(f'\t{pair[0]:6}\t{pair[1]:6}\t', end='')
                if pair in current:  # Skip it if it has been parsed before
                    print('SKIPPING (previously parsed)')
                    continue

                VCRSkf.single_csv(path, pair, all_files, target)
        else:
            for pair in atom_pair_names:
                this_mask = (map_keys[..., :2] == pair).all(-1)
                this_file_keys = sorted(map_keys[this_mask].tolist())
                this_list = []
                atom_pair = torch.tensor([atomic_numbers[i] for i in pair])
                n_r = len(np.unique(np.array(this_file_keys)[..., 2:]))
                assert len(this_file_keys) == n_r ** 2,\
                    'square of compression r number should be equal to file number'

                for key in this_file_keys:

                    this_list.append(Skf.read(map_files[tuple(key)]))

                h_data = {key: pack([isk.hamiltonian[key] for isk in this_list])
                          for key in this_list[-1].hamiltonian.keys()}
                h_data = {key: val.reshape(n_r, n_r, *val.shape[1:]).permute(
                    -2, -1, 0, 1) for key, val in h_data.items()}
                s_data = {key: pack([isk.overlap[key] for isk in this_list])
                          for key in this_list[-1].overlap.keys()}
                s_data = {key: val.reshape(n_r, n_r, *val.shape[1:]).permute(
                    -2, -1, 0, 1) for key, val in s_data.items()}

                # Grid points for distances and compression radii
                grid = this_list[-1].grid
                cr_grid = torch.from_numpy(
                    np.unique(np.array(this_file_keys)[..., 2:].astype(dtype=np.float64)))

                if atom_pair[0] == atom_pair[1]:
                    # Load the file into a Skf instance to extract the data from
                    skf_homo = Skf.read(homo_files[tuple(pair.tolist())])
                    vcrskf = cls(
                        atom_pair, h_data, s_data, grid, cr_grid, mass=skf_homo.mass,
                        on_sites=skf_homo.on_sites, hubbard_us=skf_homo.hubbard_us,
                        occupations=skf_homo.occupations)
                else:
                    vcrskf = cls(atom_pair, h_data, s_data, grid, cr_grid)

                # Store data into the target database.
                with h5py.File(target, 'a') as db:
                    group = db.create_group(f'{pair[0]}-{pair[1]}')
                    vcrskf.to_hdf5(group)

    @classmethod
    def single_csv(cls, path, pair, all_files, target):
        # Open the map file and extract the compression radii pair list
        radii = np.genfromtxt(path, delimiter=',')

        # Number of unique radii for species 1 & 2
        n_r1, n_r2 = [len(set(i)) for i in radii.T]

        # Locate the skf files and order them correctly
        re_s2 = re.compile(rf'{pair[0]}.*{pair[1]}.*\d+.skf').search
        re_s3 = re.compile(r'\d+(?=.skf)').search
        sk_files = np.array([*filter(re_s2, all_files)])
        sk_numbers = np.array([int(re_s3(f).group(0)) for f in sk_files])

        sorter = sk_numbers.argsort()
        sk_files = sk_files[sorter]
        sk_numbers = sk_numbers[sorter]

        # Ensure the expected number of files are found and are sequential
        if (n := len(sk_files)) != (m := len(radii)):
            raise IndexError(f'Read {m} values from {splitext(path)} '
                             f'but found only {n} skf files.')

        if not np.allclose(np.diff(sk_numbers), 1):
            raise IndexError('Non-sequential skf files found.')

        # Work out which lines store the electronic integrals.
        t = open(sk_files[0]).readlines()
        o = int(t[0].startswith('@'))  # offset for comment if present
        n_grid = int(t[o].split()[1])  # Number of lines to get
        g_dist = float(t[o].split()[0])

        # Electronic integrals will always start at line "2 + o" & "3 + o"
        # for triatomic and homoatomic systems respectively.
        s_hetro = slice(2 + o, 2 + o + n_grid)
        s_homo = slice(3 + o, 3 + o + n_grid)

        # When dealing with homoatomic systems only the files where the
        # two atoms have the same compression radii will actually contain
        # homoatomic data. Thus, a list of slicers is created to take this
        # into account when reading.
        slicers = [s_homo if (r1 == r2 and pair[0] == pair[1])
                   else s_hetro for r1, r2 in
                   zip(radii[:, 0], radii[:, 1])]

        # Read the electronic integrals from all files and place into a
        # dictionary; method similar to that used in `Skf.from_skf`.
        cat = ''.join
        h_data, s_data = np.split(
            np.fromstring(
                cat([cat(open(i).readlines()[s])
                     for i, s in zip(sk_files, slicers)]),
                sep=' ', dtype=np.float64
            ).reshape(len(sk_files), n_grid, -1),
            2, -1)

        l_max = int(tetrahedral_root(h_data.shape[-1]))
        l_pairs = np.stack(np.triu_indices(l_max)).T
        sort = cls._sorter if l_max == 3 else cls._sorter_e
        splitter = np.cumsum(l_pairs[:, 0] + 1)[:-1]

        shape = (-1, n_r1, n_r2, n_grid)  # Reshapes integrals to grid
        h_data, s_data = [{
            tuple(l_pair.tolist()): torch.tensor(i.reshape(shape))
            for l_pair, i in
            zip(l_pairs, np.split(i.transpose((2, 0, 1))[sort], splitter, 0))
            if not (i == 0.).all()}
            for i in [h_data, s_data]]

        # Parse the data into a VCRSkf instance
        atom_pair = torch.tensor([atomic_numbers[i] for i in pair])
        grid = torch.arange(1, n_grid + 1) * g_dist
        cr_grid = torch.stack(torch.meshgrid(
            [torch.tensor(sorted(list(set(i)))) for i in radii.T]))

        if pair[0] == pair[1]:  # Load homoatomic data if appropriate
            # Load the file into an Skf instance to extract the data from
            skf = Skf.from_skf(sk_files[0])
            vcrskf = cls(
                atom_pair, h_data, s_data, grid, cr_grid, mass=skf.mass,
                on_sites=skf.on_sites, hubbard_us=skf.hubbard_us,
                occupations=skf.occupations)
        else:
            vcrskf = cls(atom_pair, h_data, s_data, grid, cr_grid)

        # Store data into the target database.
        with h5py.File(target, 'a') as db:
            group = db.create_group(f'{pair[0]}-{pair[1]}')
            vcrskf.to_hdf5(group)

    @classmethod
    def from_dir_raw(cls, source: str, target: str):
        """Parse multiple skf files into a hdf5 `VCRSkf` instance.

        When run against a `target` directory this method scans for map files
        which indicate the presence of data that can be used to build variable
        compression radii Slater-Koster files instance (`VCRSkf`).

        Arguments:
            source: director holding the skf and map files.
            target: path to the hdf5 database in which the results should
                be stored.

        Notes:
            Directories may contain multiple maps & their associated sk-files.
            Map files must be skf formatted & named "{X}-{Y}-{Rx}-{Ry}.skf"
            where "X" & "Y" are the elemental symbols of the two relevant
            species. "Rx" and "Ry" are the corresponding compression radii.
            An example file format: "H-H-03.00-03.00.skf".


        """
        e = '[A-Z][a-z]'

        # Validate the path to the source directory
        if not isdir(source):
            raise NotADirectoryError(
                    f'`source` must be a valid directory: ({source})')

        # If the database exists; open it & identify what systems are present.
        # This avoids trying to parse systems that have already been parsed.
        current = []
        if exists(target):
            with h5py.File(target, 'a') as db:
                current += [tuple(k.split('-')) for k in db
                            if re.fullmatch(fr'{e}*-{e}*', k)]

        # Locate any map files present the directory
        all_files = glob.glob(join(source, '*'))
        re_s1 = re.compile(fr'(?<=\/){e}?-{e}?-[0-9]?.[0-9]?-[0-9].[0-9].skf').search
        map_files = {tuple(ele.group(0).split('-')): i for i in all_files
                     if (ele := re_s1(i))}

        # Report systems found to the user [print header]
        print(f'Map files found for {len(map_files)} system(s):\n'
              '\tAtom 1\tAtom 2\tAction')

        # Loop over each system
        for pair, path in map_files.items():
            print(f'\t{pair[0]:6}\t{pair[1]:6}\t', end='')
            if pair in current:  # Skip it if it has been parsed before
                print('SKIPPING (previously parsed)')
                continue


#########################
# Convenience Functions #
#########################
def _s2t(text: Union[str, List[str]], sep: str = ' \t', **kwargs) -> Tensor:
    """Converts string to tensor.

    This uses the `np.fromstring` method to quickly convert blocks of text
    into arrays, which are then converted into tensors.

    Arguments:
        text: string to extract the tensor from. If a list of strings is
            supplied then they will be joined prior to tensor extraction.
        sep: possible delimiters. [DEFAULT=' \t']

    Keyword Arguments:
        kwargs: these will be passed into the `torch.tensor` call.

    """
    text = sep.join(text) if isinstance(text, list) else text
    return torch.tensor(np.fromstring(text, sep=sep, dtype=np.float64),
                        **kwargs)


def _esr(text: str) -> str:
    """Expand stared number representations.

    This is primarily used to resolve the skf file specification violations
    which are found in some early skf files. Specifically the user of
    started notations like `10*1.0` to represent a value of one repeated ten
    times, or the mixed use of spaces, tabs and commas.

    Arguments:
        text: string to be rectified.

    Returns:
        r_text: rectified string.

    Notes:
        This finds strings like `3*.0` & `10*1` and replaces them with
        `.0 .0 .0` & `1 1 1 1 1 1 1 1 1 1` respectively.
    """
    # Strip out unnecessary commas
    text = text.replace(',', ' ')
    if '*' in text:
        for i in set(re.findall(r'[0-9]+\*[0-9|.]+', text)):
            n, val = i.strip(',').split('*')
            text = text.replace(i, f"{' '.join([val] * int(n))}")
    return text
