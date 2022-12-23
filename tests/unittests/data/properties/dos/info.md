# DoS Test Data
## Introduction
A selection of DFT level reference data-points calculated using FHI-aims. This data is
intended to be used when testing the density of states functionality of TBMaLT. Within
this directory are three subdirectories:
* H2
* CH4
* HCOOH

### Raw Data
Each of which holding a different molecular system. Within each molecular system's directory
there are 8 csv file archetypes, each of which is detailed below:
<details>
    <summary>Molecular System Files</summary>
    <blockquote>
        <details>
            <summary>eigenvalues.csv</summary>
            Eigen-values, aka energy levels, of the associated system (Units: Hartree).
        </details>
        <details>
            <summary>eigenvectors.csv</summary>
            Eigen-vectors, aka coefficient matrix, in column form (Units: n/a).
        </details>
        <details>
            <summary>overlap.csv</summary>
            Overlap matrix (Units: n/a).
        </details>
        <details>
            <summary>fermi.csv</summary>
            Fermi energy (Units: Hartree).
        </details>
        <details>
            <summary>sigma.csv</summary>
            Smearing width used when generating the reference DoSs & PDoSs (Units: Hartree).
        </details>
        <details>
            <summary>dos.csv</summary>
            Density of states (DoS), and the energy values at which it was evaluated
            (Units: Hartree).
        </details>
        <details>
            <summary>pdos_{element}.csv</summary>
            Projected densities of states (PDoS), and the energy values at which they
            were evaluated. One file is produced per element. (Units: Hartree).
        </details>
        <details>
            <summary>bases.csv</summary>
            Information about the various basis functions (Units: n/a).
        </details>
    </blockquote>
</details>

### Parsed Data
The aforementioned datasets are parsed into a recursive dictionaries which are made available
as module level variables. These variables are named after the system whose data they contain.
Each of the molecular properties made available by this dictionary have been listed and detailed
below:
<details>
    <summary>Data Dictionaries' Contents</summary>
    <blockquote>
        <details>
            <summary>eigenvalues</summary>
            1D tensor holding the eigenvalues.
        </details>
        <details>
            <summary>eigenvectors</summary>
            2D tensor whose columns correspond to the system's eigenvectors.
        </details>
        <details>
            <summary>overlap</summary>
            2D tensor holding the overlap matrix.
        </details>
        <details>
            <summary>fermi</summary>
            Float representing the fermi energy value.
        </details>
        <details>
            <summary>sigma</summary>
            Float representing the smearing value.
        </details>
        <details>
            <summary>dos</summary>
            Dictionary with:
                <li>"total":  tensor holding the molecule's DoS.</li>
                <li>"energy": tensor listing the energy values at which said DoS was
                    evaluated.</li>
        </details>
        <details>
            <summary>pdos</summary>
            Dictionary with a sub-dictionary per element, (H, C, etc.) ,each containing:
                <li>"energy": Tensor listing the energy values at which the PDoSs were
                    evaluated.</li>
                <li>"total": total PDoS associated with the element.</li>
                <li>"0": PDoS associated with the s orbitals.</li>
                <li>"1": PDoS associated with the p orbitals.</li>
                <li>"2": PDoS associated with the d orbitals.</li>
                <li>"{l}": PDoS associated with the {l} orbitals, and so on.</li>
        </details>
        <details>
             <summary>bases</summary>
             Dictionary keyed by strings and valued by integer tensors:
        </details>
    </blockquote>
</details>


#### Footnotes:
* DFT: Density Functional Theory.
* FHI-Aims: Fritz-Haber-Institute ab initio simulation package.

