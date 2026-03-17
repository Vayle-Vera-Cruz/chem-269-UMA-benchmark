#!/usr/bin/env python

import sys
from ase.io import read, write
from ase import Atoms
from ase.optimize import BFGS

from fairchem.core import pretrained_mlip, FAIRChemCalculator


def main(xyz_file: str, out_file: str = "relaxed_simple.xyz",
         charge=0, spin=1):

    # 1. Read initial geometry
    atoms = read(xyz_file)

    # 2. Set molecular charge and spin multiplicity
    atoms.info["charge"] = charge        # total charge (integer)
    atoms.info["spin"] = spin            # multiplicity: 1=singlet, 2=doublet, etc.

    # 3. Set up UMA calculator
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cpu")
    calc = FAIRChemCalculator(predictor, task_name="omol")
    atoms.calc = calc

    # 4. Optimize with BFGS
    opt = BFGS(atoms, logfile="opt_bfgs.log")
    opt.run(fmax=0.05)

    # 5. Compute and print final energy
    energy = atoms.get_potential_energy()
    print(f"Final potential energy: {energy:.6f} eV")

    # 6. Build clean output structure with no calculator or forces
    relaxed = Atoms(
        symbols=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        cell=atoms.cell,
        pbc=atoms.pbc,
    )

    write(out_file, relaxed, format="xyz")
    print(f"Relaxed geometry written to {out_file}")

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage: python UMA_OPT_BFGS.py input.xyz "
            "[output.xyz] [charge] [spin]"
        )
        sys.exit(1)

    xyz_in = sys.argv[1]

    # Optional output filename
    out_file = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].endswith(".xyz") \
        else "relaxed_simple.xyz"

    # Optional charge and spin
    arg_offset = 3 if out_file != "relaxed_simple.xyz" else 2
    charge = int(sys.argv[arg_offset]) if len(sys.argv) > arg_offset else 0
    spin = int(sys.argv[arg_offset + 1]) if len(sys.argv) > arg_offset + 1 else 1

    main(xyz_in, out_file, charge=charge, spin=spin)

