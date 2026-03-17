#!/usr/bin/env python

import argparse
import numpy as np
from ase.io import read, write
from ase import Atoms, units
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo

from fairchem.core import pretrained_mlip, FAIRChemCalculator


def main(
    xyz_file: str,
    charge: int = 0,
    multiplicity: int = 1,
    model: str = "uma-s-1p1",
    device: str = "cpu",
    fmax: float = 0.05,
    max_steps: int = 500,
    vib_name: str = "vib",
    delta: float = 0.01,
    temperature: float = 298.15,
    pressure: float = 101325.0,  # 1 atm in Pa
    geometry: str = "nonlinear",         # "linear" or "nonlinear"
    symmetrynumber: int = 1,
    relaxed_xyz: str = "relaxed_simple.xyz",
    ignore_imag: bool = True,
    imag_energy_threshold: float = 0.0,  # eV; use small positive (e.g. 1e-4) to drop near-zero too
):
    # -------------------------
    # 1) Read input geometry
    # -------------------------
    atoms = read(xyz_file, index=0)

    # UMA molecular task metadata
    atoms.info["charge"] = int(charge)          # total charge (integer)
    atoms.info["spin"] = int(multiplicity)      # multiplicity: 1 singlet, 2 doublet, ...

    # -------------------------
    # 2) UMA calculator (CPU/MPS)
    # -------------------------
    predictor = pretrained_mlip.get_predict_unit(model, device=device)
    atoms.calc = FAIRChemCalculator(predictor, task_name="omol")

    # -------------------------
    # 3) Geometry optimization (BFGS)
    # -------------------------
    opt = BFGS(atoms, logfile="opt_bfgs.log")
    opt.run(fmax=fmax, steps=max_steps)

    E_elec = atoms.get_potential_energy()  # eV
    print(f"\nOptimized electronic energy (UMA): {E_elec:.6f} eV")

    # Write a SIMPLE XYZ (symbols + positions only)
    relaxed = Atoms(
        symbols=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        cell=atoms.cell,
        pbc=atoms.pbc,
    )
    write(relaxed_xyz, relaxed, format="xyz")
    print(f"Relaxed geometry written to: {relaxed_xyz}")
    # -------------------------
    # 4) Vibrations
    # -------------------------
    vib = Vibrations(atoms, name=vib_name, delta=delta)
    vib.run()
    vib.summary()

    import numpy as np
    vib_energies = np.array(vib.get_energies(), dtype=float)
    vib.clean()

    # -------------------------
    # 5) Thermochemistry
    # -------------------------
    S = (int(multiplicity) - 1) / 2.0

    if ignore_imag:
        thresh = max(float(imag_energy_threshold), 1e-8)
        mask = vib_energies > thresh
        vib_energies_for_thermo = vib_energies[mask].tolist()

        n_removed = int((~mask).sum())
        if n_removed > 0:
            print(
                f"\nWARNING: Removed {n_removed} mode(s) with vibrational energy "
                f"<= {thresh} eV (imaginary and/or near-zero)."
            )
    else:
        vib_energies_for_thermo = vib_energies.tolist()

    if any(e <= 0.0 for e in vib_energies_for_thermo):
        raise RuntimeError(
            "Filtered vib_energies still contain non-positive values. "
            "Increase --imag_energy_threshold."
        )

    thermo = IdealGasThermo(
        vib_energies=vib_energies_for_thermo,
        potentialenergy=E_elec,
        atoms=atoms,
        geometry=geometry,
        symmetrynumber=int(symmetrynumber),
        spin=S,
    )

    T = float(temperature)
    P = float(pressure)

    G = thermo.get_gibbs_energy(temperature=T, pressure=P)
    H = thermo.get_enthalpy(temperature=T)
    S_th = thermo.get_entropy(temperature=T, pressure=P)

    print("\n=== Ideal Gas Thermochemistry (ASE) ===")
    print(f"T = {T:.2f} K")
    print(f"P = {P:.2f} Pa")
    print(f"H  = {H:.6f} eV")
    print(f"S  = {S_th:.6e} eV/K")
    print(f"G  = {G:.6f} eV")

    with open("thermo_summary.txt", "w") as f:
        f.write(f"Input XYZ: {xyz_file}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Charge: {charge}\n")
        f.write(f"Multiplicity: {multiplicity}\n")
        f.write(f"Electronic energy (eV): {E_elec:.6f}\n")
        f.write(f"Temperature (K): {T:.2f}\n")
        f.write(f"Pressure (Pa): {P:.2f}\n")
        f.write(f"Geometry: {geometry}\n")
        f.write(f"Symmetry number: {symmetrynumber}\n")
        f.write(f"Ignore imaginary: {ignore_imag}\n")
        f.write(f"Imag energy threshold (eV): {imag_energy_threshold}\n")
        f.write(f"Modes total: {len(vib_energies)}\n")
        f.write(f"Modes used for thermo: {len(vib_energies_for_thermo)}\n")
        f.write(f"H (eV): {H:.6f}\n")
        f.write(f"S (eV/K): {S_th:.6e}\n")
        f.write(f"G (eV): {G:.6f}\n")

    print("Thermochemistry summary written to: thermo_summary.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize a molecule with UMA (BFGS), then compute Gibbs free energy from ASE vibrations."
    )
    parser.add_argument("xyz", help="Input XYZ file (single structure)")
    parser.add_argument("--charge", type=int, default=0, help="Total molecular charge (integer, default: 0)")
    parser.add_argument("--multiplicity", type=int, default=1, help="Spin multiplicity (1 singlet, 2 doublet, ...)")
    parser.add_argument("--model", default="uma-s-1p1", help="UMA model name (default: uma-s-1p1)")
    parser.add_argument("--device", default="cpu", help='Device: "cpu" or "mps" (Apple Silicon)')

    parser.add_argument("--fmax", type=float, default=0.05, help="Optimization convergence criterion (eV/Å)")
    parser.add_argument("--max_steps", type=int, default=500, help="Max BFGS steps (default: 500)")

    parser.add_argument("--vib_name", default="vib", help="Prefix for vibration scratch files (default: vib)")
    parser.add_argument("--delta", type=float, default=0.01, help="Finite-difference step in Å (default: 0.01)")

    parser.add_argument("--temperature", type=float, default=298.15, help="Temperature in K (default: 298.15)")
    parser.add_argument(
        "--pressure",
        type=float,
        default=101325.0,
        help="Pressure in Pa (default: 101325 Pa = 1 atm)",
    )

    parser.add_argument("--geometry", choices=["linear", "nonlinear"], default="nonlinear",
                        help='Molecular geometry: "linear" or "nonlinear"')
    parser.add_argument("--symmetrynumber", type=int, default=1, help="Rotational symmetry number (default: 1)")

    parser.add_argument("--relaxed_xyz", default="relaxed_simple.xyz", help="Output relaxed XYZ filename")

    parser.add_argument("--ignore_imag", action="store_true",
                        help="Ignore imaginary/near-zero vibrational modes in thermochemistry")
    parser.add_argument("--imag_energy_threshold", type=float, default=0.0,
                        help="Threshold (eV) below which vib modes are excluded if --ignore_imag is set "
                             "(e.g. 1e-4 to drop near-zero too)")

    args = parser.parse_args()

    main(
        xyz_file=args.xyz,
        charge=args.charge,
        multiplicity=args.multiplicity,
        model=args.model,
        device=args.device,
        fmax=args.fmax,
        max_steps=args.max_steps,
        vib_name=args.vib_name,
        delta=args.delta,
        temperature=args.temperature,
        pressure=args.pressure,
        geometry=args.geometry,
        symmetrynumber=args.symmetrynumber,
        relaxed_xyz=args.relaxed_xyz,
        ignore_imag=args.ignore_imag,
        imag_energy_threshold=args.imag_energy_threshold,
    )

