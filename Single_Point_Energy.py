import sys
from ase.io import read
from fairchem.core import pretrained_mlip, FAIRChemCalculator

def main():
    # ---- command-line arguments ----
    # python single_point_uma.py structure.xyz [charge] [multiplicity]
    if len(sys.argv) not in (2, 4):
        print("Usage: python single_point_uma.py <structure.xyz> [charge] [multiplicity]")
        print("Example: python single_point_uma.py molecule.xyz 0 1")
        sys.exit(1)

    xyz_file = sys.argv[1]

    # Defaults
    charge = 0
    multiplicity = 1

    if len(sys.argv) == 4:
        charge = int(sys.argv[2])
        multiplicity = int(sys.argv[3])

    # -------- settings --------
    device = "cpu"        # use "cuda" if available
    task_name = "omol"    # molecules / polymers
    model_name = "uma-s-1p1"
    # --------------------------

    # Read structure
    atoms = read(xyz_file)

    # ASE / FAIR-Chem convention:
    #   spin = number of unpaired electrons = multiplicity - 1
    spin = multiplicity - 1

    atoms.info.update({
        "charge": charge,
        "spin": spin
    })

    # Load UMA model
    predictor = pretrained_mlip.get_predict_unit(
        model_name,
        device=device
    )

    # Attach calculator
    atoms.calc = FAIRChemCalculator(
        predictor,
        task_name=task_name
    )

    # Single-point energy
    energy = atoms.get_potential_energy()  # eV

    print(f"File: {xyz_file}")
    print(f"Charge: {charge}")
    print(f"Spin multiplicity: {multiplicity}")
    print(f"Single-point energy: {energy:.6f} eV")


if __name__ == "__main__":
    main()

