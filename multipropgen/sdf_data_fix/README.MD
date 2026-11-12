## QM9 SDF Bond and Charge Fix Pipeline

This directory provides a three-step pipeline to correct bond orders, formal charges, and structural issues in the QM9 SDF dataset.

### Step 1: Generate Fixed Version
Run the bond/charge fixing script on the original QM9 SDF file (requires precomputed InChI list):

```bash
python qm9_bond_charge_fix.py \
  --inchi_pkl path/to/inchi_list.pkl \
  --input_sdf path/to/original_qm9.sdf \
  --output_sdf path/to/fixed_qm9.sdf \
  --bad_indices bad_indices.npy
```

Outputs a first-pass corrected SDF (`fixed_qm9.sdf`) plus:
- `bad_indices.npy`: indices of molecules that could not be fixed in this stage

Fixes may include:
- Improper / missing bond orders
- Incorrect formal charges
- Valence inconsistencies
- Connectivity transferred from original InChl

### Step 2: Compare and Replace
Use the fixed SDF from Step 1 and replace molecules based on InChl matching:

```bash
python compare_and_replace.py \
    --sdf path/to/fixed_qm9.sdf \
    --inchi path/to/inchi_list.pkl \
    --may_not_match path/to/maynotmatch.npy \
    --out rQM9.sdf \
    --log replaced.log
```

`maynotmatch.npy` corresponds to the indices of the uncharacterized cases where the SMILES after optimization does not match the original input SMILES for QM9.

This produces `rQM9.sdf`:
- Replaces entries where an improved (corrected) structure is validated via InChl
- Logs replacements to `replaced.log`

### Step 3: Detect Remaining Problematic Molecules
Scan the refined file for any molecules still exhibiting structural issues:

```bash
python find_problematic_mols.py \
  --input_sdf path/to/rQM9.sdf \
  --output_indices problematic_indices.npy
```

Generates a list (`problematic_indices.npy`) of molecule indices requiring manual inspection.

### Outputs Summary
- Intermediate: `fixed_qm9.sdf` (first-pass corrections)
- Refined Final: `rQM9.sdf` (post-replacement, INCHI-validated)
- Diagnostics: `problematic_indices.npy` (indices of unresolved cases)
- Logs: `replaced.log` (replacement details), `bad_indices.npy` (Step 1 unfixed indices)

### Notes
- All scripts expect SDF files with explicit hydrogens.
- Indices reported correspond to zero-based order in the processed SDF supplier.
- Use `rQM9.sdf` for downstream featurization and training.

### Example End-to-End
```bash
python multipropgen/sdf_data_fix/qm9_bond_charge_fix.py --inchi_pkl data/qm9_raw/inchi_list.pkl --input_sdf data/qm9_raw/gdb9.sdf --output_sdf data/qm9_raw/fixed.sdf --bad_indices data/qm9_raw/bad_indices.npy
python multipropgen/sdf_data_fix/compare_and_replace.py --sdf data/qm9_raw/fixed.sdf --inchi data/qm9_raw/inchi_list.pkl --plain_smiles data/qm9_raw/plain_smiles_list.pkl --may_not_match data/qm9_raw/maynotmatch.npy --out data/qm9_raw/rQM9.sdf --log data/qm9_raw/replaced.log
python multipropgen/sdf_data_fix/find_problematic_mols.py --input_sdf data/qm9_raw/rQM9.sdf --output_indices data/qm9_raw/problematic_indices.npy
```

After Step 3, inspect any remaining problematic indices before large-scale processing.
