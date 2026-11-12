import os
import argparse
import json
import math
import numpy as np
import torch
from pathlib import Path
from rdkit import Chem
from typing import List

# FlowMol imports
from multipropgen.models.flowmol import FlowMol
from multipropgen.models.flowmol_multi import FlowMolMutli
from multipropgen.analysis.metrics import SampleAnalyzer
from multipropgen.analysis.multipropgen_metrics import compute_all_standard_metrics

def float_or_json(value):
    # try float first
    try:
        return float(value)
    except ValueError:
        pass

    # try JSON
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError(
            f"--properties_for_sampling must be a float or JSON, got: {value}"
        )

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate molecules with MultiPropGen.")
    
    # Model checkpoint
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path to the model checkpoint file")
    
    parser.add_argument('--sample-multi', action='store_true', 
                        help='Whether to sample from multi-property model')

    # Sampling parameters
    parser.add_argument("--n_mols", type=int, default=1000,
                        help="Number of molecules to sample")
    parser.add_argument("--max_batch_size", type=int, default=128,
                        help="Maximum batch size for sampling")
    parser.add_argument("--n_atoms_per_mol", type=int, default=None,
                        help="Number of atoms per molecule (if fixed)")
    parser.add_argument("--n_timesteps", type=int, default=100,
                        help="Number of timesteps for sampling")
    
    # Trajectory and metrics options
    parser.add_argument("--xt_traj", action="store_true",
                        help="Store x_t trajectory")
    parser.add_argument("--ep_traj", action="store_true",
                        help="Store episode trajectory")
    
    # Stochasticity and thresholds
    parser.add_argument("--stochasticity", type=float, default=None,
                        help="Stochasticity parameter")
    parser.add_argument("--hc_thresh", type=float, default=None,
                        help="High confidence threshold")
    
    # Property conditioning - Single property
    parser.add_argument("--properties_for_sampling", type=float_or_json, default=None,
                        help="Property value (float for single, JSON dict for multi)")
    parser.add_argument("--property_name", type=str, default=None,
                        help="Property name (for single property)")
    
    # Property conditioning - Multi property
    parser.add_argument("--property_names", type=str, nargs='+', default=None,
                        help="List of property names for multi-property model (e.g., --property_names homo lumo gap)")
    
    # Common options
    parser.add_argument("--training_mode", action="store_true",
                        help="Use training mode")
    parser.add_argument("--normalization_file_path", type=str, default=None,
                        help="Path to property normalization file")
    parser.add_argument("--properties_handle_method", type=str, default="concatenate_sum",
                        help="Method to handle properties")
    parser.add_argument("--multilple_values_file", type=str, default=None,
                        help="Path to numpy (.npy) or npz file containing multiple property values")
    parser.add_argument("--number_of_atoms", type=str, default=None,
                        help="Path to numpy file containing number of atoms per molecule")

    # Output
    parser.add_argument("--output_file", type=str, default="sampled_molecules.sdf",
                        help="Output SDF file path")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze generated molecules")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID (or -1 for CPU)")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_checkpoint}")
    if args.sample_multi:
        model = FlowMolMutli.load_from_checkpoint(args.model_checkpoint, strict=False)
        print("Loaded multi-property model")
    else:
        model = FlowMol.load_from_checkpoint(args.model_checkpoint, strict=False)
        print("Loaded single-property model")
    model.to(device)
    model.eval()
    
    # Initialize variables
    multilple_values_to_one_property = None
    number_of_atoms = None
    property_names = None
    
    # Load conditioning data based on model type
    if args.sample_multi:
        # MULTI-PROPERTY MODEL
        if args.multilple_values_file is not None:
            # Load from npz file with multiple properties
            print(f"Loading multiple property values from {args.multilple_values_file}")
            data = np.load(args.multilple_values_file)
            
            # Get property names from file or arguments
            if 'property_names' in data:
                property_names = data['property_names'].tolist()
            elif args.property_names is not None:
                property_names = args.property_names
            else:
                raise ValueError("property_names must be provided either in npz file or via --property_names")
            
            # Load property values as dictionary
            multilple_values_to_one_property = {}
            for prop_name in property_names:
                if prop_name not in data:
                    raise ValueError(f"Property '{prop_name}' not found in {args.multilple_values_file}")
                multilple_values_to_one_property[prop_name] = data[prop_name].tolist()[:args.n_mols]
            
            # Load number of atoms if available
            if 'n_atoms' in data:
                number_of_atoms = data['n_atoms'].tolist()[:args.n_mols]
            elif args.number_of_atoms is not None:
                print(f"Loading number of atoms from {args.number_of_atoms}")
                number_of_atoms = np.load(args.number_of_atoms).tolist()[:args.n_mols]
            
            print(f"Loaded {len(next(iter(multilple_values_to_one_property.values())))} values for properties: {property_names}")
            
        else:
            # Use single values for all molecules (properties_for_sampling should be a dict)
            if args.properties_for_sampling is not None:
                if not isinstance(args.properties_for_sampling, dict):
                    raise ValueError("For multi-property model, --properties_for_sampling must be a JSON dict")
                
                if args.property_names is None:
                    raise ValueError("--property_names must be provided for multi-property sampling")
                
                property_names = args.property_names
                # properties_for_sampling is already a dict, will be used directly
                print(f"Using single property values for all molecules: {args.properties_for_sampling}")
            
            # Load number of atoms
            if args.number_of_atoms is not None:
                print(f"Loading number of atoms from {args.number_of_atoms}")
                number_of_atoms = np.load(args.number_of_atoms).tolist()[:args.n_mols]
    
    else:
        # SINGLE-PROPERTY MODEL
        if args.multilple_values_file is not None:
            # Load list of property values (one per molecule)
            print(f"Loading multiple property values from {args.multilple_values_file}")
            multilple_values_to_one_property = np.load(args.multilple_values_file).tolist()[:args.n_mols]
            print(f"Loaded {len(multilple_values_to_one_property)} property values")
        
        # Load number of atoms
        if args.number_of_atoms is not None:
            print(f"Loading number of atoms from {args.number_of_atoms}")
            number_of_atoms = np.load(args.number_of_atoms).tolist()[:args.n_mols]

    # Sample molecules in batches
    n_batches = math.ceil(args.n_mols / args.max_batch_size)
    molecules = []

    print(f"\nSampling {args.n_mols} molecules in {n_batches} batches")
    if args.sample_multi and property_names:
        print(f"Conditioning on properties: {property_names}\n")
    elif args.property_name:
        print(f"Conditioning on property: {args.property_name}\n")

    for batch_idx in range(n_batches):
        n_mols_needed = args.n_mols - len(molecules)
        batch_size = min(n_mols_needed, args.max_batch_size)
        
        # Extract batch-specific data
        batch_property = None
        batch_no_of_atoms = None
        
        # Slice property values for this batch
        if multilple_values_to_one_property is not None:
            start_idx = len(molecules)
            end_idx = start_idx + batch_size
            
            if args.sample_multi:
                # Multi-property: slice each property's values
                batch_property = {}
                for prop_name, prop_values in multilple_values_to_one_property.items():
                    batch_property[prop_name] = prop_values[start_idx:end_idx]
            else:
                # Single-property: slice list
                batch_property = multilple_values_to_one_property[start_idx:end_idx]
        
        # Slice number of atoms for this batch
        if number_of_atoms is not None:
            start_idx = len(molecules)
            end_idx = start_idx + batch_size
            batch_no_of_atoms = number_of_atoms[start_idx:end_idx]

        # Prepare sampling kwargs
        sample_kwargs = {
            'device': device,
            'n_timesteps': args.n_timesteps,
            'xt_traj': args.xt_traj,
            'ep_traj': args.ep_traj,
            'stochasticity': args.stochasticity,
            'high_confidence_threshold': args.hc_thresh,
            'training_mode': args.training_mode,
            'normalization_file_path': args.normalization_file_path,
            'properties_handle_method': args.properties_handle_method,
        }
        
        # Add model-specific parameters
        if args.sample_multi:
            # Multi-property model
            sample_kwargs['property_names'] = property_names
            if batch_property is not None:
                sample_kwargs['multilple_values_to_one_property'] = batch_property
            elif args.properties_for_sampling is not None:
                sample_kwargs['properties_for_sampling'] = args.properties_for_sampling
        else:
            # Single-property model
            sample_kwargs['property_name'] = args.property_name
            if batch_property is not None:
                sample_kwargs['multilple_values_to_one_property'] = batch_property
            elif args.properties_for_sampling is not None:
                sample_kwargs['properties_for_sampling'] = args.properties_for_sampling

        # Sample molecules
        if args.n_atoms_per_mol is None:
            # Random sizes
            sample_kwargs['number_of_atoms'] = batch_no_of_atoms
            batch_molecules = model.sample_random_sizes(batch_size, **sample_kwargs)
        else:
            # Fixed size
            n_atoms = torch.full((batch_size,), args.n_atoms_per_mol, dtype=torch.long, device=device)
            batch_molecules = model.sample(n_atoms, **sample_kwargs)
        
        molecules.extend(batch_molecules)
        print(f"Batch {batch_idx+1}/{n_batches}: Generated {len(batch_molecules)} molecules (Total: {len(molecules)})")

    print(f"\nFinished generating {len(molecules)} molecules")

    # Analyze molecules if requested
    if args.analyze:
        print("\nAnalyzing molecules...")
        analyzer = SampleAnalyzer()
        analysis_results = analyzer.analyze(molecules, energy_div=True, functional_validity=True)
        print("\nFlowMol original metrics:")
        for metric, value in analysis_results.items():
            print(f"  {metric}: {value}")
    
    # Write molecules to SDF file
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    sdf_writer = Chem.SDWriter(args.output_file)
    sdf_writer.SetKekulize(False)
    valid_count = 0
    for mol in molecules:
        rdkit_mol = mol.rdkit_mol
        if rdkit_mol is not None:
            sdf_writer.write(rdkit_mol)
            valid_count += 1
    sdf_writer.close()
    print(f"\nSuccessfully wrote {valid_count}/{len(molecules)} valid molecules to {args.output_file}")

    # Compute new metrics
    print("\nComputing MultiPropGen metrics...")
    new_metrics = compute_all_standard_metrics(args.output_file, n_mols=len(molecules))
    print("\nMultiPropGen Metrics:")
    for metric, value in new_metrics.items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()