import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from typing import List, Dict, Tuple, Optional
from property_to_size_model import PropertyToSizeModel

class MultiPropertySampler:
    """
    Sample from joint distribution of multiple properties.
    Uses multi-dimensional histograms to capture correlations between properties.
    """
    
    def __init__(
        self,
        property_dict: Dict[str, np.ndarray],
        n_atoms: np.ndarray,
        num_bins: int = 50,
        seed: Optional[int] = None
    ):
        """
        Args:
            property_dict: Dict mapping property names to values
                          e.g., {'homo': [N], 'lumo': [N], 'gap': [N]}
            n_atoms: [N] array of molecule sizes
            num_bins: Number of bins per dimension for discretizing
            seed: Random seed
        """
        self.property_names = sorted(property_dict.keys())  # Keep consistent order
        self.num_properties = len(self.property_names)
        self.num_bins = num_bins
        self.n_atoms = n_atoms
        
        if seed is not None:
            np.random.seed(seed)
        
        # Stack property values into [N, num_properties] array
        self.property_values = np.stack([
            property_dict[name] for name in self.property_names
        ], axis=1)
        
        # Store property ranges for each dimension
        self.prop_mins = self.property_values.min(axis=0)
        self.prop_maxs = self.property_values.max(axis=0)
        self.prop_ranges = self.prop_maxs - self.prop_mins
        
        # Create joint histogram using np.histogramdd
        # This captures correlations between properties
        self._create_joint_distribution()
        
        # Also store node distribution for each joint bin
        self._create_conditional_node_distributions()
    
    def _create_joint_distribution(self):
        """Create multi-dimensional histogram of property joint distribution"""
        
        # Define bin edges for each property dimension
        bin_edges = [
            np.linspace(self.prop_mins[i], self.prop_maxs[i], self.num_bins + 1)
            for i in range(self.num_properties)
        ]
        
        # Create joint histogram
        self.histogram, self.bin_edges = np.histogramdd(
            self.property_values,
            bins=bin_edges
        )
        
        # Convert to probability distribution
        total = self.histogram.sum()
        if total > 0:
            self.prob_dist = self.histogram / total
        else:
            raise ValueError("Empty histogram - check your data")
        
        # Create flat indices for sampling
        self.flat_prob = self.prob_dist.flatten()
        self.flat_prob = self.flat_prob / self.flat_prob.sum()  # Renormalize
        
    def _create_conditional_node_distributions(self):
        """
        For each bin in the joint histogram, store the distribution of n_atoms.
        This allows us to sample n_atoms conditioned on property values.
        """
        # Digitize property values to find which bin each molecule belongs to
        bin_indices = []
        for i, prop_name in enumerate(self.property_names):
            # np.digitize returns 1-indexed bins, we want 0-indexed
            bins = np.digitize(
                self.property_values[:, i], 
                self.bin_edges[i][1:-1]  # Exclude first and last edge
            )
            bin_indices.append(bins)
        
        bin_indices = np.stack(bin_indices, axis=1)  # [N, num_properties]
        
        # For each unique bin combination, store n_atoms distribution
        self.bin_to_n_atoms = {}
        
        for idx in range(len(self.n_atoms)):
            bin_tuple = tuple(bin_indices[idx])
            
            # Check if bin is valid (within bounds)
            valid = all(0 <= b < self.num_bins for b in bin_tuple)
            if not valid:
                continue
                
            if bin_tuple not in self.bin_to_n_atoms:
                self.bin_to_n_atoms[bin_tuple] = []
            
            self.bin_to_n_atoms[bin_tuple].append(self.n_atoms[idx])
        
        # Convert lists to arrays
        for bin_tuple in self.bin_to_n_atoms:
            self.bin_to_n_atoms[bin_tuple] = np.array(self.bin_to_n_atoms[bin_tuple])
    
    def _bin_indices_to_values(self, bin_indices: np.ndarray) -> np.ndarray:
        """
        Convert bin indices to continuous property values.
        Samples uniformly within each bin.
        
        Args:
            bin_indices: [n_samples, num_properties] array of bin indices
        
        Returns:
            [n_samples, num_properties] array of property values
        """
        n_samples = bin_indices.shape[0]
        property_values = np.zeros((n_samples, self.num_properties))
        
        for i in range(self.num_properties):
            for j in range(n_samples):
                bin_idx = bin_indices[j, i]
                
                # Get bin edges
                left = self.bin_edges[i][bin_idx]
                right = self.bin_edges[i][bin_idx + 1]
                
                # Sample uniformly within bin
                property_values[j, i] = np.random.uniform(left, right)
        
        return property_values
    
    def sample(
        self, 
        n_samples: int,
        return_n_atoms: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Sample from joint property distribution.
        
        Args:
            n_samples: Number of samples to generate
            return_n_atoms: If True, also sample n_atoms for each property sample
        
        Returns:
            property_samples: [n_samples, num_properties] array of property values
            n_atoms_samples: [n_samples] array of molecule sizes (if return_n_atoms=True)
        """
        # Sample flat bin indices from the joint distribution
        flat_indices = np.random.choice(
            len(self.flat_prob),
            size=n_samples,
            p=self.flat_prob
        )
        
        # Convert flat indices to multi-dimensional indices
        bin_indices = np.array(np.unravel_index(flat_indices, self.histogram.shape)).T
        
        # Convert bin indices to continuous values
        property_samples = self._bin_indices_to_values(bin_indices)
        
        if not return_n_atoms:
            return property_samples, None
        
        # Sample n_atoms for each property combination
        n_atoms_samples = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            bin_tuple = tuple(bin_indices[i])
            
            if bin_tuple in self.bin_to_n_atoms and len(self.bin_to_n_atoms[bin_tuple]) > 0:
                # Sample from observed n_atoms in this bin
                n_atoms_samples[i] = np.random.choice(self.bin_to_n_atoms[bin_tuple])
            else:
                # Fallback: sample from overall n_atoms distribution
                n_atoms_samples[i] = np.random.choice(self.n_atoms)
        
        return property_samples, n_atoms_samples
    
    def get_property_dict(self) -> List[str]:
        """Return property names in the order they appear in sampled arrays"""
        return self.property_names


class MultiPropertySamplerWithMLP:
    """
    Combines multi-property sampling with MLP-based n_atoms prediction.
    This is useful when you want more flexible n_atoms prediction.
    """
    
    def __init__(
        self,
        property_dict: Dict[str, np.ndarray],
        mlp_checkpoint: Optional[str] = None,
        num_bins: int = 50,
        seed: Optional[int] = None
    ):
        """
        Args:
            property_dict: Dict mapping property names to values
            mlp_checkpoint: Path to trained PropertyToSizeModel checkpoint
            num_bins: Number of bins for joint histogram
            seed: Random seed
        """
        self.property_names = sorted(property_dict.keys())
        self.num_bins = num_bins
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Stack property values
        self.property_values = np.stack([
            property_dict[name] for name in self.property_names
        ], axis=1)
        
        # Create joint distribution (without n_atoms conditioning)
        self.prop_mins = self.property_values.min(axis=0)
        self.prop_maxs = self.property_values.max(axis=0)
        
        bin_edges = [
            np.linspace(self.prop_mins[i], self.prop_maxs[i], self.num_bins + 1)
            for i in range(len(self.property_names))
        ]
        
        self.histogram, self.bin_edges = np.histogramdd(
            self.property_values,
            bins=bin_edges
        )
        
        total = self.histogram.sum()
        self.prob_dist = self.histogram / total if total > 0 else self.histogram
        self.flat_prob = self.prob_dist.flatten()
        self.flat_prob = self.flat_prob / self.flat_prob.sum()
        
        # Load MLP model if provided
        self.mlp_model = None
        if mlp_checkpoint is not None:
            self.mlp_model = PropertyToSizeModel.load_from_checkpoint(mlp_checkpoint)
            self.mlp_model.eval()
    
    def _bin_indices_to_values(self, bin_indices: np.ndarray) -> np.ndarray:
        """Convert bin indices to continuous property values"""
        n_samples = bin_indices.shape[0]
        property_values = np.zeros((n_samples, len(self.property_names)))
        
        for i in range(len(self.property_names)):
            for j in range(n_samples):
                bin_idx = bin_indices[j, i]
                left = self.bin_edges[i][bin_idx]
                right = self.bin_edges[i][bin_idx + 1]
                property_values[j, i] = np.random.uniform(left, right)
        
        return property_values
    
    def sample(
        self,
        n_samples: int,
        use_mlp_for_n_atoms: bool = True,
        temperature: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample property values and corresponding n_atoms.
        
        Args:
            n_samples: Number of samples
            use_mlp_for_n_atoms: If True, use MLP to predict n_atoms
            temperature: Temperature for MLP sampling
        
        Returns:
            property_samples: [n_samples, num_properties]
            n_atoms_samples: [n_samples]
        """
        # Sample property values from joint distribution
        flat_indices = np.random.choice(
            len(self.flat_prob),
            size=n_samples,
            p=self.flat_prob
        )
        
        bin_indices = np.array(np.unravel_index(flat_indices, self.histogram.shape)).T
        property_samples = self._bin_indices_to_values(bin_indices)
        
        # Use MLP to sample n_atoms for each property combination
        if use_mlp_for_n_atoms and self.mlp_model is not None:
            n_atoms_samples = np.zeros(n_samples, dtype=int)
            
            for i in range(n_samples):
                # Create property dict for this sample
                prop_dict = {
                    name: property_samples[i, j]
                    for j, name in enumerate(self.property_names)
                }
                
                # Sample n_atoms using MLP
                n_atoms = self.mlp_model.sample_n_atoms(
                    prop_dict,
                    n_samples=1,
                    temperature=temperature
                )
                n_atoms_samples[i] = n_atoms[0]
        else:
            # Without MLP, return None or raise error
            raise ValueError("MLP model required for n_atoms sampling. Provide mlp_checkpoint.")
        
        return property_samples, n_atoms_samples


def create_multi_property_samples_for_training(
    qm9_csv_path: str = "../data/qm9_raw/gdb9.sdf.csv",
    sdf_path: str = "../data/qm9_raw/all_fixed_gdb9.sdf",
    train_idx_path: str = "../data/qm9_raw/train_mol_idxs.npy",
    property_names: List[str] = ['homo', 'lumo'],  # Specify which properties
    n_samples: int = 10000,
    num_bins: int = 1000,
    output_dir: str = "multi_property_sampling",
    mlp_checkpoint: Optional[str] = None,
    seed: int = 42
):
    """
    Create sampling files for multi-property conditioning.
    
    Args:
        property_names: List of 2-6 properties to jointly sample
        n_samples: Number of samples to generate
        mlp_checkpoint: Optional path to trained MLP for n_atoms prediction
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading QM9 data...")
    train_ind = np.load(train_idx_path)
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    n_atoms = np.array([mol.GetNumAtoms() for mol in suppl])
    
    df = pd.read_csv(qm9_csv_path)
    
    # Get training data
    train_df = df.iloc[train_ind]
    train_n_atoms = n_atoms[train_ind]
    
    # Create property dict for training data
    property_dict = {
        prop: train_df[prop].values for prop in property_names
    }
    
    print(f"\nSampling from joint distribution of: {property_names}")
    print(f"Number of properties: {len(property_names)}")
    print(f"Number of training samples: {len(train_ind)}")
    
    # Choose sampling method
    if mlp_checkpoint is not None:
        print(f"Using MLP model from: {mlp_checkpoint}")
        sampler = MultiPropertySamplerWithMLP(
            property_dict=property_dict,
            mlp_checkpoint=mlp_checkpoint,
            num_bins=num_bins,
            seed=seed
        )
        prop_samples, n_atoms_samples = sampler.sample(
            n_samples=n_samples,
            use_mlp_for_n_atoms=True,
            temperature=1.0
        )
    else:
        print("Using histogram-based sampling for n_atoms")
        sampler = MultiPropertySampler(
            property_dict=property_dict,
            n_atoms=train_n_atoms,
            num_bins=num_bins,
            seed=seed
        )
        prop_samples, n_atoms_samples = sampler.sample(
            n_samples=n_samples,
            return_n_atoms=True
        )
    
    # Save results
    props_str = '_'.join(property_names)
    
    # Save property samples as dict
    save_dict = {
        prop: prop_samples[:, i] 
        for i, prop in enumerate(property_names)
    }
    save_dict['n_atoms'] = n_atoms_samples
    
    # Save as individual files
    for prop in property_names:
        np.save(
            f"{output_dir}/sampled_{prop}_from_prop_{props_str}.npy",
            save_dict[prop]
        )
    
    np.save(f"{output_dir}/sampled_n_atoms_{props_str}.npy", n_atoms_samples)
    
    # Also save as combined file for convenience
    np.savez(
        f"{output_dir}/multi_property_samples_{props_str}.npz",
        **save_dict,
        property_names=property_names
    )
    
    # Print statistics
    print(f"\nSaved {n_samples} samples to {output_dir}/")
    print("\nProperty value ranges:")
    for i, prop in enumerate(property_names):
        print(f"  {prop}: [{prop_samples[:, i].min():.3f}, {prop_samples[:, i].max():.3f}]")
    
    print(f"\nn_atoms distribution:")
    unique, counts = np.unique(n_atoms_samples, return_counts=True)
    for n, c in zip(unique, counts):
        print(f"  {n} atoms: {c} ({c/n_samples*100:.1f}%)")
    
    return prop_samples, n_atoms_samples


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--properties', nargs='+', default=['homo', 'lumo'],
                       help='List of properties to jointly sample')
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--num_bins', type=int, default=1000)
    parser.add_argument('--mlp_checkpoint', type=str, default=None,
                       help='Path to trained MLP checkpoint')
    parser.add_argument('--output_dir', type=str, default='multi_property_sampling')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    prop_samples, n_atoms_samples = create_multi_property_samples_for_training(
        property_names=args.properties,
        n_samples=args.n_samples,
        num_bins=args.num_bins,
        output_dir=args.output_dir,
        mlp_checkpoint=args.mlp_checkpoint,
        seed=args.seed
    )
    
    print("\nDone!")