import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, List, Dict, Tuple
import pandas as pd
from rdkit import Chem

class PropertyToSizeDataset(Dataset):
    """Dataset for training property -> molecule size prediction"""
    
    def __init__(
        self,
        property_values: np.ndarray,  # [N, 6] for 6 properties
        n_atoms: np.ndarray,  # [N]
        property_names: List[str],
        normalize: bool = True
    ):
        """
        Args:
            property_values: [N, num_properties] array of property values
            n_atoms: [N] array of molecule sizes
            property_names: List of property names in order
            normalize: Whether to normalize properties
        """
        self.property_values = torch.FloatTensor(property_values)
        self.n_atoms = torch.LongTensor(n_atoms)
        self.property_names = property_names
        self.num_properties = len(property_names)
        
        # Compute normalization statistics
        if normalize:
            self.prop_mean = self.property_values.mean(dim=0)
            self.prop_std = self.property_values.std(dim=0)
            self.property_values = (self.property_values - self.prop_mean) / (self.prop_std + 1e-8)
        else:
            self.prop_mean = torch.zeros(self.num_properties)
            self.prop_std = torch.ones(self.num_properties)
        
        # Get unique n_atoms for creating target labels
        self.unique_n_atoms = torch.unique(self.n_atoms).sort()[0]
        self.n_atoms_to_idx = {n.item(): i for i, n in enumerate(self.unique_n_atoms)}
        self.max_n_atoms = int(self.unique_n_atoms.max())
        self.min_n_atoms = int(self.unique_n_atoms.min())
        self.num_classes = len(self.unique_n_atoms)
        
        # Convert n_atoms to class indices
        self.n_atoms_idx = torch.LongTensor([
            self.n_atoms_to_idx[n.item()] for n in self.n_atoms
        ])
        
    def __len__(self):
        return len(self.n_atoms)
    
    def __getitem__(self, idx):
        return {
            'properties': self.property_values[idx],
            'n_atoms': self.n_atoms[idx],
            'n_atoms_idx': self.n_atoms_idx[idx]
        }
    
    def get_stats(self):
        """Return normalization statistics"""
        return {
            'prop_mean': self.prop_mean,
            'prop_std': self.prop_std,
            'unique_n_atoms': self.unique_n_atoms,
            'n_atoms_to_idx': self.n_atoms_to_idx
        }


class PropertyToSizeModel(pl.LightningModule):
    """
    MLP that predicts p(n_atoms | properties)
    Supports flexible conditioning on any subset of properties
    """
    
    def __init__(
        self,
        num_properties: int = 6,
        num_classes: int = 20,  # Number of unique molecule sizes
        hidden_dims: List[int] = [256, 256, 128],
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        property_names: Optional[List[str]] = None,
        prop_mean: Optional[torch.Tensor] = None,
        prop_std: Optional[torch.Tensor] = None,
        unique_n_atoms: Optional[torch.Tensor] = None,
        n_atoms_to_idx: Optional[Dict[int, int]] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_properties = num_properties
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.property_names = property_names or [f'prop_{i}' for i in range(num_properties)]
        
        # Normalization stats
        self.register_buffer('prop_mean', prop_mean if prop_mean is not None 
                           else torch.zeros(num_properties))
        self.register_buffer('prop_std', prop_std if prop_std is not None 
                           else torch.ones(num_properties))
        self.register_buffer('unique_n_atoms', unique_n_atoms if unique_n_atoms is not None
                           else torch.arange(5, 30))
        self.n_atoms_to_idx = n_atoms_to_idx or {}
        
        # Build MLP
        # Input: properties + mask (indicating which properties are active)
        input_dim = num_properties * 2  # properties + mask
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(
        self, 
        properties: torch.Tensor,  # [batch, num_properties]
        mask: Optional[torch.Tensor] = None  # [batch, num_properties]
    ):
        """
        Args:
            properties: [batch, num_properties] property values
            mask: [batch, num_properties] binary mask (1 = use this property, 0 = ignore)
                  If None, use all properties
        
        Returns:
            logits: [batch, num_classes] unnormalized log probabilities
        """
        batch_size = properties.shape[0]
        
        if mask is None:
            mask = torch.ones_like(properties)
        
        # Apply mask to properties (masked properties become 0)
        masked_properties = properties * mask
        
        # Concatenate properties and mask as input
        x = torch.cat([masked_properties, mask], dim=-1)
        
        logits = self.network(x)
        return logits
    
    def training_step(self, batch, batch_idx):
        properties = batch['properties']
        targets = batch['n_atoms_idx']
        
        # Random masking during training (dropout on properties)
        # This helps the model learn to condition on any subset
        if self.training:
            mask = torch.bernoulli(torch.ones_like(properties) * 0.8)  # 20% dropout
        else:
            mask = torch.ones_like(properties)
        
        logits = self(properties, mask)
        loss = F.cross_entropy(logits, targets)
        
        # Accuracy
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == targets).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        properties = batch['properties']
        targets = batch['n_atoms_idx']
        
        # Use all properties during validation
        logits = self(properties)
        loss = F.cross_entropy(logits, targets)
        
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == targets).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
    
    def normalize_properties(self, properties: np.ndarray) -> torch.Tensor:
        """Normalize property values using training statistics"""
        properties = torch.FloatTensor(properties)
        return (properties - self.prop_mean) / (self.prop_std + 1e-8)
    
    def sample_n_atoms(
        self,
        property_dict: Dict[str, float],
        n_samples: int = 1000,
        return_distribution: bool = False,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Sample molecule sizes conditioned on specified properties.
        
        Args:
            property_dict: Dict mapping property names to values
                          e.g., {'homo': -6.5, 'lumo': -2.3}
            n_samples: Number of samples to draw
            return_distribution: If True, return full probability distribution
            temperature: Temperature for sampling (higher = more diverse)
        
        Returns:
            If return_distribution=False: [n_samples] array of sampled n_atoms
            If return_distribution=True: tuple of (n_atoms_values, probabilities)
        """
        self.eval()
        with torch.no_grad():
            # Create property tensor and mask
            properties = torch.zeros((1, self.num_properties))
            mask = torch.zeros((1, self.num_properties))
            
            for prop_name, value in property_dict.items():
                if prop_name in self.property_names:
                    idx = self.property_names.index(prop_name)
                    properties[0, idx] = value
                    mask[0, idx] = 1.0
                else:
                    raise ValueError(f"Unknown property: {prop_name}")
            
            # Normalize properties
            properties = properties.to(self.device)  # Move to CUDA first
            mask = mask.to(self.device)
            properties = (properties - self.prop_mean) / (self.prop_std + 1e-8)  # CUDA - CUDA = OK!
            
            # Get logits and apply temperature
            logits = self(properties, mask)
            probs = F.softmax(logits / temperature, dim=-1).cpu().squeeze()
            
            if return_distribution:
                return self.unique_n_atoms.numpy(), probs.numpy()
            
            # Sample from categorical distribution
            dist = torch.distributions.Categorical(probs)
            samples_idx = dist.sample((n_samples,))
            
            # Convert indices back to n_atoms values
            sampled_n_atoms = self.unique_n_atoms.cpu()[samples_idx].numpy()
            
            return sampled_n_atoms


# Training script
def train_property_to_size_model(
    qm9_csv_path: str = "../data/qm9_raw/gdb9.sdf.csv",
    sdf_path: str = "../data/qm9_raw/all_fixed_gdb9.sdf",
    train_idx_path: str = "../data/qm9_raw/train_mol_idxs.npy",
    property_names: List[str] = ['alpha', 'gap', 'homo', 'lumo', 'mu', 'cv'],
    batch_size: int = 256,
    max_epochs: int = 100,
    num_workers: int = 4,
    **model_kwargs
):
    """Train the property to size prediction model"""
    
    # Load data
    print("Loading QM9 data...")
    train_ind = np.load(train_idx_path)
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    n_atoms = np.array([mol.GetNumAtoms() for mol in suppl])
    
    df = pd.read_csv(qm9_csv_path)
    
    # Get property values
    property_values = df[property_names].values  # [N, 6]
    
    # Split train/val
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_ind] = True
    val_mask = ~train_mask
    
    # Create datasets
    train_dataset = PropertyToSizeDataset(
        property_values[train_mask],
        n_atoms[train_mask],
        property_names,
        normalize=True
    )
    
    stats = train_dataset.get_stats()
    
    val_dataset = PropertyToSizeDataset(
        property_values[val_mask],
        n_atoms[val_mask],
        property_names,
        normalize=True
    )
    val_dataset.prop_mean = stats['prop_mean']
    val_dataset.prop_std = stats['prop_std']
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create model
    model = PropertyToSizeModel(
        num_properties=len(property_names),
        num_classes=train_dataset.num_classes,
        property_names=property_names,
        prop_mean=stats['prop_mean'],
        prop_std=stats['prop_std'],
        unique_n_atoms=stats['unique_n_atoms'],
        n_atoms_to_idx=stats['n_atoms_to_idx'],
        **model_kwargs
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        devices=1,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=3,
                filename='prop2size-{epoch:02d}-{val_loss:.4f}'
            ),
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                mode='min'
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        ],
        logger=pl.loggers.TensorBoardLogger('lightning_logs', name='prop2size'),
        gradient_clip_val=1.0
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    return model, trainer


# Sampling script
def sample_with_trained_model(
    checkpoint_path: str,
    property_conditions: Dict[str, float],
    n_samples: int = 1000,
    temperature: float = 1.0
):
    """
    Load trained model and sample n_atoms.
    
    Example usage:
        # Sample conditioned on only HOMO and LUMO
        samples = sample_with_trained_model(
            'checkpoints/best.ckpt',
            {'homo': -6.5, 'lumo': -2.3},
            n_samples=1000
        )
        
        # Sample conditioned on all 6 properties
        samples = sample_with_trained_model(
            'checkpoints/best.ckpt',
            {'alpha': 75.0, 'gap': 8.5, 'homo': -6.5, 
             'lumo': -2.3, 'mu': 2.5, 'cv': 30.0},
            n_samples=1000
        )
    """
    # Load model
    model = PropertyToSizeModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Sample
    sampled_n_atoms = model.sample_n_atoms(
        property_conditions,
        n_samples=n_samples,
        temperature=temperature
    )
    
    print(f"\nSampling with conditions: {property_conditions}")
    print(f"Sampled {n_samples} molecule sizes")
    print(f"Distribution:")
    unique, counts = np.unique(sampled_n_atoms, return_counts=True)
    for n, c in zip(unique, counts):
        print(f"  {n} atoms: {c} samples ({c/n_samples*100:.1f}%)")
    
    return sampled_n_atoms


# Example usage script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'sample'], required=True)
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint for sampling')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--n_samples', type=int, default=1000)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Training model...")
        model, trainer = train_property_to_size_model(
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            hidden_dims=[256, 256, 128],
            dropout=0.1,
            lr=1e-3
        )
        print("Training complete!")
        
    elif args.mode == 'sample':
        # Example: Sample conditioned on HOMO and LUMO only
        samples = sample_with_trained_model(
            args.checkpoint,
            property_conditions={'homo': -6.5, 'lumo': -2.3},
            n_samples=args.n_samples,
            temperature=1.0
        )
        
        # Save samples
        np.save('sampled_n_atoms.npy', samples)