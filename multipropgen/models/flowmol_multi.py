from typing import Dict, List, Optional
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
import dgl
import torch.nn.functional as fn
from torch.distributions import Exponential
from pathlib import Path

from multipropgen.models.lr_scheduler import LRScheduler
from multipropgen.models.interpolant_scheduler import InterpolantScheduler
from multipropgen.models.vector_field import EndpointVectorField, VectorField, DirichletVectorField
from multipropgen.models.ctmc_vector_field_multi import CTMCVectorFieldMulti
from multipropgen.models.flowmol import FlowMol 

from multipropgen.data_processing.utils import build_edge_idxs, get_upper_edge_mask, get_batch_idxs
from multipropgen.data_processing.priors import uniform_simplex_prior, biased_simplex_prior, batched_rigid_alignment, rigid_alignment
from multipropgen.data_processing.priors import inference_prior_register, edge_prior
from multipropgen.analysis.molecule_builder import SampledMolecule
from multipropgen.analysis.metrics import SampleAnalyzer
from einops import rearrange
from .property_embeddings import PropertyEmbedder


class FlowMolMutli(FlowMol):
    """A normalizing flow model for molecular graphs with multi-property
     conditional generation capability."""

    def __init__(self,
                 property_names: Optional[List[str]] = None,  # NEW: List of properties to condition on
                 **kwargs):
        super().__init__(property_names=property_names, **kwargs)

        self.property_names = property_names if property_names is not None else []

        # NEW: Create a ModuleDict of property embedders, one for each property
        self.property_embedders = nn.ModuleDict()
        
        if property_names is not None and len(property_names) > 0:
            for prop_name in property_names:
                if self.gaussian_expansion:
                    # Use GaussianExpansion-based embedder
                    embedder = PropertyEmbedder(
                        input_dim=1, 
                        embedding_dim=self.property_embedding_dim,
                        start=self.gaussian_start, 
                        stop=self.gaussian_stop,
                        n_gaussians=self.n_gaussians, 
                        use_activation=True
                    )
                else:
                    # Use simple MLP embedder
                    embedder = nn.Sequential(
                        nn.Linear(1, self.property_embedding_dim),
                        nn.SiLU(),
                        nn.Linear(self.property_embedding_dim, self.property_embedding_dim),
                        nn.LayerNorm(self.property_embedding_dim)
                    )
                self.property_embedders[prop_name] = embedder
            
            print(f"Initialized property embedders for: {sorted(property_names)}")
        else:
            print("No property names provided - model will run unconditionally")

        if self.parameterization == 'ctmc':
            # NEW: Pass the ModuleDict of property embedders instead of single embedder
            self.vector_field = CTMCVectorFieldMulti(
                n_atom_types=self.n_atom_types,
                canonical_feat_order=self.canonical_feat_order,
                interpolant_scheduler=self.interpolant_scheduler, 
                n_charges=self.n_atom_charges, 
                n_bond_types=self.n_bond_types,
                exclude_charges=self.exclude_charges,
                property_embedding_dim=self.property_embedding_dim,
                property_embedders=self.property_embedders,  # Pass ModuleDict
                properties_handle_method=self.properties_handle_method,
                conditional_generation=self.conditional_generation,
                property_names=self.property_names,  # Pass property names list
                **self.vector_field_config
            )

    
    def forward(self, g: dgl.DGLGraph):
        
        batch_size = g.batch_size
        device = g.device

        # check if the attribute loss_fn_dict exists
        if not hasattr(self, 'loss_fn_dict'):
            self.configure_loss_fns(device=g.device)

        # get batch indicies of every atom and edge
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)

        # create a mask which selects all of the upper triangle edges from the batched graph
        upper_edge_mask = get_upper_edge_mask(g)

        # sample timepoints for each molecule in the batch
        t = torch.rand(batch_size, device=device).float()

        # NEW: Get multi-property embeddings if available
        if hasattr(g, 'props') and self.conditional_generation and len(self.property_names) > 0:
            # g.props should have shape [batch_size, n_properties]
            # Sort property names to ensure consistent ordering
            sorted_prop_names = sorted(self.property_names)
            
            # Print property info during first batch
            if not hasattr(self, '_printed_prop_info'):
                print(f"\nTraining with properties: {sorted_prop_names}")
                print(f"Properties shape: {g.props.shape}")
                self._printed_prop_info = True
            
            # Move embedders to device
            for prop_name in sorted_prop_names:
                self.property_embedders[prop_name] = self.property_embedders[prop_name].to(device)
            
            # Ensure properties are on device
            g.props = g.props.to(device)
            
            # Get embeddings for each property
            property_embeddings = []
            for i, prop_name in enumerate(sorted_prop_names):
                prop_val = g.props[:, i:i+1]  # [batch_size, 1]
                prop_emb = self.property_embedders[prop_name](prop_val)
                property_embeddings.append(prop_emb)
            
            # Concatenate all property embeddings
            prop_emb_concat = torch.cat(property_embeddings, dim=-1)  # [batch_size, n_props * emb_dim]
            
            # Concatenate with time
            t_combined = torch.cat([t.unsqueeze(-1), prop_emb_concat], dim=-1)
        else:
            t_combined = t
            if self.conditional_generation and len(self.property_names) > 0:
                print("WARNING: conditional_generation=True but no 'props' attribute in graph")

        # construct interpolated molecules
        g = self.vector_field.sample_conditional_path(g, t, node_batch_idx, edge_batch_idx, upper_edge_mask)

        # forward pass for the vector field
        vf_output = self.vector_field(g, t_combined, node_batch_idx=node_batch_idx, upper_edge_mask=upper_edge_mask)

        # get the target (label) for each feature
        targets = {}
        alpha_t_prime = self.interpolant_scheduler.alpha_t_prime(t)
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata

            # compute the target for endpoint parameterization
            if self.parameterization in ['endpoint', 'dirichlet', 'ctmc']:
                target = data_src[f'{feat}_1_true']
                if feat == "e":
                    target = target[upper_edge_mask]
                if feat in ['a', 'c', 'e']:
                    if self.target_blur == 0.0:
                        target = target.argmax(dim=-1)
                    else:
                        target = target + torch.randn_like(target)*self.target_blur
                        target = fn.softmax(target, dim=-1)
            #  compute the target for vector-field parameterization
            elif self.parameterization == 'vector-field':
                alpha_t_prime_i = alpha_t_prime[:, feat_idx]
                x_1 = data_src[f'{feat}_1_true']
                x_0 = data_src[f'{feat}_0']

                if feat == 'e':
                    alpha_t_prime_i = alpha_t_prime_i[edge_batch_idx][upper_edge_mask].unsqueeze(-1)
                    x_1 = x_1[upper_edge_mask]
                    x_0 = x_0[upper_edge_mask]
                else:
                    alpha_t_prime_i = alpha_t_prime_i[node_batch_idx].unsqueeze(-1)

                target = alpha_t_prime_i*(x_1 - x_0)

            # for CTMC parameterization, we do not apply loss on already unmasked features
            if self.parameterization == 'ctmc' and feat in ['a', 'c', 'e']:
                if feat == 'e':
                    xt_idxs = data_src[f'{feat}_t'][upper_edge_mask].argmax(-1)
                else:
                    xt_idxs = data_src[f'{feat}_t'].argmax(-1)
                # note that we use the default ignore_index of the CrossEntropyLoss class here
                target[ xt_idxs != self.n_cat_dict[feat] ] = -100 # set the target to ignore_index when the feature is already unmasked in xt

            targets[feat] = target

        # get the time-dependent loss weights if necessary
        if self.time_scaled_loss:
            time_weights = self.interpolant_scheduler.loss_weights(t)
            
        # compute losses
        losses = {}
        for feat_idx, feat in enumerate(self.canonical_feat_order):

            if self.time_scaled_loss:
                weight = time_weights[:, feat_idx]
                if feat == 'e':
                    weight = weight[edge_batch_idx][upper_edge_mask]
                else:
                    weight = weight[node_batch_idx]
                weight = weight.unsqueeze(-1)
            else:
                weight = 1.0

            # compute the losses
            target = targets[feat]
            losses[feat] = self.loss_fn_dict[feat](vf_output[feat], target)*weight

            # when time_scaled_loss is True, we set the reduction to 'none' so that each training example can be scaled by the time-dependent weight.
            # however, this means that we also need to do the reduction ourselves here.
            if self.time_scaled_loss:
                losses[feat] = losses[feat].mean()

        return losses

    def sample_random_sizes(self, n_molecules: int, device="cuda:0",
        stochasticity=None, high_confidence_threshold=None, 
        xt_traj=False, ep_traj=False,
        normalization_file_path: str = None, 
        conditional_generation: bool = True,
        property_names: Optional[List[str]] = None,  # NEW: list of property names
        properties_for_sampling: Optional[Dict[str, float]] = None,  # NEW: dict of property values
        training_mode: bool = True,   
        properties_handle_method: str = None,
        multilple_values_to_one_property: Optional[Dict[str, List[float]]] = None,  # NEW: dict of lists
        number_of_atoms: List[int]|None = None, 
        **kwargs):
        """Sample n_molecules with the number of atoms sampled from the distribution of the training set."""

        if multilple_values_to_one_property is not None and properties_for_sampling is not None:
            raise ValueError('You can not provide both multilple_values_to_one_property and properties_for_sampling, only one of them should be provided')
        
        # get the number of atoms that will be in each molecules
        if number_of_atoms:
            atoms_per_molecule = torch.tensor(number_of_atoms).to(device)
        else:
            atoms_per_molecule = self.sample_n_atoms(n_molecules).to(device)

        # Validate multi-property values
        if multilple_values_to_one_property is not None:
            for prop_name, values in multilple_values_to_one_property.items():
                assert len(values) == len(atoms_per_molecule), \
                    f"Length of values for {prop_name} ({len(values)}) != n_molecules ({len(atoms_per_molecule)})"

        return self.sample(
            atoms_per_molecule, 
            device=device,  
            stochasticity=stochasticity, 
            high_confidence_threshold=high_confidence_threshold,
            xt_traj=xt_traj,
            ep_traj=ep_traj,
            normalization_file_path=normalization_file_path,
            conditional_generation=conditional_generation,
            property_names=property_names,  # NEW
            properties_for_sampling=properties_for_sampling,  # NEW
            training_mode=training_mode,
            properties_handle_method=properties_handle_method,
            multilple_values_to_one_property=multilple_values_to_one_property,  # NEW
            **kwargs
        )

    @torch.no_grad()
    def sample(self, n_atoms: torch.Tensor, n_timesteps: int = None, device="cuda:0",
        stochasticity=None, high_confidence_threshold=None, xt_traj=False, ep_traj=False,
        normalization_file_path: str = None, 
        conditional_generation: bool = True,
        property_names: Optional[List[str]] = None,  # NEW
        properties_for_sampling: Optional[Dict[str, float]] = None,  # NEW
        training_mode: bool = True,   
        properties_handle_method: str = 'concatenate_sum',
        multilple_values_to_one_property: Optional[Dict[str, List[float]]] = None,  # NEW
        **kwargs):
        """
        Sample molecules with given number of atoms.
        
        Args:
            n_atoms: Tensor of shape (batch_size,) containing number of atoms per molecule
            property_names: List of property names to condition on (e.g., ['homo', 'lumo', 'gap'])
            properties_for_sampling: Dict of {property_name: value} for conditioning (same value for all molecules)
            multilple_values_to_one_property: Dict of {property_name: [values]} for batch conditioning (different values per molecule)
            
        Example usage:
            # Single value for all molecules
            molecules = model.sample(
                n_atoms=torch.tensor([10, 15, 20]),
                property_names=['homo', 'lumo', 'gap'],
                properties_for_sampling={'homo': -5.0, 'lumo': -2.0, 'gap': 3.0}
            )
            
            # Different values per molecule
            molecules = model.sample(
                n_atoms=torch.tensor([10, 15, 20]),
                property_names=['homo', 'lumo'],
                multilple_values_to_one_property={
                    'homo': [-5.0, -4.5, -6.0],
                    'lumo': [-2.0, -1.5, -3.0]
                }
            )
        """
        if n_timesteps is None:
            n_timesteps = self.default_n_timesteps

        if xt_traj or ep_traj:
            visualize = True
        else:
            visualize = False

        batch_size = n_atoms.shape[0]

        # Print property information for sampling
        if conditional_generation and property_names is not None:
            print(f"\nSampling with properties: {sorted(property_names)}")
            if properties_for_sampling is not None:
                print("Property values (same for all molecules):")
                for prop in sorted(property_names):
                    print(f"  {prop}: {properties_for_sampling[prop]}")
            elif multilple_values_to_one_property is not None:
                print("Property values (per molecule):")
                for prop in sorted(property_names):
                    values = multilple_values_to_one_property[prop]
                    print(f"  {prop}: {values[:3]}{'...' if len(values) > 3 else ''}")

        # Get edge indices for each unique number of atoms
        edge_idxs_dict = {}
        for n_atoms_i in torch.unique(n_atoms):
            edge_idxs_dict[int(n_atoms_i)] = build_edge_idxs(n_atoms_i)

        # Construct a graph for each molecule
        g = []
        for n_atoms_i in n_atoms:
            edge_idxs = edge_idxs_dict[int(n_atoms_i)]
            g_i = dgl.graph((edge_idxs[0], edge_idxs[1]), num_nodes=n_atoms_i, device=device)
            g.append(g_i)

        # Batch the graphs
        g = dgl.batch(g)

        # Get upper edge mask
        upper_edge_mask = get_upper_edge_mask(g)

        # Compute node_batch_idx
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)

        # Sample molecules from prior
        g = self.sample_prior(g, node_batch_idx, upper_edge_mask)

        # Integrate trajectories
        integrate_kwargs = {
            'upper_edge_mask': upper_edge_mask,
            'n_timesteps': n_timesteps,
            'visualize': visualize,
            'normalization_file_path': normalization_file_path,
            'conditional_generation': conditional_generation,
            'property_names': property_names,  # NEW
            'properties_for_sampling': properties_for_sampling,  # NEW
            'training_mode': training_mode,
            'properties_handle_method': properties_handle_method,
            'multilple_values_to_one_property': multilple_values_to_one_property  # NEW
        }
        if self.parameterization == 'ctmc':
            integrate_kwargs['stochasticity'] = stochasticity
            integrate_kwargs['high_confidence_threshold'] = high_confidence_threshold

        itg_result = self.vector_field.integrate(g, node_batch_idx, **integrate_kwargs, **kwargs)

        if visualize:
            g, traj_frames = itg_result
        else:
            g = itg_result

        g.edata['ue_mask'] = upper_edge_mask
        g = g.to('cpu')

        if self.parameterization == 'ctmc':
            ctmc_mol = True
        else:
            ctmc_mol = False

        molecules = []
        for mol_idx, g_i in enumerate(dgl.unbatch(g)):
            args = [g_i, self.atom_type_map]
            if visualize:
                args.append(traj_frames[mol_idx])

            molecules.append(SampledMolecule(*args, 
                ctmc_mol=ctmc_mol, 
                build_xt_traj=xt_traj,
                build_ep_traj=ep_traj,
                exclude_charges=self.exclude_charges))

        return molecules
