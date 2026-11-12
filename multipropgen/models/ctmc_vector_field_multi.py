from typing import List, Dict, Optional
import torch
import torch.nn as nn
import dgl
from multipropgen.models.ctmc_vector_field import CTMCVectorField, PROPERTY_MAP
from torch.nn.functional import one_hot
from torch.distributions.categorical import Categorical
from multipropgen.data_processing.utils import get_edge_batch_idxs
import torch.nn.functional as F

from multipropgen.utils.ctmc_utils import purity_sampling
from typing import Union, Callable


class CTMCVectorFieldMulti(CTMCVectorField):
    def __init__(self, *args, 
                 property_names: Optional[List[str]]=None,  # NEW: list of property names
                 property_embedders: Optional[nn.ModuleDict]=None,
                 **kwargs):
        super().__init__(*args, property_names=property_names, 
                            property_embedders=property_embedders, **kwargs)

        self.property_names = property_names if property_names is not None else []
        self.property_embedders = property_embedders if property_embedders is not None else nn.ModuleDict()

        # For multi-property, the concatenated embedding dimension will be
        # property_embedding_dim * num_properties
        n_properties = len(self.property_names) if self.property_names else 1
        total_prop_dim = self.property_embedding_dim * n_properties
        
        self.scalar_embedding_cond = nn.Sequential(
            nn.Linear(self.n_hidden_scalars + total_prop_dim, self.n_hidden_scalars),
            nn.SiLU(),
            nn.Linear(self.n_hidden_scalars, self.n_hidden_scalars),
        )
        
        # Initialize instance variables for sampling (will be set by integrate())
        self._sampling_property_names = None
        self._sampling_properties_for_sampling = None
        self._sampling_multilple_values_to_one_property = None
        self._sampling_normalization_file_path = None

    def integrate(self, g: dgl.DGLGraph, node_batch_idx: torch.Tensor, 
        upper_edge_mask: torch.Tensor, n_timesteps: int, 
        visualize=False, 
        dfm_type='campbell',
        stochasticity=8.0, 
        high_confidence_threshold=0.9,
        cat_temp_func=None,
        forward_weight_func=None,
        tspan=None,
        normalization_file_path: str = None,
        conditional_generation: bool = True,
        property_names: Optional[List[str]] = None,  # NEW: multi-property support
        properties_for_sampling: Optional[Dict[str, float]] = None,  # NEW: dict for multi-property
        training_mode: bool = True,
        properties_handle_method: str = None,
        multilple_values_to_one_property: Optional[Dict[str, List[float]]] = None,  # NEW: dict for multi-property
        **kwargs):
        """Integrate the trajectories of molecules along the vector field.
        
        Overrides parent to properly store multi-property parameters.
        """
        
        # Store multi-property sampling parameters as instance variables
        # These will be accessed by forward() during sampling
        self._sampling_property_names = property_names
        self._sampling_properties_for_sampling = properties_for_sampling
        self._sampling_multilple_values_to_one_property = multilple_values_to_one_property
        self._sampling_normalization_file_path = normalization_file_path
        
        # Also store for parent compatibility
        self.properties_for_sampling = properties_for_sampling
        self.property_name = None  # Not used in multi-property
        self.conditional_generation = conditional_generation
        self.normalization_file_path = normalization_file_path
        self.training_mode = training_mode
        self.properties_handle_method = properties_handle_method
        self.multilple_values_to_one_property = multilple_values_to_one_property

        if cat_temp_func is None:
            cat_temp_func = self.cat_temp_func
        if forward_weight_func is None:
            forward_weight_func = self.forward_weight_func

        # get edge_batch_idx
        edge_batch_idx = get_edge_batch_idxs(g)

        # get the timepoint for integration
        if tspan is None:
            t = torch.linspace(0, 1, n_timesteps, device=g.device)
        else:
            t = tspan

        # get the corresponding alpha values for each timepoint
        alpha_t = self.interpolant_scheduler.alpha_t(t)
        alpha_t_prime = self.interpolant_scheduler.alpha_t_prime(t)

        # set x_t = x_0
        for feat in self.canonical_feat_order:
            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata
            data_src[f'{feat}_t'] = data_src[f'{feat}_0']

        # if visualizing the trajectory, create a datastructure to store the trajectory
        if visualize:
            traj_frames = {}
            for feat in self.canonical_feat_order:
                if feat == "e":
                    data_src = g.edata
                    split_sizes = g.batch_num_edges()
                else:
                    data_src = g.ndata
                    split_sizes = g.batch_num_nodes()

                split_sizes = split_sizes.detach().cpu().tolist()
                init_frame = data_src[f'{feat}_0'].detach().cpu()
                init_frame = torch.split(init_frame, split_sizes)
                traj_frames[feat] = [init_frame]
                traj_frames[f'{feat}_1_pred'] = []
    
        for s_idx in range(1, t.shape[0]):
            s_i = t[s_idx]
            t_i = t[s_idx - 1]
            alpha_t_i = alpha_t[s_idx - 1]
            alpha_s_i = alpha_t[s_idx]
            alpha_t_prime_i = alpha_t_prime[s_idx - 1]

            last_step = (s_idx == t.shape[0] - 1)

            g = self.step(g, s_i, t_i, alpha_t_i, alpha_s_i, 
                alpha_t_prime_i, 
                node_batch_idx, 
                edge_batch_idx, 
                upper_edge_mask, 
                cat_temp_func=cat_temp_func,
                forward_weight_func=forward_weight_func,
                dfm_type=dfm_type,
                stochasticity=stochasticity, 
                high_confidence_threshold=high_confidence_threshold,
                last_step=last_step,
                normalization_file_path=normalization_file_path,
                conditional_generation=conditional_generation,
                property_names=property_names,  # Pass multi-property names
                properties_for_sampling=properties_for_sampling,  # Pass dict
                multilple_values_to_one_property=multilple_values_to_one_property,  # Pass dict
                training_mode=training_mode,
                **kwargs)

            if visualize:
                for feat in self.canonical_feat_order:
                    if feat == "e":
                        g_data_src = g.edata
                    else:
                        g_data_src = g.ndata

                    if feat == 'e':
                        split_sizes = g.batch_num_edges()
                    else:
                        split_sizes = g.batch_num_nodes()
                    split_sizes = split_sizes.detach().cpu().tolist()
                    frame = g_data_src[f'{feat}_t'].detach().cpu()
                    frame = torch.split(frame, split_sizes)
                    traj_frames[feat].append(frame)

                    ep_frame = g_data_src[f'{feat}_1_pred'].detach().cpu()
                    ep_frame = torch.split(ep_frame, split_sizes)
                    traj_frames[f'{feat}_1_pred'].append(ep_frame)

        # set x_1 = x_t
        for feat in self.canonical_feat_order:
            if feat == "e":
                g_data_src = g.edata
            else:
                g_data_src = g.ndata
            g_data_src[f'{feat}_1'] = g_data_src[f'{feat}_t']

        if visualize:
            reshaped_traj_frames = []
            for mol_idx in range(g.batch_size):
                molecule_dict = {}
                for feat in traj_frames.keys():
                    feat_traj = []
                    n_frames = len(traj_frames[feat])
                    for frame_idx in range(n_frames):
                        feat_traj.append(traj_frames[feat][frame_idx][mol_idx])
                    molecule_dict[feat] = torch.stack(feat_traj)
                reshaped_traj_frames.append(molecule_dict)
            return g, reshaped_traj_frames
        
        return g

    def forward(self, g: dgl.DGLGraph, t: torch.Tensor, 
                node_batch_idx: torch.Tensor,
                upper_edge_mask: torch.Tensor,
                apply_softmax: bool = False,
                remove_com: bool = True,
                normalization_file_path: str = None,
                property_names: Optional[List[str]] = None,
                properties_for_sampling: Optional[Dict[str, float]] = None,
                multilple_values_to_one_property: Optional[Dict[str, List[float]]] = None,
                **kwargs):
        
        device = g.device
        
        # FIX: Use instance variables as fallback for sampling parameters
        # This ensures compatibility with the parent's integrate() -> step() -> forward() flow
        if property_names is None:
            property_names = self._sampling_property_names
        if properties_for_sampling is None:
            properties_for_sampling = self._sampling_properties_for_sampling
        if multilple_values_to_one_property is None:
            multilple_values_to_one_property = self._sampling_multilple_values_to_one_property
        if normalization_file_path is None:
            normalization_file_path = self._sampling_normalization_file_path
        
        with g.local_scope():
            # Determine if this is conditional generation
            is_conditional = self.conditional_generation and (
                len(t.shape) > 1 or  # Training with properties in t
                properties_for_sampling is not None or  # Sampling with single values
                multilple_values_to_one_property is not None  # Sampling with multiple values
            )

            # Prepare time tensor
            if len(t.shape) > 1:
                time_tensor = t[:, 0].unsqueeze(-1)
            else:
                time_tensor = t.unsqueeze(-1)

            # Construct base scalar features (unconditional embedding)
            base_features = [
                g.ndata['a_t'],
                time_tensor[node_batch_idx]
            ]
            
            if not self.exclude_charges:
                base_features.append(g.ndata['c_t'])
            base_features = torch.cat(base_features, dim=-1)
            node_scalar_features = self.scalar_embedding_uncond(base_features)

            # Handle conditional generation with multiple properties
            if is_conditional:
                try:
                    prop_emb = None
                    
                    # Case 1: Property embeddings already in t (from FlowMol.forward during training)
                    if len(t.shape) > 1 and t.shape[1] > 1:
                        # t has shape [batch_size, 1 + n_properties * embedding_dim]
                        # Property embeddings are ALREADY embedded by FlowMol, just extract them
                        prop_emb = t[:, 1:]  # [batch_size, n_props * emb_dim]
                        prop_emb = prop_emb[node_batch_idx]  # Repeat for each node
                    
                    # Case 2: Need to embed properties (for sampling)
                    elif properties_for_sampling is not None or multilple_values_to_one_property is not None:
                        # Load normalization parameters if needed
                        norm_params = None
                        if normalization_file_path:
                            norm_params = torch.load(normalization_file_path)
                        
                        # Determine which property names to use (in sorted order)
                        if property_names is None:
                            property_names = sorted(self.property_embedders.keys())
                        else:
                            property_names = sorted(property_names)
                        
                        property_embeddings = []
                        for prop_name in property_names:
                            if prop_name not in self.property_embedders:
                                raise ValueError(f"Property '{prop_name}' not found in property_embedders. "
                                               f"Available: {list(self.property_embedders.keys())}")
                            
                            # Get the property value(s)
                            if multilple_values_to_one_property is not None:
                                if prop_name not in multilple_values_to_one_property:
                                    raise ValueError(f"Property {prop_name} not in multilple_values_to_one_property. "
                                                   f"Available: {list(multilple_values_to_one_property.keys())}")
                                prop_values = multilple_values_to_one_property[prop_name]
                                if len(prop_values) != g.batch_size:
                                    raise ValueError(f"Number of values for {prop_name} ({len(prop_values)}) != batch_size ({g.batch_size})")
                                
                                # Normalize if needed
                                if norm_params is not None:
                                    property_idx = int(PROPERTY_MAP.get(prop_name))
                                    mean = norm_params['mean'][property_idx].item()
                                    std = norm_params['std'][property_idx].item()
                                    prop_values = [(val - mean) / std for val in prop_values]
                                
                                prop_batch = torch.tensor(prop_values, device=device, dtype=torch.float32).view(g.batch_size, 1)
                            
                            elif properties_for_sampling is not None:
                                if prop_name not in properties_for_sampling:
                                    raise ValueError(f"Property {prop_name} not in properties_for_sampling. "
                                                   f"Available: {list(properties_for_sampling.keys())}")
                                prop_value = properties_for_sampling[prop_name]
                                
                                # Normalize if needed
                                if norm_params is not None:
                                    property_idx = int(PROPERTY_MAP.get(prop_name))
                                    mean = norm_params['mean'][property_idx].item()
                                    std = norm_params['std'][property_idx].item()
                                    prop_value = (prop_value - mean) / std
                                
                                prop_batch = torch.full((g.batch_size, 1), prop_value, device=device, dtype=torch.float32)
                            
                            # Get embedding for this property
                            self.property_embedders[prop_name] = self.property_embedders[prop_name].to(device)
                            prop_emb_i = self.property_embedders[prop_name](prop_batch)
                            property_embeddings.append(prop_emb_i)
                        
                        # Concatenate all property embeddings
                        if len(property_embeddings) > 0:
                            prop_emb = torch.cat(property_embeddings, dim=-1)  # [batch_size, n_props * emb_dim]
                            prop_emb = prop_emb[node_batch_idx]  # Repeat for each node
                        else:
                            raise ValueError("No property embeddings generated")
                    
                    if prop_emb is None:
                        raise ValueError("No property information available for conditional generation")

                    # Handle properties with different methods
                    if self.properties_handle_method == 'concatenate_sum':
                        intermediate_features = torch.cat([node_scalar_features, prop_emb], dim=-1)
                        intermediate_features = self.scalar_embedding_cond(intermediate_features)
                        node_scalar_features = node_scalar_features + intermediate_features
                    elif self.properties_handle_method == 'concatenate':
                        intermediate_features = torch.cat([node_scalar_features, prop_emb], dim=-1)
                        node_scalar_features = self.scalar_embedding_cond(intermediate_features)
                    elif self.properties_handle_method == 'sum':
                        node_scalar_features = node_scalar_features + prop_emb
                    elif self.properties_handle_method == 'multiply':
                        prop_emb = torch.sigmoid(prop_emb) + 0.5
                        node_scalar_features = node_scalar_features * prop_emb
                    elif self.properties_handle_method == 'concatenate_multiply':
                        intermediate_features = torch.cat([node_scalar_features, prop_emb], dim=-1)
                        intermediate_features = self.scalar_embedding_cond(intermediate_features)
                        intermediate_features = torch.sigmoid(prop_emb) + 0.5
                        node_scalar_features = node_scalar_features * intermediate_features
                    else:
                        raise ValueError(f"Invalid properties_handle_method: {self.properties_handle_method}")

                except Exception as e:
                    print(f"Debug info: is_conditional={is_conditional}, "
                        f"t.shape={t.shape}, "
                        f"property_names={property_names}, "
                        f"properties_for_sampling={properties_for_sampling is not None}, "
                        f"multilple_values_to_one_property={multilple_values_to_one_property is not None}"
                        )
                    raise e

            # Rest of forward remains same as EndpointVectorField
            node_positions = g.ndata['x_t']
            edge_features = g.edata['e_t']
            edge_features = self.edge_embedding(edge_features)

            # update atom features, bond features and positions (GVP or transformer backbone)
            node_scalar_features, edge_features, node_positions = self._run_backbone(
                g, node_scalar_features, edge_features, node_positions, time_tensor[:, 0], node_batch_idx)

            # predict final charges and atom type logits
            node_scalar_features = self.node_output_head(node_scalar_features)
            atom_type_logits = node_scalar_features[:, :self.n_atom_types]
            if not self.exclude_charges:
                atom_charge_logits = node_scalar_features[:, self.n_atom_types:]

            # predict the final edge logits
            ue_feats = edge_features[upper_edge_mask]
            le_feats = edge_features[~upper_edge_mask]
            edge_logits = self.to_edge_logits(ue_feats + le_feats)

            # project node positions back into zero-COM subspace
            if remove_com:
                g.ndata['x_1_pred'] = node_positions
                g.ndata['x_1_pred'] = g.ndata['x_1_pred'] - dgl.readout_nodes(g, feat='x_1_pred', op='mean')[node_batch_idx]
                node_positions = g.ndata['x_1_pred']

        # build a dictionary of predicted features
        dst_dict = {
            'x': node_positions,
            'a': atom_type_logits,
            'e': edge_logits
        }
        if not self.exclude_charges:
            dst_dict['c'] = atom_charge_logits

        # apply softmax to categorical features, if requested
        if apply_softmax:
            for feat in dst_dict.keys():
                if feat in ['a', 'c', 'e']:
                    dst_dict[feat] = torch.softmax(dst_dict[feat], dim=-1)

        return dst_dict

    def step(self, g: dgl.DGLGraph, s_i: torch.Tensor, t_i: torch.Tensor,
             alpha_t_i: torch.Tensor, alpha_s_i: torch.Tensor, alpha_t_prime_i: torch.Tensor,
             node_batch_idx: torch.Tensor, edge_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor,
             cat_temp_func: Callable,
             forward_weight_func: Callable, 
             dfm_type: str = 'campbell',
             stochasticity: float = 8.0,
             high_confidence_threshold: float = 0.9, 
             last_step: bool = False,
             inv_temp_func: Callable = None,
             normalization_file_path: str = None,
             conditional_generation: bool = True,
             property_names: Optional[List[str]] = None,
             properties_for_sampling: Optional[Dict[str, float]] = None,
             multilple_values_to_one_property: Optional[Dict[str, List[float]]] = None,
             training_mode: bool = True,
             **kwargs):
        """Override step to pass multi-property parameters to forward."""
        
        device = g.device

        if stochasticity is None:
            eta = self.eta
        else:
            eta = stochasticity

        if high_confidence_threshold is None:
            hc_thresh = self.hc_thresh
        else:
            hc_thresh = high_confidence_threshold

        if dfm_type is None:
            dfm_type = self.dfm_type

        if inv_temp_func is None:
            inv_temp_func = lambda t: 1.0
        
        # Predict the destination - pass multi-property parameters
        dst_dict = self(
            g, 
            t=torch.full((g.batch_size,), t_i, device=g.device),
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=True,
            remove_com=True,
            normalization_file_path=normalization_file_path,
            property_names=property_names,
            properties_for_sampling=properties_for_sampling,
            multilple_values_to_one_property=multilple_values_to_one_property,
        )
        
        # Rest is same as parent
        dt = s_i - t_i

        # take integration step for positions
        x_1 = dst_dict['x']
        x_t = g.ndata['x_t']
        vf = self.vector_field(x_t, x_1, alpha_t_i[0], alpha_t_prime_i[0])
        g.ndata['x_t'] = x_t + dt*vf*inv_temp_func(t_i)

        # record predicted endpoint for visualization
        g.ndata['x_1_pred'] = x_1.detach().clone()

        # take integration step for node categorical features
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat == 'x':
                continue

            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata

            xt = data_src[f'{feat}_t'].argmax(-1)

            if feat == 'e':
                xt = xt[upper_edge_mask]

            p_s_1 = dst_dict[feat]
            temperature = cat_temp_func(t_i)
            p_s_1 = F.softmax(torch.log(p_s_1)/temperature, dim=-1)

            if dfm_type == 'campbell':
                xt, x_1_sampled = \
                self.campbell_step(p_1_given_t=p_s_1, 
                                xt=xt, 
                                stochasticity=eta, 
                                hc_thresh=hc_thresh, 
                                alpha_t=alpha_t_i[feat_idx], 
                                alpha_t_prime=alpha_t_prime_i[feat_idx],
                                dt=dt, 
                                batch_size=g.batch_size, 
                                batch_num_nodes=g.batch_num_edges()//2 if feat == 'e' else g.batch_num_nodes(), 
                                n_classes=self.n_cat_feats[feat]+1,
                                mask_index=self.mask_idxs[feat],
                                last_step=last_step,
                                batch_idx=edge_batch_idx[upper_edge_mask] if feat == 'e' else node_batch_idx,
                                )

            elif dfm_type == 'gat':
                x_1_sampled = torch.cat([p_s_1, torch.zeros_like(p_s_1[:, :1])], dim=-1)
                xt = self.gat_step(
                    p_1_given_t=p_s_1, 
                    xt=xt, 
                    alpha_t=alpha_t_i[feat_idx], 
                    alpha_t_prime=alpha_t_prime_i[feat_idx],
                    dt=dt, 
                    n_classes=self.n_cat_feats[feat]+1
                )

            # if we are doing edge features, we need to modify xt and x_1_sampled to have upper and lower edges
            if feat == 'e':
                e_t = torch.zeros_like(g.edata['e_t'])
                e_t[upper_edge_mask] = xt
                e_t[~upper_edge_mask] = xt
                xt = e_t

                e_1_sampled = torch.zeros_like(g.edata['e_t'])
                e_1_sampled[upper_edge_mask] = x_1_sampled
                e_1_sampled[~upper_edge_mask] = x_1_sampled
                x_1_sampled = e_1_sampled
            
            data_src[f'{feat}_t'] = xt
            data_src[f'{feat}_1_pred'] = x_1_sampled

        return g