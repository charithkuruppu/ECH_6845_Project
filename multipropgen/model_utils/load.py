from multipropgen.models.flowmol import FlowMol
from multipropgen.models.flowmol_multi import FlowMolMutli
from pathlib import Path
import yaml
from multipropgen.data_processing.data_module import MoleculeDataModule

# for qm9 dataset: find min and max values for Gaussian expansion
Gaussian_expansion_start_dict = {
    'mu': 0.0, 'alpha': 6.31, 'homo': -0.4286, 'lumo': -0.175, 'gap': 0.0246, 'cv': 6.002,
}

Gaussian_expansion_stop_dict = {
    'mu': 29.5564, 'alpha': 196.62, 'homo': -0.1017,'lumo': 0.1935, 'gap': 0.6221, 'cv': 46.969,
}

def read_config_file(config_file: Path) -> dict:
    # process config file into dictionary
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def model_from_config(config: dict, seed_ckpt: Path = None) -> FlowMol:
    atom_type_map = config['dataset']['atom_map']
    conditional_generation = config['dataset']['conditioning']['enabled']
    if config.get('model_setting'):
        property_embedding_dim = config['model_setting']['property_embedding_dim']
        gaussian_expansion = config['model_setting']['gaussian_expansion']['enabled']
        n_gaussians = config['model_setting']['gaussian_expansion']['n_gaussians']
        enable_tune_cv_min = config['model_setting']['gaussian_expansion'].get('enable_tune_cv_min', False)
        properties_handle_method = config['model_setting']['properties_handle_method']
    else:
        property_embedding_dim = None
        gaussian_expansion = False
        n_gaussians = None
        enable_tune_cv_min = False
        properties_handle_method = None

    gaussian_start, gaussian_stop = None, None

    if gaussian_expansion and conditional_generation:
        properties = config['dataset']['conditioning']['property']
        gaussian_start = Gaussian_expansion_start_dict.get(properties)
        gaussian_stop = Gaussian_expansion_stop_dict.get(properties)

    # get the sample interval (how many epochs between drawing/evaluating)
    sample_interval = config['training']['evaluation']['sample_interval']
    mols_to_sample = config['training']['evaluation']['mols_to_sample']


    # get the filepath of the n_atoms histogram
    processed_data_dir = Path(config['dataset']['processed_data_dir'])
    n_atoms_hist_filepath = processed_data_dir / 'train_data_n_atoms_histogram.pt'
    marginal_dists_file = processed_data_dir / 'train_data_marginal_dists.pt'

    if seed_ckpt is not None:
        model = FlowMol.load_from_checkpoint(seed_ckpt, 
                                            strict=False,
                                            atom_type_map=atom_type_map,
                                            n_atoms_hist_file=n_atoms_hist_filepath,
                                            marginal_dists_file=marginal_dists_file,
                                            sample_interval=sample_interval,
                                            n_mols_to_sample=mols_to_sample,
                                            vector_field_config=config['vector_field'],
                                            interpolant_scheduler_config=config['interpolant_scheduler'], 
                                            lr_scheduler_config=config['lr_scheduler'],
                                            property_embedding_dim=property_embedding_dim,
                                            gaussian_expansion=gaussian_expansion,
                                            gaussian_start=gaussian_start,
                                            gaussian_stop=gaussian_stop,
                                            n_gaussians=n_gaussians,
                                            conditional_generation=conditional_generation,
                                            properties_handle_method=properties_handle_method,
                                            **config['mol_fm'])
    else:
        model = FlowMol(atom_type_map=atom_type_map, 
                        n_atoms_hist_file=n_atoms_hist_filepath,
                        marginal_dists_file=marginal_dists_file,
                        sample_interval=sample_interval,
                        n_mols_to_sample=mols_to_sample,
                        vector_field_config=config['vector_field'],
                        interpolant_scheduler_config=config['interpolant_scheduler'], 
                        lr_scheduler_config=config['lr_scheduler'],
                        property_embedding_dim=property_embedding_dim,
                        gaussian_expansion=gaussian_expansion,
                        gaussian_start=gaussian_start,
                        gaussian_stop=gaussian_stop,
                        n_gaussians=n_gaussians,
                        conditional_generation=conditional_generation,
                        properties_handle_method=properties_handle_method,
                        **config['mol_fm'])
    
    return model

def model_from_config_multi(config: dict, seed_ckpt: Path = None) -> FlowMolMutli:
    atom_type_map = config['dataset']['atom_map']
    conditional_generation = config['dataset']['conditioning']['enabled']
    property_embedding_dim = config['model_setting']['property_embedding_dim']
    gaussian_expansion = config['model_setting']['gaussian_expansion']['enabled']
    n_gaussians = config['model_setting']['gaussian_expansion']['n_gaussians']
    gaussian_start, gaussian_stop = None, None
    enable_tune_cv_min = config['model_setting']['gaussian_expansion'].get('enable_tune_cv_min', False)

    if gaussian_expansion and conditional_generation:
        properties = config['dataset']['conditioning']['property']
        gaussian_start = Gaussian_expansion_start_dict.get(properties)
        gaussian_stop = Gaussian_expansion_stop_dict.get(properties)

    properties_handle_method = config['model_setting']['properties_handle_method']

    # get the sample interval (how many epochs between drawing/evaluating)
    sample_interval = config['training']['evaluation']['sample_interval']
    mols_to_sample = config['training']['evaluation']['mols_to_sample']

    # get the list of property names
    property_names = config['model_setting'].get('property_names', [])

    # get the filepath of the n_atoms histogram
    processed_data_dir = Path(config['dataset']['processed_data_dir'])
    n_atoms_hist_filepath = processed_data_dir / 'train_data_n_atoms_histogram.pt'
    marginal_dists_file = processed_data_dir / 'train_data_marginal_dists.pt'

    if seed_ckpt is not None:
        model = FlowMolMutli.load_from_checkpoint(seed_ckpt, 
                                            atom_type_map=atom_type_map,
                                            n_atoms_hist_file=n_atoms_hist_filepath,
                                            marginal_dists_file=marginal_dists_file,
                                            sample_interval=sample_interval,
                                            n_mols_to_sample=mols_to_sample,
                                            vector_field_config=config['vector_field'],
                                            interpolant_scheduler_config=config['interpolant_scheduler'], 
                                            lr_scheduler_config=config['lr_scheduler'],
                                            property_embedding_dim=property_embedding_dim,
                                            gaussian_expansion=gaussian_expansion,
                                            gaussian_start=gaussian_start,
                                            gaussian_stop=gaussian_stop,
                                            n_gaussians=n_gaussians,
                                            conditional_generation=conditional_generation,
                                            properties_handle_method=properties_handle_method,
                                            property_names=property_names,
                                            **config['mol_fm'])
    else:
        model = FlowMolMutli(atom_type_map=atom_type_map, 
                        n_atoms_hist_file=n_atoms_hist_filepath,
                        marginal_dists_file=marginal_dists_file,
                        sample_interval=sample_interval,
                        n_mols_to_sample=mols_to_sample,
                        vector_field_config=config['vector_field'],
                        interpolant_scheduler_config=config['interpolant_scheduler'], 
                        lr_scheduler_config=config['lr_scheduler'],
                        property_embedding_dim=property_embedding_dim,
                        gaussian_expansion=gaussian_expansion,
                        gaussian_start=gaussian_start,
                        gaussian_stop=gaussian_stop,
                        n_gaussians=n_gaussians,
                        conditional_generation=conditional_generation,
                        properties_handle_method=properties_handle_method,
                        property_names=property_names,
                        **config['mol_fm'])
    
    return model

def data_module_from_config(config: dict) -> MoleculeDataModule:
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']

    # determine if we are doing distributed training
    if config['training']['trainer_args']['devices'] > 1:
        distributed = True
    else:
        distributed = False

    data_module = MoleculeDataModule(dataset_config=config['dataset'],
                                     dm_prior_config=config['mol_fm']['prior_config'],
                                     batch_size=batch_size, 
                                     num_workers=num_workers, 
                                     distributed=distributed)
    
    return data_module