## Totally have 6 properties
alpha, cv, gap, homo, lumo, mu

## Training
To train the regressor with specific property, run **train_regressor.py** with correspoding config file.
E.g. `python train_regressor.py --config=configs/regressor_alpha.yaml`. Config files for regressor are in configs dir under property_regressor parent dir.

## Testing 
To use the regressor to do prediction and calculate MAE value with input property values, we need to pass several argument to  **test_regressor.py** function. E.g. 
`python test_regressor.py --checkpoint=model_output/alpha/checkpoints/gvp-regressor-epoch=358-val_loss=0.0061.ckpt --config=configs/test.yaml --input=../../sample_result/example_alpha.sdf --properties_values=../../many2many-sampling/train_half_sampled_no_atoms_alpha.npy --property_name=alpha`.
The meaning of each argument can be found in main function of **test_regressor.py**. 