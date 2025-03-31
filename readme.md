# U-LFormer

Code for "Enhancing Diffusion Estimation in Single-Particle Experiments through Motion Change Analysis Using Deep Learning".

### Environment

Run the following command to create our conda environment:
```
conda env create -f environment.yml
```
However, conda cannnot correctly install GPU version PyTorch, so install it with pip:
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

### Training Procedure

First, a training dataset should be generated with `generate_dataset.py`

To train the model, change the `train_address` and `valid_address` and run the python command below to get a model predicting alpha and K.
```
python3 maintrack_others.py
```


### Reproducing results

#### Single Trajectory Task

Run the first part (Single Trajectory Task) in `Evaluate.ipynb`.

#### Video Ensemble Task

For video trajectory linking, we use MatLab and TrackMate together to extract trajectories from videos. Code is in `code/traj_link/main.m`

Then, run the second part (Ensemble Task) in `Evaluate.ipynb`.