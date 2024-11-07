> Full implementation will be made available upon acceptance of the paper.

# <span style="color: orange">M3DSR</span>

Official repository for M3DSR, a generative adversarial network for 3D medical image reconstruction. 

![M3DSR qualitative results across every training epoch](./figures/sample.gif)

<hr style="border:2px solid orange">

# <span style="color: orange">Citation</span>

If you find this project useful, please consider citing the paper:
```
@article{
...
}
```

<hr style="border:2px solid orange">

# <span style="color: orange">Quick Inference</span>

#### Reconstruction

   ```python
   import torch
   from monai.inferers.inferer import SlidingWindowInferer
   
   # Define device
   device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
   
   # Set patch size (D, H, W)
   patch_size = (16, 16, 16)
   
   # Define inferer (Optional)
   inferer = SlidingWindowInferer(roi_size=patch_size, overlap=0.125, mode='gaussian', sw_batch_size=16, sw_device=device, device=device)
   
   # Two random volumes (batch size of 2) with 1 channel and size of 64x64x64 (D, H, W). Mean is 0 and STD is 1.
   volumes = torch.randn(2, 1, 64, 64, 64).to(device)
   
   # Avoid HTTP Error 403
   torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
   
   # Load model (check pretrained_models/README.md)
   model = torch.hub.load(
      'joaolcguerreiro/M3DSR', 'load_model',
      checkpoint_url='https://drive.google.com/uc?export=download&id=15mrIzDnomlko1PXFHsnlF12DY9s4lf2p', pretrained=True,
      in_channels=1, out_channels=1, num_features=192, num_rrdb_blocks=1, num_rrdb_db_blocks=2, num_rrdb_db_conv_blocks=3, num_rrdb_db_growth_features=96, scale_factor=2, residual_learning=True
   ).to(device)
   model.eval()
   
   # Generated volumes. Model expects z-score normalized input volumes.
   gen_volumes = model(volumes) if not inferer else inferer(inputs=volumes, network=model)
   ```

<hr style="border:1px solid gray">

#### Feature Extraction

   ```python
   import torch
   from monai.inferers.inferer import SlidingWindowInferer
   
   # Same as Reconstruction
   ...
   
   # Load model (check pretrained_models/README.md)
   model = torch.hub.load(
      'joaolcguerreiro/M3DSR', 'load_model',
      checkpoint_url='https://drive.google.com/uc?export=download&id=15mrIzDnomlko1PXFHsnlF12DY9s4lf2p', pretrained=True,
      in_channels=1, out_channels=1, num_features=192, num_rrdb_blocks=1, num_rrdb_db_blocks=2, num_rrdb_db_conv_blocks=3, num_rrdb_db_growth_features=96, scale_factor=2, residual_learning=True
   ).to(device)
   model.forward = model.extract_features
   model.eval()
   
   # Extract features. Model expects z-score normalized input volumes.
   features = model(volumes) if not inferer else inferer(inputs=volumes, network=model)
   ```

<hr style="border:2px solid orange">

# <span style="color: orange">Usage</span>

### Setup Environment

1) Clone GitHub repository:

   ```
   git clone https://github.com/joaolcguerreiro/M3DSR.git
   cd M3DSR
   ```

2) Create environment:
   ```
   conda create -n m3dsr python=3.10.11
   conda activate m3dsr
   ```

3) Install Python dependencies (Python 3.10 is recommended):
   ```
   pip install -r requirements.txt
   ```

<hr style="border:1px solid gray">

### Prepare Data

1) See [data/README.md](data/README.md) for details.

2) After preparing the data, navigate to `configs/config.yaml` and adjust data paths.

<hr style="border:1px solid gray">

### Train

1) Navigate to `configs/config.yaml` and adjust configuration parameters needed. For example configurations see [configs/examples](configs/examples).

2) Navigate to `src/scripts/`.

3) Run script:
   ```
   python train.py
   ```

> Note: Validation is optional. Use option `--val=True` if validation is intended.

> Note: Set `LR_PATCH_SIZE` in `config.yaml` file for patch-based training (recommended). Patch-based training is not employed when `LR_PATCH_SIZE=[-1, -1, -1]` (more computationally expensive). Also, for `LR_PATCH_SIZE` equal or smaller than `[8, 8, 8]` the `batch_size` is required to be greater than `1` because of the discriminator batch normalization.

<hr style="border:1px solid gray">

### Load Pre-trained

1) See [pretrained_models/README.md](pretrained_models/README.md) to download available pre-trained models or use checkpoint path from a previous experiment.

2) Navigate to `configs/config.yaml` and update `PRE_TRAINED_PATH` parameters.

3) Navigate to `src/scripts/`.

4) Run script:
   ```
   python train.py
   ```

<hr style="border:1px solid gray">

### Resume Train

1) Navigate to `experiments/` and copy experiment path.

2) Navigate to `src/scripts/`.

3) Run script:
   ```
   python train.py --resume=<experiment_path>
   ```

> Note: Load pre-trained differs from resume training since in pre-training only the model weights are loaded and the training is reset. If you resume training the last epoch checkpoint weights will be prioritized over the pre-trained weights.

<hr style="border:1px solid gray">

### Test

1) Navigate to `experiments/` and copy experiment path.

2) Navigate to `src/scripts/`.

3) Run script:
   ```
   python test.py --exp=<experiment_path> --epoch=<epoch> [--lr-patch-size=<lr_patch_size>]
   ```

> Note: Results are in `<experiment_path>/results`. Additionally, check `<experiment_path>/logs/test` and `<experiment_path>/figures/test` for more insights.

> Note: Increase testing `PATCH_SLIDING_WINDOW_BATCH_SIZE` in `config.yaml` file to accelerate testing time. It controls the number of concurrent sliding window patches being processed. Notice, this will also increase memory used.

> Note: Set `--lr-patch-size` to override the training low-resolution patch size.

<hr style="border:1px solid gray">

### Inference

1) Get model checkpoint path.

2) Navigate to `src/scripts/`.

3) Run script:
   ```
    python inference.py --checkpoint=<path_to_checkpoint> --lr-dir=<path_to_lr_dir> --save-dir=<path_to_save_dir> [--batch-size=<batch_size>] [--patch-sliding-window-batch-size=<patch_sliding_window_batch_size>] [--lr-patch-size=<lr_patch_size>]
   ```

> Note: Increase `--patch-sliding-window-batch-size` to accelerate inference time. It controls the number of concurrent sliding window patches being processed. Notice, this will also increase memory used.

> Note: Set `--lr-patch-size` to override the training low-resolution patch size.

<hr style="border:1px solid gray">

### Experiment Tracking

Everytime you train a model, an experiment folder is built (unless training is resumed, in which case the folder from the prior run is used). The configuration file for the execution is stored in the experiment folder, allowing the original parameters to be retrieved when resuming training.

See [experiments/README.md](experiments/README.md) for details.

<hr style="border:2px solid orange">

# <span style="color: orange">License</span>

This project is licensed under MIT license. See [LICENSE](LICENSE) for details.
