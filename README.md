# aireadi_dataloader
A PyTorch Dataset for loading and processing [AI-READI](https://docs.aireadi.org/docs/2/about) dataset. 

# PatientDataset Class Overview
The `PatientDataset` class is a PyTorch `Dataset` designed to handle patient data from the AI-READI dataset. It supports various configurations to filter, preprocess, and transform patient data for tasks like image processing and clinical analysis. This module provides MONAI-compatible dataset classes for loading OCT/OCTA/Photography imaging data, supporting both cache-based and on-demand access. It is tested with `monai.data.DataLoader` and follows MONAI-style dictionary-based outputs.

Each dataset returns a dictionary per sample with the following keys:

- `data_dict["frames"]`: the imaging data (2D or 3D tensor depending on settings)
- `data_dict["label"]`: the ground truth label associated with the sample

We strongly recommend using the [`build_dataset.py`](./examples/build_dataset.py) script to construct datasets, as it encapsulates essential logic for transformation, caching, and configuration.

### Key Features:  
- **Concept ID-based Data Loading**: Loads data with respect to a specific clinical observation or measurement, which is coded as a clinical concept ID.  
- **Easy Image-Target Pairing**: Facilitates seamless loading of image and target pairs via the dataloader, making it convenient for model training.  
- **Seamless Integration with MONAI Pipeline**: Built on MONAI’s dataset class, allowing users to apply flexible preprocessing and transformations as part of their data pipeline.
- **Support for Multiple Imaging Modalities and Devices**: Handles images from various devices, models, imaging techniques (OCT, IR, CFP, OCTA, etc.), and anatomical regions. Users can specify which device or imaging type to load, making it easy to extract specific datasets via the API.  


---

## Dataset Classes

### PatientDataset

Base dataset class for on-demand loading. Used when `args.cache_rate == 0`. Suitable for low-memory environments or quick debugging.

- Efficient lazy loading to minimize memory usage
- Great for low-resource environments and fast prototyping
- Compatible with MONAI `DataLoader`

---

### PatientCacheDataset

Caching dataset class used when `args.cache_rate > 0`. Loads a portion (or all) of the dataset into memory at initialization for faster training.

- Partial or full caching supported via `cache_rate`
- Good balance between speed and memory
- Compatible with MONAI `DataLoader`

---

### PatientFastAccessDataset

Optimized caching class for 2D slice datasets when `args.patient_dataset_type == "slice"` and imaging is `"oct"` or `"octa"`.

- Inherits from PyTorch's `IterableDataset`, enabling efficient streaming of data without full dataset indexing.
- Tailored for slice-based imaging
- Optimized for caching and fast access to OCT/OCTA slices.
- Compatible with MONAI `DataLoader`

---

## Key Parameters

### `root_dir` (str)
The root directory containing all dataset files, including imaging data, clinical metadata, and other related files.

After downloading the dataset, the structure should resemble: [your dataset path]/AIREADI/YEAR2/

Make sure to set `root_dir` to this path (i.e., the full path to the `YEAR2` folder). For example:

```python
args.root_dir = "/home/user/datasets/AIREADI/YEAR2/"
```

### `split` (str)
Specifies the dataset split to use (`'train'`, `'test'`, `'val'`). Determines the subset of patients loaded based on AIREADI predefined splits.

### `mode` (str, default: `'slice'`)
Specifies how the data should be loaded:
- **`slice`**: Loads imges for CFP and IR imaging, or individual slices from the volume data for OCT or OCTA.
- **`center_slice`**: Loads only the central slice of each volume (only available for volume data; OCT or OCTA).
- **`volume`**: Loads the entire volume (only available for volume data; OCT or OCTA).

### `imaging` (str, default: `'oct'`)
Type of imaging data to load. Should be set to one of the following options. 

- `oct`
- `octa`
- `ir`
- `cfp`
  
### `anatomic_region` (str, default: `'Macula'`)
Filters data based on the anatomical region imaged. The available options for `anatomic_region` include:

- `Macula`
- `Macula_6x6`
- `Macula_12x12`
- `Wide_Field`
- `Optic_Disc`
- `Temporal_Periphery`
- `Optic_Disc_6x6`
- `Mosaic`
- `Nasal`

**Note:** Ensure you input a valid combination of `anatomic_region` with respect to device and imaging conditions. Refer to the [AIREADI Dataloader Access Table](https://github.com/uw-biomedical-ml/AIREADI_dataloader/blob/810d8daca0ef3bfde2bc297e289eb3615284ca51/dataloader_access_table.csv) for possible valid combinations.

### `device` (str, required)
Filters patients based on the OCT device used for imaging (e.g., `'Maestro2'`,  `'Cirrus'`). This argument is required and must be explicitly set by the user to avoid biasing the dataset toward a particular device or vendor.

### `concept_id` (int, default: `0`)
Used to filter clinical data based on the specified concept ID. If `0`, the dataset returns class indices (`cls_idx`) from the `study_group`.
The `study_group` categorizes patients based on their diabetes status. If `concept_id` is set to `0`, the dataloader assigns labels according to the following categories:  

- **Healthy**: `0`  
- **Pre-diabetes (lifestyle controlled)**: `1`  
- **Oral medication and/or non-insulin injectable medication controlled**: `2`  
- **Insulin-dependent**: `3`  

If `concept_id == -1`: The dataset returns laterality labels:
- `0` for right eye (OD)
- `1` for left eye (OS)

This setting can be helpful for sanity-checking your model pipeline. Since laterality is an easily learnable signal in most imaging modalities, a model trained with `concept_id == -1` typically converges near 100% accuracy very quickly.

### `ignore_values` (list, default: `[555, 777, 888, 999]`)
A list of values to ignore in the clinical data (e.g., `[777, 999]`).

### `cache_rate` (optional, default: `None`)
A preprocessing function applied to the data before any transformations.

### `transform` (callable, optional, default: `None`)
Optional transformations to be applied to the data samples (e.g., augmentations).  
If `None`, a default transform will be used, which:

- extracts images from DICOM files,
- maps the raw label using the `concept_id`,
- and converts the `"frames"` and `"label"` keys to `torch.float32` tensors using `ToTensord`.

This ensures the data is in the correct format for training even when no custom transforms are provided.

### `octa_enface_imaging` (str or None, default: `False`)
Specifies the type of en-face slab to load when using OCTA imaging. Should be set to one of the following options when extracting en-face slab data from OCTA:

- `superficial`
- `deep`
- `outer_retina`
- `choriocapillaris`

For OCTA flow cube data, set `imaging='octa'` and keep `octa_enface_imaging = None`

### `kwargs`
Additional keyword arguments to configure the dataset.

---

## Usage

To construct datasets, use the `build_dataset()` function in `build_dataset.py`:

```python
from build_dataset import build_dataset
from monai.data import DataLoader

train_dataset = build_dataset(is_train=True, args=args)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

test_dataset = build_dataset(is_train=False, args=args)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
```

**Note:** `shuffle=True` has no effect when using `PatientFastAccessDataset`, which inherits from PyTorch's `IterableDataset`.

During training, data samples are accessed using:

```python
for batch in dataloader:
    images = batch["frames"].to(device)
    labels = batch["label"].to(device).long() // for classification
    ...
```

Refer to the example training scripts for full workflows:

- [`train_2d.py`](./examples/train_2d.py) for 2D model training
- [`train_3d.py`](./examples/train_3d.py) for 3D model training



