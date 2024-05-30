# ASR Model implementations for NAIP
Includes a handful of model types implemented to get transcriptions (for single audio files - batches not yet implemented), timestamps, and find pauses over a certain duration. Also contains simple metrics like WER/CER

## Installation
To intall say-what, follow the instructions below

```
   $ git clone https://github.com/dwiepert/say-what.git
   $ cd say-what
   $ conda env create -f environment.yml
   $ conda activate say_what
   $ pip install .
```

## Using WaveDataset
### dataset_config
### collate functions with DataLoader

## Using ASR Models
See examples of using ASR model in [run_asr.py](https://github.com/dwiepert/say-what/blob/main/run_asr.py).
Please note that Whisper models take a longer time to transcribe an audio file depending on the length of the file as it was originally built to only handle 30s clips.
### Model Checkpoints
There are a handful of ways to specify and work with checkpoints
1. Some models can be given a `repo_id` or that is a relative path/model name. For example, Whisper can take `"large"` as a checkpoint to specify which model to use (see [Whisper models](https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages) for more. In this case, the model size is the checkpoint). Wav2Vec2 can take some hugging face repo names like `"facebook/wav2vec2-base-960h"`. For some more examples, see [_constants.py](https://github.com/dwiepert/say-what/blob/main/say_what/constants/_constants.py).
2. All models can be given a local directory or file with the pretrained model. This package includes a few ways to download these models to the local directory.
   - Whisper: For local checkpoints, you must download a `.pt` file. These checkpoints can be downloaded using `download_checkpoint_from_url`. If a downloaded model is saved on google cloud storage for space saving, it can also be downloaded using `download_checkpoint_from_gcs`. When giving a local checkpoint to the model, it must be given as the full file path to the `.pt` file (e.g. `local_dir_path\medium.pt`)
   - W2V2: For local checkpoints, you must download the entire checkpoint directory from hugging face using `download_checkpoint_from_hf`. Otherwise, if the directory is saved to the cloud, it can also be downloaded using `download_checkpoint_from_gcs`. Note that `download_checkpoint_from_hf` has pre-built options that can be downloaded ("base", "large", "large-robust", "large-self"), or you can give a `repo_id` and other identifying informationto download checkpoints outside of the pre-built options. The `repo_id` must be from [huggingface.co](https://huggingface.co). When giving a local checkpoint to the model, it must be given as the full path to a downloaded checkpoint directory (e.g. `local_dir_path\w2v2-base-960h`)
   
#### Using download functions
`download_checkpoint_from_hf` takes the following parameters:
- checkpoint: str, Path, directory where the checkpoint file(s) should be saved. Must be full file path to where to SAVE the checkpoint (e.g., `local_dir_path/checkpoint_name`). 
- model_size: str, size of the model to download (e.g., large, small, tiny, tiny.en). This is not required if instead downloading using repo_id.
- model_type: str, specify model type (e.g. whisper, w2v2). This is not required if instead downloading using repo_id.
- repo_id: str, repo_id in hugging face. This is used if you want to download a model that may not be available to download through specification of model size and type.
- filename: optional filename if downloading a single file instead of directory
- subfolder: str, optional, specify if there is a file in a subdirectory of the hugging face repo

`download_checkpoint_from_url` takes the following parameters:
- checkpoint: str, path to where checkpoint should be stored. This should be the full file path as you use this variable to specify the download target ( .pt for Whisper)
- model_size: str, size of the model to download (e.g., large, small, tiny, tiny.en)
-  model_type: str, specify model type (e.g. whisper)
- in_memory: bool set to false

`download_checkpoint_from gcs` takes the following parameters:
- checkpoint: str, path to save checkpoint to on local computer. If saving a directory, you only need to give path to the directory where you want to save it. Otherwise, full file path must be given.
- checkpoint_prefix: str, location of checkpoint in bucket. Bucket name should be stripped from this.
- dir: bool, specify if downloading a directory or single file
- project_name: cloud storage project name
- bucket_name: cloud storage bucket name
Note that this function requires access to GCS, which can be given with `gcloud auth application-default login` and `gcloud auth application-default set-quota-project project_name`



### Models and functionality
Three types of models are available to use: Whisper, W2V2. Note that whisper does NOT allow non-words while W2V2 does.

All models have the following method:
- `model.transcribe(audio: either audio path or waveform, return_timestamps: bool, return_pauses: bool, pause_s: float, threshold for long pause in seconds)`: transcribe a single audio file and optionally get timestamps and long pauses
-

#### Whisper
Initialize whisper with the following parameters:
```
from say_what.models import WhisperForASR
model = WhisperForASR(checkpoint = checkpoint, model_size=model_size)
```
Where `checkpoint` is either just the model size as a string (e.g. 'base', 'medium') or a full file path to a `.pt` file.


#### W2V2
Initialize W2V2 with the following parameters:
```
from say_what.models import W2V2ForASR
model = W2V2ForASR(checkpoint = checkpoint, model_size=model_size)
```
Where `checkpoint` is either a hugging face repo name or a path to a local directory containing the model files.

### Metrics
Word error rate (`wer(...)`) and character error rate (`cer(...)`)are both implemented. Both metrics take the following paramters:
- reference: target transcription
- hypothesis: predicted transcription
- print: boolean indicating whether to print aligned transcription to console (default = False)

