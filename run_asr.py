import argparse
import ast
import os
from pathlib import Path
import json
import glob
import tempfile
from google.cloud import storage
from say_what.models import W2V2ForASR, WhisperForASR
from say_what.io import upload_to_gcs, search_gcs, download_checkpoint_from_gcs



def save_results(results, output_dir:Path, bucket = None):
    """
    """
    json_string = json.dumps(results)
    outpath = output_dir / 'ASR_results.json'
    if bucket is None:
        with open(outpath, 'w') as outfile:
            json.dump(json_string, outfile)
    else:
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmppath = Path(tmpdirname) / 'ASR_results.json'
            with open(tmppath, 'w') as outfile:
                json.dump(json_string, outfile)
            
            upload_to_gcs(outpath, tmppath, bucket)

def run(args):
    """
    """
    if any(args.cloud.values()):
        data = search_gcs('waveform.wav', args.input_dir, args.bucket)
    else:
        pat = args.input_dir / '*/waveform.wav'
        data = glob.glob(str(pat))#get full file paths in the input directory
    
    if args.model_type == 'w2v2':
        model = W2V2ForASR(args.checkpoint, args.model_size)
    elif args.model_type == 'whisper':
        model = WhisperForASR(args.checkpoint, args.model_size)
    else:
        raise NotImplementedError(f'{args.model_type} is not an implemented ASR model.')

    results = {}
    for d in data:
        r = model.transcribe(d, return_timestamps=args.return_timestamps, return_pauses=args.return_pauses, pause_s=args.pause_s)
        results[d] = r
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default = '', help='Set path to directory with wav files to process. Can be local directory or google cloud storage bucket.')
    parser.add_argument("--save_outputs", default=False, type=ast.literal_eval, help="Specify whether to save outputs.")
    parser.add_argument("--output_dir", default= '', help='Set path to directory where outputs should be saved.')
    #run methods
    #model specifics
    parser.add_argument("--model_type", default='w2v2', choices = ['w2v2','whisper'], help='Specify model to use.')
    parser.add_argument("--model_size", default="base", help='Specify model size.')
    parser.add_argument("--checkpoint", default = '', help='Specify model checkpoint')
    #transcription specifics
    parser.add_argument("--return_timestamps", default=True, type=ast.literal_eval, help= "Specify whether to get timestamps for each audio file")
    parser.add_argument("--return_pauses", default=True, type=ast.literal_eval, help="Specify whether to find long pauses.")
    parser.add_argument("--pause_s", default=0.1, type=float, help='Set threshold for a long pause in SECONDS.')
    #GCS
    parser.add_argument("--cloud",  nargs="+", type=ast.literal_eval, default=[False, False, False], help="Specify which files are located on the cloud/should be located on the cloud [input_dir, output_dir, checkpoint]")
    parser.add_argument("--local_dir", default='', help="Specify location to save files downloaded from bucket")
    parser.add_argument('-b','--bucket_name', default='', help="google cloud storage bucket name")
    parser.add_argument('-p','--project_name', default='', help='google cloud platform project name')
    
    args = parser.parse_args()

    args.input_dir = Path(args.input_dir)
    args.output_dir = Path(args.output_dir)
    args.local_dir = Path(args.local_dir).absolute()
    args.checkpoint = Path(args.checkpoint)

    #check for cloud
    assert len(args.cloud) == 3, 'Must have a True/False value for all directory/file inputs'
    if any(args.cloud):
        assert args.project_name is not None, 'Must give a project name for use with cloud.'
        assert args.bucket_name is not None, 'Must give a bucket name for use with cloud.'
        client = storage.Client(args.project_name)
        args.bucket = client.get_bucket(args.bucket_name)
    else:
        args.bucket = None

    args.cloud = {'input':args.cloud[0], 'output':args.cloud[1], 'checkpoint': args.cloud[2]}

    if not args.cloud['input']:
        args.input_dir = args.input_dir.absolute()
        assert args.input_dir.exists(), f'Input directory {args.input_dir} does not exist locally.'
    if not args.cloud['output']:
        args.output_dir = args.output_dir.absolute()
        if not args.output_dir.exists():
            os.makedirs(args.output_dir)
    if not args.cloud['checkpoint']:
        args.checkpoint = args.checkpoint.absolute()
        assert args.checkpoint.exists(), f'Checkpoint {args.checkpoint} does not exist locally.'
    else:
        args.checkpoint = Path(args.checkpoint)
        local_checkpoint = args.local_dir / 'checkpoints'
        download_checkpoint_from_gcs(checkpoint_prefix = args.checkpoint, local_path = local_checkpoint, bucket=args.bucket)
        args.checkpoint = local_checkpoint
    

    results = run(args)

    if args.save_outputs:
        save_results(results, args.output_dir, args.bucket)

if __name__ == "__main__":
    main()
   