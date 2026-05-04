import argparse
import os
import shutil

import torch
import yaml

from documenter import Documenter
from trainer import Trainer
import logging


def main():
    parser = argparse.ArgumentParser(description='train network')
    parser.add_argument('param_file', help='yaml file location with all the parameters')
    parser.add_argument('-c', '--use_cuda', action='store_true', default=False,
        help='whether cuda should be used')
    parser.add_argument('-p', '--plot', action='store_true', default=False,
        help='only run the evaluation script')
    parser.add_argument('-g', '--generate', action='store_true', default=False,
        help='generate and save a new sample from a trained model')
    parser.add_argument('--generate-latent', action='store_true', default=False,
        help='encode validation showers with the trained model and save latent features')
    parser.add_argument('--generate-from-latent', action='store_true', default=False,
        help='decode samples from latent_features and incident_energies in an input hdf5 file')
    parser.add_argument('--latent_input_path', default=None,
        help='path to input hdf5 file with latent_features and incident_energies')
    parser.add_argument('-n', '--nsamples', type=int, default=100000,
        help='number of samples, only used for ds2')
    parser.add_argument(
        "-lne",
        "--log_energy",
        type=float,
        default=None,
        help="Single value of log2 energy to be conditioned",
    )
    parser.add_argument('-d', '--model_dir', default=None,
        help='directory used to load a model')
    parser.add_argument('-its', '--model_name', default='_last',
        help='name of the model used to generate the new sample')
    args = parser.parse_args()

    with open(args.param_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    use_cuda = torch.cuda.is_available() and args.use_cuda
    device = 'cuda:0' if use_cuda else 'cpu'

    if args.generate_from_latent and args.latent_input_path is None:
        parser.error("--latent_input_path is required when using --generate-from-latent")

    if args.plot or args.generate or args.generate_latent or args.generate_from_latent:
        doc = Documenter(params['run_name'], existing_run=args.model_dir)
    else:
        doc = Documenter(params['run_name'])

    try:
        shutil.copy(args.param_file, doc.get_file('params.yaml'))
    except shutil.SameFileError:
        pass
    print('device: ', device)

    dtype = params.get('dtype', '')
    if dtype=='float64':
        torch.set_default_dtype(torch.float64)
    elif dtype=='float16':
        torch.set_default_dtype(torch.float16)
    elif dtype=='float32':
        torch.set_default_dtype(torch.float32)
    
    print(args.model_name)
    trainer = Trainer(params, device, doc)
    if args.generate_latent:
        print(f"Loading model from {args.model_name} for latent inference.")
        trainer.load(args.model_name)
        trainer.generate_latent(
            args.nsamples,
            single_energy=args.log_energy,
            batch_size=params.get('batch_size', 8192),
            model_label=args.model_name,
        )
    elif args.generate_from_latent:
        print(f"Loading model from {args.model_name} for latent decoding.")
        trainer.load(args.model_name)
        trainer.generate_from_latent(
            latent_input_path=args.latent_input_path,
            num_samples=args.nsamples,
            batch_size=params.get('batch_size', 8192),
        )
        if args.plot:
            trainer.plot_default_from_caloch(
                sample_name="samples.hdf5", eval_name="final", cut=1.515e-3
            )
    elif args.generate:
        print(f"Loading model from {args.model_name} for generation.")
        trainer.load(args.model_name)
        trainer.generate(args.nsamples, single_energy=args.log_energy, batch_size=params.get('batch_size', 8192))
        if args.plot:
            trainer.plot_default_from_caloch(
                sample_name="samples.hdf5", eval_name="final", cut=1.515e-3
            )
    elif args.plot:
        trainer.plot_default_from_caloch(
                sample_name='samples.hdf5', eval_name='final', cut=1.515e-3
                ) 
    else:
        logging.info('starting training')
        trainer.train()

if __name__=='__main__':
    main()
