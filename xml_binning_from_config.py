import yaml
import sys
import xml.etree.ElementTree as ET
import particle
from argparse import ArgumentParser


def parse_args(argv):
    parser = ArgumentParser(description="Convert a YAML file to an XML file for use with Particle")
    parser.add_argument("config-yaml", help="The input YAML file")
    parser.add_argument("output-xml", help="The output XML file")
    parser.add_argument("--particle", type=str, default="pion", help="The particle type to use (default: pion)")
    return parser.parse_args(argv)

def get_voxel_spec(config):

    e_vox = {}
    h_vox = {}


def main(argv):

    args = parse_args(argv)
    with open(args.config_yaml, "r") as f:
        config = yaml.safe_load(f)
    


if __name__ == "__main__":
    main(sys.argv[1:])

