#!/usr/bin/env python2.7

import argparse

def parser():

    parser = argparse.ArgumentParser(
        description="Python program that uses OpenAI Gym to teach Marvin how \
                    to walk.")

    parser.add_argument(
        '-w',
        '--walk',
        action='store_true',
        help='display only the walking process',
        required=False)

    parser.add_argument(
        '-l',
        '--load',
        type=str,
        default=None,
        metavar='file',
        help='load weights for Marvin agent from a file \
        (skip training process if this option is specified)',
        required=False)

    parser.add_argument(
        '-s',
        '--save',
        type=str,
        default=None,
        metavar='file',
        help='save weights to a file after running the program',
        required=False)

    return parser.parse_args()

class MarvinFlags(object):

    def __init__(self):
        self.flags = parser()
        self.walk = self.flags.walk
        self.load = self.flags.load
        self.save = self.flags.save
        return None
