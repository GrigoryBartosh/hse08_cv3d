#! /usr/bin/env python3

__all__ = [
    'read_grayscale_f32'
]

import click
import cv2
import numpy as np
import pims


@pims.pipeline
def _to_gray_float32(rgb):
    rgb = (rgb / 255.0).astype(np.float32)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def read_grayscale_f32(path_to_sequence: str) -> pims.FramesSequence:
    return _to_gray_float32(pims.open(path_to_sequence))


# pylint:disable=no-value-for-parameter
@click.command()
@click.argument('frame_sequence')
def _cli(frame_sequence):
    sequence = read_grayscale_f32(frame_sequence)
    click.echo("Press 'q' to stop")
    for image in sequence:
        cv2.imshow('Video', image)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    _cli()
