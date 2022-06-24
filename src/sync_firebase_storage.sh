#!/bin/bash

gsutil -m rsync -r gs://depthbenchmarking.appspot.com/visual_alignment_benchmarking $(dirname $(pwd))/image_data
