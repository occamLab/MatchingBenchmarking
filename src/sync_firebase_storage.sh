#!/bin/bash

gsutil -m rsync -r gs://clew-sandbox/visual_alignment_benchmarking $(dirname $(pwd))/image_data_2