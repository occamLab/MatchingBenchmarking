#!/bin/bash

gsutil -m rsync -r gs://clew-sandbox.appspot.com/visual_alignment_benchmarking $(dirname $(pwd))/image_data_2
gsutil -m rsync -r gs://clew-sandbox.appspot.com/geo_location/logs $(dirname $(pwd))/image_data_2_logs
