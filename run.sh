#!/bin/bash
cd /home/msi/intrusion_track
python3  publisher/camera_publisher.py > /dev/null 2>&1 &
python3 workers/detector_service.py > /dev/null 2>&1 &
python3 workers/tracker_service.py > /dev/null 2>&1 &
python3 workers/reid_service.py > /dev/null 2>&1 &
python3 workers/linker_service.py > /dev/null 2>&1 &
python3 services/display_and_logger.py > /dev/null 2>&1 &
