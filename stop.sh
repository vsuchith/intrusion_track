#!/bin/bash
pkill -9 -f publisher/camera_publisher.py
pkill -9 -f workers/reid_service.py
pkill -9 -f workers/detector_service.py
pkill -9 -f services/display_and_logger.py
pkill -9 -f workers/linker_service.py
pkill -9 -f workers/tracker_service.py
pkill -9 -f services/caption_service.py
