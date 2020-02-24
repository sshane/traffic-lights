# traffic-lights/new_data

Simply place `.hevc` files into the `./videos` directory. They will be extracted as `.png` frames into `traffic-lights/new_data/extracted/{video_name}/{frame_idx}.png`

To start extraction, run:

```python
python traffic-lights/tools/extract_frames.py
```

To stop the extraction process before completion, simply delete the `delete_to_stop` file in `traffic-lights/new_data`. Once the current video has finished extraction, it will stop.