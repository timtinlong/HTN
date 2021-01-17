### instructions for setting up on anaconda


conda create -n myenv python=3.7
conda activate myenv
pip install tensorflow-gpu==1.15.5
pip install motpy flask loguru jsonpickle opencv-python==4.5.1.48 imageio==2.9.0 imageio-ffmpeg==0.4.3 tf-slim==1.1.0 Pillow==8.1.0 h5py==2.10.0
python people_tracking_yolo.py -m yolov3 -c 0.5 -sm 0 -i shop -v 0


### Run 

To test this code, set up the above environment - to get updates to localhost, run:

$ python people_tracking_yolo_flask_run.py -i shop

To visualize and process the entire video, run:

$ python people_tracking_yolo_local_run.py -i shop -fb 0


