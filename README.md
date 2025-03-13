# YOLOgon
Practical project of CNN techniques applied to Big Data using a live webcam.

For the _Big Data Programming_ course. University of Skövde, 2025

---

## How to Run

1. Start a `venv` and install the requirements from `requirements.txt`.
2. Run the `1_image_capture.py` script in order to generate the image database for training.
3. Run `2_auto_label.py` to generate the labels for the images.
4. Run `3_splitter.py` to split the images in _train_ and _validation_.
5. Run `4_train.py` to train the model.