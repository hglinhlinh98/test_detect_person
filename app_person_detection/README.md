# Information (tienln4)
```
* Model version: tiny_ssd_c32
* input_size: 320*240
* Data train: Wider_person, Crownd_human, COCO_person, VOC_person, cleaned_ECP, cleaned_City_person
* Echop: 63
* MR: 13%, mAP: 84, RT: 24ms (on EZ dataset, Intel core i7)
```

# Requirement
```
- Anaconda
- pytorch 1.2
- torchvision 0.4
- opencv-python
- pandas
```
# Run
```
python detect_imgs.py
```
