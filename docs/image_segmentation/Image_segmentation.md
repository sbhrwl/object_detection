# Computer Vision
Classification -> OD -> **Segementation** -> Tracking

- [Detectron2](https://colab.research.google.com/drive/1_SAxXzQOWsNJ2W-VYeP4FZ1tX58J1sJr?usp=sharing)
- [TFOD1.4](https://drive.google.com/drive/u/0/folders/12aAKFRS9rrir-Ro837Huk20Ct7wkGc0M)
- [Annotation tool](https://github.com/wkentaro/labelme)
  python labelme2coco.py label_mask --output paul.json 
- Export Inference Graph
  ```
  python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-1000 --output_directory inference_graph
  ```
- [Object Detection Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics)
- Typs of Segmentation
  <image src='typesOfSegmentation.jpg'>
