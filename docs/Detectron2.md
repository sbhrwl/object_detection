# [Detectron2](https://github.com/facebookresearch/detectron2)
- [Detectron2 Model Zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)
- [Colab setup](https://colab.research.google.com/drive/1V7cXOGYXWNuHl3lIz1VHMTcIRVxeLD5O?usp=sharing)
- [Detectron2 Balloon dataset](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
  - Other In-built features
    - Keypoint Detections
    - Panoptic Segmentation
    - Panoptic segmentation on a video
      - Install dependencies, download the video, and crop 5 seconds for processing
        - Youtube downloader library: youtube-dl
      - Run frame-by-frame inference demo on this video
- [Cards Dataset](https://drive.google.com/file/d/1hwXAfv2Li6v-bweJQuAQq-lSSKSt-PL3/view?usp=sharing)
  - Install detectron2
  - Register the dataset
    - customtrain is the name of experiment, this enables us to use dataset multiple times
    - Experiment Name
    - Annotation file location
    - Training Images location
  - Create catalogs
    - Catalogue for Annotations
      ```
      sample_metadata = MetadataCatalog.get("customtrain")
      ```
    - Catalogue for Images
      ```
      dataset_dicts = DatasetCatalog.get("customtrain")
      ```
  - Model Training
    - Import Trainer class DefaultTrainer
      ```
      from detectron2.engine import DefaultTrainer
      ```
    - Select Pretrained Model
      ```
      cfg = get_cfg()
      cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
      ```
    - Specify Train dataset
      ```
      cfg.DATASETS.TRAIN = ("customtrain",)
      cfg.DATASETS.TEST = ()
      ```
    - Initialise Hyperparameters
      ```
      cfg.DATALOADER.NUM_WORKERS = 2
      cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
      cfg.SOLVER.IMS_PER_BATCH = 2
      cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
      cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
      cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
      cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
      # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
      ```
    - Train Model
      ```
      trainer = DefaultTrainer(cfg) 
      trainer.resume_or_load(resume=True)
      trainer.train()
      ```
  - Output of Training
    ```
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
    ```
  - Inference/Predictions
    ```
    cfg.DATASETS.TEST = ("customtrain", ) # Using train dataset for Inference as well
    predictor = DefaultPredictor(cfg)
    ```
  - Visualise Inferenced Images
  - Evaluation
    - Import Libraries
      ```
      from detectron2.evaluation import COCOEvaluator, inference_on_dataset
      from detectron2.data import build_detection_test_loader
      ```
    - Initialise COCOEvaluator
      ```
      evaluator = COCOEvaluator("customtrain", output_dir="./output/")
      val_loader = build_detection_test_loader(cfg, "customtrain")
      ```
    - Metrics
      ```
      print(inference_on_dataset(trainer.model, val_loader, evaluator))
      ```
      ```
      [08/25 08:40:21 d2.evaluation.fast_eval_api]: Evaluate annotation type *bbox*
      [08/25 08:40:21 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 0.11 seconds.
      [08/25 08:40:21 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
      [08/25 08:40:21 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 0.02 seconds.
       Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.692
       Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.826
       Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.821
       Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
       Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.681
       Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.701
       Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.741
       Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.873
       Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.873
       Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
       Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.889
       Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.873
      [08/25 08:40:21 d2.evaluation.coco_evaluation]: Evaluation results for bbox: 
      |   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
      |:------:|:------:|:------:|:-----:|:------:|:------:|
      | 69.206 | 82.580 | 82.119 |  nan  | 68.134 | 70.119 |
      [08/25 08:40:21 d2.evaluation.coco_evaluation]: Some metrics cannot be computed and is shown as NaN.
      [08/25 08:40:21 d2.evaluation.coco_evaluation]: Per-category bbox AP: 
      | category   | AP     | category   | AP     | category   | AP     |
      |:-----------|:-------|:-----------|:-------|:-----------|:-------|
      | ace        | 83.419 | jack       | 53.236 | king       | 64.442 |
      | nine       | 81.029 | queen      | 62.158 | ten        | 70.954 |
      OrderedDict([('bbox', {'AP': 69.20630335401063, 'AP50': 82.57996307033767, 'AP75': 82.11883463076639, 'APs': nan, 'APm': 68.13414793860338, 'APl': 70.11922372310204, 'AP-ace': 83.41881645075392, 'AP-jack': 53.23562401798234, 'AP-king': 64.44154752427788, 'AP-nine': 81.02925635716204, 'AP-queen': 62.15809209588659, 'AP-ten': 70.95448367800098})])
      ```
