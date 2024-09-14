python3 CRAFT-pytorch/test.py --trained_model=craft_mlt_25k.pth --test_folder=small_sample --cuda=False

For Link Refiner cmd prompt (Windows):

python CRAFT-pytorch\test.py --trained_model=craft_mlt_25k.pth --refiner_model=craft_refiner_CTW1500.pth --test_folder= small_sample --refine

For Deep Text REcognition prompt Command :(Windows)

python deep-text-recognition-benchmark/demo.py --Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction CTC --image_folder CropWords --saved_model None-VGG-BiLSTM-CTC.pth