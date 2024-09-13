python3 CRAFT-pytorch/test.py --trained_model=craft_mlt_25k.pth --test_folder=small_sample --cuda=False

python3 deep-text-recognition-benchmark/demo.py --Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction CTC --image_folder 'deep-text-recognition-benchmark/demo_image' --saved_model 'None-VGG-BiLSTM-CTC (will be deprecated).pth'