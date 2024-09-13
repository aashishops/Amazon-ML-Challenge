import string
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model
import pandas as pd
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def demo(opt):
    """Open CSV file where predicted words will be saved."""
    data = pd.read_csv('E:/College/Hackathon/Amazon ML Challenge/CRAFT/Amazon-ML-Challenge/CRAFT-pytorch/data.csv')

    """ Model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('Model input parameters:', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # Load model
    print('Loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # Prepare data
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # Prepare CSV for saving results
    result_csv_path = 'predicted_words.csv'
    if not os.path.exists(result_csv_path):
        result_df = pd.DataFrame(columns=['image_name', 'predicted_word', 'confidence_score'])
    else:
        result_df = pd.read_csv(result_csv_path)

    # Predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index.data, preds_size.data)
            else:
                preds = model(image, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t {"predicted_labels":25s}\t confidence score'
            print(f'{dashed_line}\n{head}\n{dashed_line}')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                start = 'E:/College/Hackathon/Amazon ML Challenge/CRAFT/Amazon-ML-Challenge/Crop Words'  # Path to cropped words
                path = os.path.relpath(img_name, start)
                folder = os.path.dirname(path)
                image_name = os.path.basename(path)
                file_name = '_'.join(image_name.split('_')[:-8])
                txt_file = os.path.join(start, folder, file_name)

                # Save recognized words to a .txt file
                txt_file_path = f'{txt_file}_log_demo_result_vgg.txt'
                with open(txt_file_path, 'a') as log:
                    if 'Attn' in opt.Prediction:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]
                        pred_max_prob = pred_max_prob[:pred_EOS]

                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                    log.write(f'{image_name:25s}\t {pred:25s}\t {confidence_score:0.4f}\n')

                    # Print to console
                    print(f'{image_name:25s}\t {pred:25s}\t {confidence_score:0.4f}')

                # Save results to CSV
                result_df = pd.concat([result_df, pd.DataFrame({
                    'image_name': [image_name],
                    'predicted_word': [pred],
                    'confidence_score': [confidence_score.item()]
                })], ignore_index=True)

    # Save the CSV with the results
    result_df.to_csv(result_csv_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    if opt.sensitive:
        opt.character = string.printable[:-6]

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    demo(opt)
