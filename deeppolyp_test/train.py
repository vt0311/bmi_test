#!/usr/bin/env python
# Import python libraries
import argparse, os, sys, shutil, time, pickle
from distutils.dir_util import copy_tree
# import seaborn as sns
from getpass import getuser
import numpy as np

# Import keras libraries
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop
from keras.utils.vis_utils import plot_model

# Import project libraries
from models.fcn8 import build_fcn8
from metrics.metrics import cce_flatt
from callbacks.callbacks import Evaluate_model, compute_metrics
from tools.loader_sem_seg import ImageDataGenerator
from tools.logger import Logger
from tools.plot_history import plot_history
from tools.compute_mean_std import compute_mean_std
from tools.compute_class_balance import compute_class_balance
sys.setrecursionlimit(99999)


# Train the network
# 신경망 훈련시키기
def train(dataset, model_name, learning_rate, weight_decay,
          num_epochs, max_patience, batch_size, optimizer,
          savepath, train_path, valid_path, test_path,
          crop_size=(224, 224), in_shape=(3, None, None),
          n_classes=5, gtSet=None, void_class=[4], w_balance=None,
          weights_file=False, show_model=False,
          plot_hist=True, train_model=True):

    # Remove void classes from number of classes
    n_classes = n_classes - len(void_class)

    # Mask folder (For different polyp groundtruths)
    if gtSet is not None:
        mask_floder = 'masks' + str(gtSet)
    else:
        mask_floder = 'masks'

    # TODO: Get the number of images directly from data loader
    n_images_train = 30  # 547
    n_images_val = 20  # 183
    n_images_test = 20  # 182

    # Normalization mean and std computed on training set for RGB pixel values
    # 정규화 표준화 RGB 픽셀값
    print ('\n > Computing mean and std for normalization...')
    if False:
        rgb_mean, rgb_std = compute_mean_std(os.path.join(train_path, 'images'),
                                             os.path.join(train_path, mask_floder),
                                             n_classes)
        rescale = None
    else:
        rgb_mean = None
        rgb_std = None
        rescale = 1/255.
    print ('Mean: ' + str(rgb_mean))
    print ('Std: ' + str(rgb_std))

    # Compute class balance weights
    if w_balance is not None:
        class_balance_weights = compute_class_balance(masks_path=train_path + mask_floder,
                                                      n_classes=n_classes,
                                                      method=w_balance,
                                                      void_labels=void_class
                                                      )
        print ('Class balance weights: ' + str(class_balance_weights))
    else:
        class_balance_weights = None

    # Build model
    print ('\n > Building model (' + model_name + ')...')
    if model_name == 'fcn8':
        model = build_fcn8(in_shape, l2_reg=weight_decay, nclasses=n_classes,
                           weights_file=weights_file, deconv='deconv')
        model.output
    else:
        raise ValueError('Unknown model')

    # Create the optimizer
    # 옵티마이저 만들기
    print ('\n > Creating optimizer ({}) with lr ({})...'.format(optimizer,
                                                                learning_rate))
    if optimizer == 'rmsprop':
        opt = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-8, clipnorm=10)
    else:
        raise ValueError('Unknown optimizer')

    # Compile model
    print ('\n > Compiling model...')
    model.compile(loss=cce_flatt(void_class, class_balance_weights),
                  optimizer=opt)

    # Show model structure
    # 모델 구조 보기
    if show_model:
        model.summary()
        plot(model, to_file=savepath+'model.png')

    # Create the data generators
    # 데이터 생성자 만들기
    print ('\n > Reading training set...')
    dg_tr = ImageDataGenerator(crop_size=crop_size,  # Crop the image to a fixed size
                               featurewise_center=False,  # Substract mean - dataset
                               samplewise_center=False,  # Substract mean - sample
                               featurewise_std_normalization=False,  # Divide std - dataset
                               samplewise_std_normalization=False,  # Divide std - sample
                               rgb_mean=rgb_mean,
                               rgb_std=rgb_std,
                               gcn=False,  # Global contrast normalization
                               zca_whitening=False,  # Apply ZCA whitening
                               rotation_range=180,  # Rnd rotation degrees 0-180
                               width_shift_range=0.0,  # Rnd horizontal shift
                               height_shift_range=0.0,  # Rnd vertical shift
                               shear_range=0.5,  # 0.5,  # Shear in radians
                               zoom_range=0.1,  # Zoom
                               channel_shift_range=0.,  # Channel shifts
                               fill_mode='constant',  # Fill mode
                               cval=0.,  # Void image value
                               void_label=void_class[0],  # Void class value
                               horizontal_flip=True,  # Rnd horizontal flip
                               vertical_flip=True,  # Rnd vertical flip
                               rescale=rescale,  # Rescaling factor
                               spline_warp=False,  # Enable elastic deformation
                               warp_sigma=10,  # Elastic deformation sigma
                               warp_grid_size=3  # Elastic deformation gridSize
                               )
    # 훈련시킬 생성사진
    train_gen = dg_tr.flow_from_directory(train_path + 'images',
                                          batch_size=batch_size,
                                          gt_directory=train_path + mask_floder,
                                          target_size=crop_size,
                                          class_mode='seg_map',
                                          classes=n_classes,
                                          # save_to_dir=savepath,  # Save DA
                                          save_prefix='data_augmentation',
                                          save_format='png')

    print ('\n > Reading validation set...')
    dg_va = ImageDataGenerator(rgb_mean=rgb_mean, rgb_std=rgb_std, rescale=rescale)
    valid_gen = dg_va.flow_from_directory(valid_path + 'images',
                                          batch_size=1,
                                          gt_directory=valid_path + mask_floder,
                                          target_size=None,
                                          class_mode='seg_map',
                                          classes=n_classes)

    print ('\n > Reading testing set...')
    dg_ts = ImageDataGenerator(rgb_mean=rgb_mean, rgb_std=rgb_std, rescale=rescale)
    test_gen = dg_ts.flow_from_directory(test_path + 'images',
                                         batch_size=1,
                                         gt_directory=test_path + mask_floder,
                                         target_size=None,
                                         class_mode='seg_map',
                                         classes=n_classes,
                                         shuffle=False)

    # Define the jaccard validation callback
    # jaccard 유효성 콜백 정의하기
    eval_model = Evaluate_model(n_classes=n_classes,
                                void_label=void_class[0],
                                save_path=savepath,
                                valid_gen=valid_gen,
                                valid_epoch_length=n_images_val,
                                valid_metrics=['val_loss',
                                               'val_jaccard',
                                               'val_acc',
                                               'val_jaccard_perclass'])

    # Define early stopping callbacks
    early_stop_jac = EarlyStopping(monitor='val_jaccard', mode='max',
                                   patience=max_patience, verbose=0)
    early_stop_jac_class = []
    for i in range(n_classes):
        early_stop_jac_class += [EarlyStopping(monitor=str(i)+'_val_jacc_percl',
                                               mode='max', patience=max_patience,
                                               verbose=0)]

    # Define model saving callbacks
    # 콜백 저장 모델 정의하기
    checkp_jac = ModelCheckpoint(filepath=savepath+"weights.hdf5",
                                 verbose=0, monitor='val_jaccard',
                                 mode='max', save_best_only=True,
                                 save_weights_only=True)
    checkp_jac_class = []
    for i in range(n_classes):
        checkp_jac_class += [ModelCheckpoint(filepath=savepath+"weights"+str(i)+".hdf5",
                                             verbose=0,
                                             monitor=str(i)+'_val_jacc_percl',
                                             mode='max', save_best_only=True,
                                             save_weights_only=True)]

    # Train the model
    # 모델 훈련시키기
    if (train_model):
        print('\n > Training the model...')
        cb = [eval_model, early_stop_jac, checkp_jac] + checkp_jac_class
        hist = model.fit_generator(train_gen, samples_per_epoch=n_images_train,
                                nb_epoch=num_epochs,
                                callbacks=cb)

    # Compute test metrics
    print('\n > Testing the model...')
    model.load_weights(savepath + "weights.hdf5")
    color_map = [
        (255/255., 0, 0),                   # Background
        (192/255., 192/255., 128/255.),     # Polyp
        (128/255., 64/255., 128/255.),      # Lumen
        (0, 0, 255/255.),                   # Specularity
        (0, 255/255., 0),         #
        (192/255., 128/255., 128/255.),     #
        (64/255., 64/255., 128/255.),       #
    ]
    test_metrics = compute_metrics(model, test_gen, n_images_test, n_classes,
                                   metrics=['test_loss',
                                            'test_jaccard',
                                            'test_acc',
                                            'test_jaccard_perclass'],
                                   color_map=color_map, tag="test",
                                   void_label=void_class[0],
                                   out_images_folder=savepath,
                                   epoch=0,
                                   save_all_images=True,
                                   useCRF=False)
    for k in sorted(test_metrics.keys()):
        print('{}: {}'.format(k, test_metrics[k]))

    if (train_model):
        # Save the results
        print ("\n > Saving history...")
        with open(savepath + "history.pickle", 'w') as f:
            pickle.dump([hist.history, test_metrics], f)

        # Load the results
        print ("\n > Loading history...")
        with open(savepath + "history.pickle") as f:
            history, test_metrics = pickle.load(f)
            # print (str(test_metrics))

        # Show the trained model history
        if plot_hist:
            print('\n > Show the trained model history...')
            plot_history(history, savepath, n_classes)


# Main function
def main():
    # Get parameters from file parser
    # 파일 parser 파라미터 가져오기
    parser = argparse.ArgumentParser(description='DeepPolyp model training')
    parser.add_argument('-dataset', default='polyps', help='Dataset')
    parser.add_argument('-model_name', default='fcn8', help='Model')
    parser.add_argument('-model_file', default='weights.hdf5',
                        help='Model file')
    parser.add_argument('-load_pretrained', default=False,
                        help='Load pretrained model from model file')
    parser.add_argument('-learning_rate', default=0.0001, help='Learning Rate')
    parser.add_argument('-weight_decay', default=0.,
                        help='regularization constant')
    parser.add_argument('--num_epochs', '-ne', type=int, default=1000,
                        help='Optional. Int to indicate the max'
                        'number of epochs.')
    parser.add_argument('-max_patience', type=int, default=100,
                        help='Max patience (early stopping)')
    parser.add_argument('-batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--optimizer', '-opt', default='rmsprop',
                        help='Optimizer')
    args = parser.parse_args()

    # Parameters
    nClasses = 3
    w_balance = None  # 'median_freq_cost'
    crop_size = (224, 224)  # (288, 384)

    # Experiment name
    # 실험명
    experiment_name = "tmp"

    # Define paths according to user
    # 사용자에 따른 경로 설정
    usr = getuser()
    if usr == "michal":
        # Michal paths
        savepath = '/home/michal/' + experiment_name + '/'
        dataset_path = '/home/michal/polyps/polyps_split2/CVC-912/'
        train_path = dataset_path + 'train/'
        valid_path = dataset_path + 'valid/'
        test_path = dataset_path + 'test/'
    elif usr == 'vazquezd' or usr == 'romerosa':
        shared_dataset_path = '/data/lisa/exp/vazquezd/datasets/polyps_split7/'
        dataset_path = '/Tmp/'+usr+'/datasets/polyps_split7/'
        # Copy the data to the local path if not existing
        if not os.path.exists(dataset_path):
            print('The local path {} does not exist. Copying '
                  'dataset...'.format(dataset_path))
            shutil.copytree(shared_dataset_path, dataset_path)
            print('Done.')

        savepath = '/Tmp/'+usr+'/results/deepPolyp/fcn8/paper/'+experiment_name+'/'
        final_savepath = '/data/lisatmp4/' + usr + '/results/deepPolyp/fcn8/' + experiment_name + '/'
        train_path = dataset_path + 'train/'
        valid_path = dataset_path + 'valid/'
        test_path = dataset_path + 'test/'

    elif usr == 'dvazquez':
        shared_dataset_path = '/home/'+usr+'/Datasets/Polyps/'
        dataset_path = '/home/'+usr+'/Datasets/Polyps/'
        # Copy the data to the local path if not existing
        if not os.path.exists(dataset_path):
            print('The local path {} does not exist. Copying '
                  'dataset...'.format(dataset_path))
            shutil.copytree(shared_dataset_path, dataset_path)
            print('Done.')

        savepath = '/home/'+usr+'/Experiments/deepPolyp/'+experiment_name+'/'
        final_savepath = '/home/'+usr+'/Experiments/deepPolyp/'+experiment_name+'/'
        train_path = dataset_path + 'train/'
        valid_path = dataset_path + 'valid/'
        test_path = dataset_path + 'test/'
    
    elif usr == 'acorn':
        #shared_dataset_path = '/home/'+usr+'/Datasets/Polyps/'
        shared_dataset_path = 'C:\CVC-300\gtpolyp'
        
        #dataset_path = '/home/'+usr+'/Datasets/Polyps/'
        #dataset_path = '/home/'+usr+'/Datasets/Polyps/'
        dataset_path = 'C:\CVC-300\gtpolyp'
        
        # Copy the data to the local path if not existing
        if not os.path.exists(dataset_path):
            print('The local path {} does not exist. Copying '
                  'dataset...'.format(dataset_path))
            shutil.copytree(shared_dataset_path, dataset_path)
            print('Done.')

        #savepath = '/home/'+usr+'/Experiments/deepPolyp/'+experiment_name+'/'
        savepath = 'C:\CVC-300\deepPolyp'
        #final_savepath = '/home/'+usr+'/Experiments/deepPolyp/'+experiment_name+'/'
        final_savepath ='C:\CVC-300\deepPolyp'
        train_path = dataset_path + '\train'
        valid_path = dataset_path + '\valid'
        test_path = dataset_path + '\test'    

    else:
        raise ValueError('User unknown, please add your own paths!')

    # Create output folders
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # Enable log file
    sys.stdout = Logger(savepath + "logfile.log")
    print (' ---> Experiment: ' + experiment_name + ' <---')

    # Train the network
    train(dataset=args.dataset,
          model_name=args.model_name,
          learning_rate=float(args.learning_rate),
          weight_decay=float(args.weight_decay),
          num_epochs=int(args.num_epochs),
          max_patience=int(args.max_patience),
          batch_size=int(args.batch_size),
          optimizer=args.optimizer,
          savepath=savepath,
          show_model=False,
          train_path=train_path, valid_path=valid_path, test_path=test_path,
          crop_size=crop_size,
          in_shape=(3, None, None),
          n_classes=nClasses+1,
          gtSet=nClasses,
          void_class=[nClasses],
          w_balance=w_balance,
          weights_file=savepath+args.model_file if bool(args.load_pretrained) else False,
          train_model=True,
          plot_hist=False
          )
    print (' ---> Experiment: ' + experiment_name + ' <---')

    print('Copying model and other training files to {}'.format(final_savepath))
    start = time.time()
    copy_tree(savepath, final_savepath)
    open(os.path.join(final_savepath, 'lock'), 'w').close()
    print ('Copy time: ' + str(time.time()-start))

# Entry point of the script
if __name__ == "__main__":
    main()
