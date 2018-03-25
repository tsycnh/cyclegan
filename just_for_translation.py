from cyclegan_plates import CycleGAN

gan = CycleGAN()

gan.translate('good_model/model_epoch_1800.h5',total_num=1500)