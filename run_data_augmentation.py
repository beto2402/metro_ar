from data_augmentation import augment_dataset


input_folder = 'datasets/mios/resized'
output_base_folder = 'datasets/mios/augs'


augment_dataset(input_folder, output_base_folder)

