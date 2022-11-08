import os, sys
import glob
import time
import random
import pickle
import numpy as np
import types
from PIL import Image
from sklearn.model_selection import train_test_split

import torch as tc
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets

import torchvision.transforms as transforms


class FEMNISTParticipantDataset: 
    def __init__(self, x, y, participant, transform=None, image_shape=(28,28), classes=None, class_to_idx=None):
        self.participant = participant
        self.x = x
        self.y = y
        self.image_shape = image_shape
        self.transform = transforms.Compose(transform)
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.y)

    
    def __getitem__(self, index):
        sample, target = self.x[index], self.y[index]
        if sample.shape != (28, 28): 
            if len(sample) == 28*28:
                sample = np.reshape(sample, self.image_shape)
            else: 
                print(f"Warning: Sample has shape {sample.shape}")
                return None

        sample = Image.fromarray(sample)

        if self.transform is not None:
            sample = self.transform(sample)
            # sample = self.transform((sample, target))

        return sample, target
        
class FEMNIST:
    def __init__(self, args):

        # root = os.path.join('data', args.src.lower())
        
        # shortened path
        # root = "/home/aheyler/PAC-pred-set/data/femnist"

        # full path
        root = "/home/aheyler/PAC-pred-set/data/femnist"

        ## default transforms
        tforms_dft = [
            transforms.Grayscale(3), # duplicates channels x3 since ResNet assumes 3 channels
            transforms.ToTensor(), # if float, as is, but if image from 0 to 255 -- want 0 to 1
            # ToTensor will normalize
        ]

        self.train_length = 0
        self.val_length = 0
        self.test_length = 0

        if os.path.exists(root):
            
            train_loaders = []
            val_loaders = []
            test_loaders = []
            
            if args.preselected_participants: 
                sampled_participants = np.load(args.preselected_participants)
                print("Loaded preselected participants:")
                print(sampled_participants)
            else: 
                sampled_participants = os.listdir(root)
                print("All participants initially loaded")
                np.save(f"/home/aheyler/PAC-pred-set/snapshots/{args.exp_name}/participants_arr", sampled_participants)
                
            if args.num_participants: 
                np.random.seed(42)
                sampled_participants = np.random.choice(sampled_participants, args.num_participants, replace=False)
            
            for participant in sampled_participants: 
                participant_folder = os.path.join(root, participant)
                train_image_path = os.path.join(participant_folder, "training_images.npy")
                train_label_path = os.path.join(participant_folder, "training_labels.npy")
                holdout_image_path = os.path.join(participant_folder, "test_images.npy")
                holdout_label_path = os.path.join(participant_folder, "test_labels.npy")

                if os.path.isfile(train_image_path) and os.path.isfile(train_label_path) \
                    and os.path.isfile(holdout_image_path) and os.path.isfile(holdout_label_path): 
                    x_train = np.load(train_image_path)
                    y_train = np.load(train_label_path)
                    
                    # EDIT LATER: FOR NOW, ONLY INCLUDE LARGER DATASETS
                    if len(y_train) > 200: 

                        _, dim2, dim3 = x_train.shape
                        if dim2 == 28 and dim3 == 28: 
                            holdout_images = np.load(holdout_image_path)
                            holdout_labels = np.load(holdout_label_path)

                            # Split into validation and test data
                            x_test, x_val, y_test, y_val = train_test_split(holdout_images, holdout_labels, test_size=0.5, random_state=42)

                            # Create datasets
                            participant_dataset_train = FEMNISTParticipantDataset(x_train, y_train, participant, transform=tforms_dft, image_shape=(28,28))
                            participant_dataset_val = FEMNISTParticipantDataset(x_val, y_val, participant, transform=tforms_dft, image_shape=(28,28))
                            participant_dataset_test = FEMNISTParticipantDataset(x_test, y_test, participant, transform=tforms_dft, image_shape=(28,28))

                            # Update lengths
                            self.train_length += len(participant_dataset_train)
                            self.val_length += len(participant_dataset_val)
                            self.test_length += len(participant_dataset_test)
                            
                            # Create dataloaders
                            participant_dataloader_train = DataLoader(participant_dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, drop_last=True)
                            participant_dataloader_val = DataLoader(participant_dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, drop_last=True)
                            participant_dataloader_test = DataLoader(participant_dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, drop_last=True)

                            # Append to list
                            train_loaders.append(participant_dataloader_train)
                            val_loaders.append(participant_dataloader_val)
                            test_loaders.append(participant_dataloader_test)

                else: 
                    print(f"{participant} has no data")

            # Set self
            self.train = train_loaders
            self.val = val_loaders
            self.test = test_loaders
            self.num_participants = len(train_loaders)

            print(f'#train = {self.train_length}, #val = {self.val_length}, #test = {self.test_length}')
            print(f"#num_participants = {len(train_loaders)}")
        else:
            return None
    

# Can run this script to see if the dataset construction works
if __name__ == '__main__':
    dsld = FEMNIST(types.SimpleNamespace(src='FEMNIST', batch_size=100, seed=0, n_workers=10))
    dsld.train_length




