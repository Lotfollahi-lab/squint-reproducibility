"""
MultiTimepointDataModule: A custom datamodule that can handle multiple timepoints and combine them into batches.

This module allows you to:
1. Create data loaders for different timepoints
2. Set batch size to 1 for each individual config of a timepoint (one source node and its neighbors)
3. Create a combined dataloader that returns batches of size x containing x calls from each timepoint stratified by patient
Todo: Add shuffle=True to the dataloader for training and validation in a correct way after talking to @Arpit (currently no shuffling is done).
"""

import random
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Any, Iterator
import pytorch_lightning as pl
from torch_geometric.data import Data, Batch
from torch_geometric.loader.dataloader import Collater

from vqniche.initializers.initialize import initialize_dataset_blob, initialize_databatch, initialize_datamodule


def multi_timepoint_collate_fn(batch):
    """
    Custom collate function for MultiTimepointDataset.
    """
    # Use PyTorch Geometric's collate function to combine all samples
    collater = Collater(None, None)
    return collater(batch)


class MultiTimepointDataset(Dataset):
    """
    A dataset that combines multiple timepoints and returns one sample from each timepoint.
    """
    
    def __init__(self, config: Dict, time_points_dict: Dict, patients: Dict):
        """
        Initialize the MultiTimepointDataset.
        
        Parameters:
        -----------
        - config: Dict
            Configuration dictionary
        - time_points_dict: Dict
            Dictionary of batches in each time point
        - patients: Dict
            Dictionary of patients with time point
        - batch_size: int
            The size of each batch
        """
        self.config = config
        self.time_points_dict = time_points_dict
        self.patients = patients
        self.dataloaders = {}
        self.iterators = {}
        
        self.anchor_time_point = 't3'
        
        # Initialize dataset blob
        dataset_blob = initialize_dataset_blob(config)
        
        all_batch_idx = [x for lst in patients.values() for x in lst]
        self.batch_idx_patient_map = {adata_batch_idx:patient_id for patient_id, lst in patients.items() for adata_batch_idx in lst}
        self.t_anchor_batch_idx_list = [adata_batch_idx for adata_batch_idx in all_batch_idx if adata_batch_idx in time_points_dict[self.anchor_time_point]]
        self.t_anchor_batch_idx_copy = self.t_anchor_batch_idx_list.copy()
        
        for adata_batch_idx in all_batch_idx:
            
            # Set batch_size to 1 and adata_batch_idx to the current time point
            config['datamodule']['loader_params']['batch_size'] = 1
            # config['datamodule']['loader_params']['disjoint'] = True
            config['dataset']['adata_batch_idx'] = adata_batch_idx
            # config['datamodule']['loader_params']['shuffle'] = True # TODO: shuffle in training, not tested yet
            
            # Initialize data batch
            data_batch = initialize_databatch(
                config=config,
                dataset_blob=dataset_blob,
            )
            
            # Initialize datamodule
            datamodule = initialize_datamodule(
                config=config,
                data=data_batch,
            )
            
            # Create iterators for each dataloader
            dataloader = datamodule.predict_dataloader()
            self.dataloaders[adata_batch_idx] = dataloader
            self.iterators[adata_batch_idx] = iter(dataloader)
    
    def __len__(self):
        """Return the total number of batches."""
        # Calculate the number of batches across time point 3
        num_t_anchor_batches = sum([len(self.dataloaders[dataloader_key]) for dataloader_key in self.t_anchor_batch_idx_list])
        return num_t_anchor_batches
    
    def get_random_patient_idx(self):
        if len(self.t_anchor_batch_idx_copy) == 0:
            for t_anchor_batch_idx in self.t_anchor_batch_idx_list:
                self.iterators[t_anchor_batch_idx] = iter(self.dataloaders[t_anchor_batch_idx])
                self.t_anchor_batch_idx_copy.append(t_anchor_batch_idx)
        selected_t_anchor_batch_idx = random.choice(self.t_anchor_batch_idx_copy)
        patient_idx = self.batch_idx_patient_map[selected_t_anchor_batch_idx]
        samples = {}
        try:
            sample = next(self.iterators[selected_t_anchor_batch_idx])
            samples[self.anchor_time_point] = sample
        except StopIteration:
            # Remove batch index if its iterator is exhausted
            self.t_anchor_batch_idx_copy.remove(selected_t_anchor_batch_idx)
            return self.get_random_patient_idx()
        
        patient_set = set(self.patients[patient_idx])
        selected_patient_idx = {tp: next(iter(patient_set & set(self.time_points_dict[tp]))) for tp in self.time_points_dict}
        return samples, selected_patient_idx
    
    def __getitem__(self, idx):
        """
        Get a one sample from each time point.
        """
        
        samples, selected_patient_idx = self.get_random_patient_idx()
        
        for time_point, adata_batch_idx in selected_patient_idx.items():
            # Skip anchor time point
            if time_point == self.anchor_time_point:
                continue
            # Get a sample from this time point
            time_point_sample = None
            try:
                sample = next(self.iterators[adata_batch_idx])
                time_point_sample = sample
            except StopIteration:
                # Reset iterator if exhausted
                self.iterators[adata_batch_idx] = iter(self.dataloaders[adata_batch_idx])
                sample = next(self.iterators[adata_batch_idx])
                time_point_sample = sample
            
            samples[time_point] = time_point_sample
        
        # Return the samples as a dict
        return samples


class MultiTimepointDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule that handles multiple timepoints and combines them into batches.
    """
    
    def __init__(self, config: Dict, time_points_dict: Dict, train_patients: Dict, test_patients: Dict, batch_size: int = 4):
        """
        Initialize the MultiTimepointDataModule.
        
        Parameters:
        -----------
        - config: Dict
            Configuration dictionary
        - time_points_dict: Dict
            Dictionary of batches in each time point
        - train_patients: Dict
            Dictionary of train patients with time point
        - test_patients: Dict
            Dictionary of test patients with time point
        - batch_size: int
            The size of each batch
        """
        super().__init__()
        self.config = config
        self.time_points_dict = time_points_dict
        self.batch_size = batch_size
        self.train_patients = train_patients
        self.test_patients = test_patients
    
    def train_dataloader(self):
        self.dataset = MultiTimepointDataset(self.config, self.time_points_dict, self.train_patients)
        """Return the training dataloader."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=multi_timepoint_collate_fn,
        )
    
    def val_dataloader(self):
        self.dataset = MultiTimepointDataset(self.config, self.time_points_dict, self.test_patients)
        """Return the validation dataloader."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=multi_timepoint_collate_fn,
        )
    
    def test_dataloader(self):
        self.dataset = MultiTimepointDataset(self.config, self.time_points_dict, self.test_patients)
        """Return the test dataloader."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=multi_timepoint_collate_fn,
        )
    
    def predict_dataloader(self):
        """Return the prediction dataloader."""
        self.dataset = MultiTimepointDataset(self.config, self.time_points_dict, self.test_patients)
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=multi_timepoint_collate_fn,
        )


def create_multi_timepoint_datamodule(config: Dict, time_points_dict: Dict, train_patients: Dict, test_patients: Dict, batch_size: int = 4) -> MultiTimepointDataModule:
    """
    Create a MultiTimepointDataModule for different timepoints.
    
    Parameters:
    -----------
    - config: Dict
        Configuration dictionary
    - time_points_dict: Dict
        Dictionary of batches in each time point
    - train_patients: Dict
        Dictionary of train patients with time point
    - test_patients: Dict
        Dictionary of test patients with time point
    - batch_size: int
        The size of each batch
    
    Returns:
    --------
    - MultiTimepointDataModule
        A configured MultiTimepointDataModule instance
    """
    return MultiTimepointDataModule(config, time_points_dict, train_patients, test_patients, batch_size)

