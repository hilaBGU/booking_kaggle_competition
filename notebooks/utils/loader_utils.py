from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class UsersReviewsTrainDataset(Dataset):
    def __init__(self, datapath, review_tokenizer, features_tokenizer, batch_size):
        self.data = pd.read_csv(datapath)
        self.acc_ids = self.data['accommodation_id'].value_counts().sort_values(ascending=False).index.tolist()[:2500]
        self.review_tokenizer = review_tokenizer
        self.features_tokenizer = features_tokenizer
        self.batch_size = batch_size

    def __len__(self):
        return len(self.acc_ids)

    def __getitem__(self, idx):
        acc_id = self.acc_ids[idx]
        new_indices = self.data[self.data['accommodation_id'] == acc_id].index.tolist()
        all_samples = self.data.iloc[new_indices]
        all_samples = all_samples.sample(n=min(self.batch_size, len(all_samples)), replace=False).reset_index(drop=True)
        if len(all_samples) < self.batch_size:
            resample = self.data[self.data['accommodation_id'] != acc_id].sample(n=self.batch_size-len(all_samples), replace=False).reset_index(drop=True)
            all_samples = pd.concat([all_samples, resample], axis=0)
        all_text = self.review_tokenizer(all_samples['all_text'].values.tolist(),
                                         padding='max_length',
                                         truncation=True,
                                         return_tensors='pt',
                                         max_length=self.review_tokenizer.model_max_length)
        all_text.data['input_ids'] = all_text.data['input_ids'].squeeze()
        all_text.data['attention_mask'] = all_text.data['attention_mask'].squeeze()
        all_text.data['token_type_ids'] = all_text.data['token_type_ids'].squeeze()

        # features = sample.drop(['all_text']).values.astype(np.float32)
        features = self.features_tokenizer(all_samples['features_text'].values.tolist(),
                                           padding='max_length',
                                           truncation=True,
                                           return_tensors='pt',
                                           max_length=self.features_tokenizer.model_max_length)
        features.data['input_ids'] = features.data['input_ids'].squeeze()
        features.data['attention_mask'] = features.data['attention_mask'].squeeze()
        features.data['token_type_ids'] = features.data['token_type_ids'].squeeze()
        return all_text, features


class UsersReviewsValDataset(Dataset):
    def __init__(self, datapath, review_tokenizer, features_tokenizer, frac, batch_size):
        data = pd.read_csv(datapath)
        self.data = data.sample(frac=frac).reset_index(drop=True)
        self.review_tokenizer = review_tokenizer
        self.features_tokenizer = features_tokenizer
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        all_text = self.review_tokenizer(sample['all_text'],
                                         padding='max_length',
                                         truncation=True,
                                         return_tensors='pt',
                                         max_length=self.review_tokenizer.model_max_length)
        all_text.data['input_ids'] = all_text.data['input_ids'].squeeze()
        all_text.data['attention_mask'] = all_text.data['attention_mask'].squeeze()
        all_text.data['token_type_ids'] = all_text.data['token_type_ids'].squeeze()

        # features = sample.drop(['all_text']).values.astype(np.float32)
        features = self.features_tokenizer(sample['features_text'],
                                           padding='max_length',
                                           truncation=True,
                                           return_tensors='pt',
                                           max_length=self.features_tokenizer.model_max_length)
        features.data['input_ids'] = features.data['input_ids'].squeeze()
        features.data['attention_mask'] = features.data['attention_mask'].squeeze()
        features.data['token_type_ids'] = features.data['token_type_ids'].squeeze()
        return all_text, features
