import torch
from transformers import AutoformerForPrediction, AutoformerConfig
from tqdm.auto import tqdm
import numpy as np

class AutoformerModel:
    def __init__(self, model_name, past_length, future_length, lr = 1e-4, weight_decay = 1e-5, num_epochs = 10):
        self.past_length = past_length
        self.future_length = future_length
        self.config = AutoformerConfig.from_pretrained(model_name)
        self.config.context_length = self.past_length - max(self.config.lags_sequence)
        self.config.prediction_length = self.future_length
        
        self.model = AutoformerForPrediction.from_pretrained(
            model_name,
            config = self.config,
            ignore_mismatched_sizes=True
        )
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2, verbose=True)

        
    def trainLoop(self,train_loader,test_loader):
        num_training_steps = self.num_epochs * len(train_loader)
        progress_bar = tqdm(range(num_training_steps))
        
        self.model.train()
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            running_loss = 0.0
            for step, batch in enumerate(train_loader):
                input_batch = { k : v for k,v in batch.items() }
                outputs = self.model(
                    past_values = input_batch['past_values'],
                    past_time_features = input_batch['past_time_features'],
                    past_observed_mask = input_batch['past_observed_mask'],
                    static_categorical_features = input_batch['static_categorical_features'],
                    future_values = input_batch['future_values'],
                    future_time_features = input_batch['future_time_features'])

                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                running_loss += loss.item()
                epoch_loss += loss.item()
                progress_bar.update(1)
                
                if (step + 1) % 10 == 0:
                    avg_loss = running_loss / 10
                    tqdm.write(f"Epoch [{epoch + 1}/{self.num_epochs}], Step[{step + 1}/{len(train_loader)}], Loss: {avg_loss}")
                    running_loss = 0.0
            
            val_avg_loss = self.evaluateLoop(test_loader)
            self.lr_scheduler.step(val_avg_loss)
            
            epoch_avg_loss = epoch_loss / len(train_loader)
            tqdm.write(f"Epoch [{epoch + 1}/{self.num_epochs}] completed, Training Loss: {epoch_avg_loss}, Validation Loss: {val_avg_loss}")
        
        progress_bar.close()
        
    def evaluateLoop(self,test_loader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                input_batch = { k : v for k, v in batch.items()}
                outputs = self.model(
                    past_values = input_batch['past_values'],
                    past_time_features = input_batch['past_time_features'],
                    past_observed_mask = input_batch['past_observed_mask'],
                    static_categorical_features = input_batch['static_categorical_features'],
                    future_values = input_batch['future_values'],
                    future_time_features = input_batch['future_time_features'])
                
                loss = outputs.loss
                val_loss += loss.item()
        return val_loss / len(test_loader)
    
    def inference(self,test_batch,sequence_index):
        past_vals = []
        future_vals = []
        past_dates = []
        future_dates = []
        sums_pred = []
        sums_target = []
        
        input_batch = { k : v for k, v in test_batch.items() }
        test_output = self.model.generate(
            past_values = input_batch['past_values'],
            past_time_features = input_batch['past_time_features'],
            past_observed_mask = input_batch['past_observed_mask'],
            static_categorical_features = input_batch['static_categorical_features'],
            future_time_features = input_batch['future_time_features'],
        )
        
        past_vals.append(input_batch['past_values'])
        future_vals.append(input_batch['future_values'])
        past_dates.append(input_batch['dates_past'])
        future_dates.append(input_batch['dates_future'])
        
        predictions_85th = torch.quantile(test_output.sequences, 0.85, dim=1)
        predictions_75th = torch.quantile(test_output.sequences, 0.75, dim=1)
        predictions_65th = torch.quantile(test_output.sequences, 0.65, dim=1)
        predictions_mean = test_output.sequences.mean(dim=1)
        
        predictions_mean_res = predictions_mean[sequence_index].numpy()
        predictions_65_res = predictions_65th[sequence_index].numpy()
        predictions_75_res = predictions_75th[sequence_index].numpy()
        predictions_85_res = predictions_85th[sequence_index].numpy()
        
        results = { "Mean" : predictions_mean_res, "65th" : predictions_65_res, "75th" : predictions_75_res, "85th" : predictions_85_res }
        return results
    
    def predictionProcessing(self,income,results):
        # savings potential at each prediction level
        savings_potential_object = { key : income - np.cumsum(val)[-1] for key,val in results.items() }
        return savings_potential_object