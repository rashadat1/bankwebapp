from huggingface_hub import hf_hub_download
import torch
from transformers import AutoformerForPrediction, AutoformerConfig
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

file = hf_hub_download(repo_id="hf-internal-testing/tourism-monthly-batch",filename="train-batch.pt", repo_type="dataset")
# batch has type dict
batch = torch.load(file)

model = AutoformerForPrediction.from_pretrained("huggingface/autoformer-tourism-monthly")

print(f"past_values shape: {batch['past_values'].shape}")
print(f"past_time_features shape: {batch['past_time_features'].shape}")
print(f"future_values shape: {batch['future_values'].shape}")
print(f"future_time_features shape: {batch['future_time_features'].shape}")
print(f"past_observed_mask shape: {batch['past_observed_mask'].shape}")
print(f"static_categorical_features shape: {batch['static_categorical_features'].shape}")

device = torch.device('mps' if torch.backends.mps.is_available else 'cpu')

past_values = batch['past_values'].to(device)
past_time_features = batch['past_time_features'].to(device)
future_values = batch['future_values'].to(device)
future_time_features = batch['future_time_features'].to(device)
past_observed_mask = batch['past_observed_mask'].to(device)
static_categorical_features = batch['static_categorical_features'].to(device)

model.to(device)

def forward_with_cpu_fallback(model, device, **inputs):
    # Move model to CPU
    model_cpu = model.to('cpu')
    # Move inputs to CPU
    inputs_on_cpu = {k: v.to('cpu') for k, v in inputs.items()}
    # Perform forward pass on CPU
    with torch.no_grad():
        outputs = model_cpu(**inputs_on_cpu)
    # Move outputs back to the original device
    outputs_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}
    return outputs_on_device

def generate_with_cpu_fallback(model, device, **inputs):
    # Move model to CPU
    model_cpu = model.to('cpu')
    # Move inputs to CPU
    inputs_on_cpu = {k: v.to('cpu') for k, v in inputs.items()}
    # Perform generate pass on CPU
    with torch.no_grad():
        outputs = model_cpu.generate(**inputs_on_cpu)
    # Move outputs back to the original device
    if isinstance(outputs, tuple):
        outputs_on_device = tuple([o.to(device) for o in outputs])
    elif isinstance(outputs, torch.Tensor):
        outputs_on_device = outputs.to(device)
    else:
        outputs_on_device = outputs
    return outputs_on_device


# Perform a forward pass
# for training purposes
outputs = forward_with_cpu_fallback(
    model,
    device,
    past_values=past_values,
    past_time_features=past_time_features,
    future_values=future_values,
    future_time_features=future_time_features,
    past_observed_mask=past_observed_mask,
    static_categorical_features=static_categorical_features
)

# 
predictions = generate_with_cpu_fallback(
    model,
    device,
    past_values=past_values,
    past_time_features=past_time_features,
    past_observed_mask=past_observed_mask,
    static_categorical_features=static_categorical_features,
    future_time_features=future_time_features)

print()
# the generated predictions are 100 sequences for each of the 64 output sequences. Each sequence has length 24
# we average along the 2nd dimension (get the mean of the 100 sequences for each time step in the output) leaving us 
# with shape [64,24]
print(predictions.sequences.shape)
print(predictions.sequences.mean(dim=1))
print(predictions.sequences.mean(dim=1).shape)


'''
It looks like the batch size is 64 (we are predicting 64 future time 
series using 64 input time series). Then the length of the input sequences is 
61 (so we have 64 length 61 sequences in past_values). In future values we 
are predicting 64 sequences of length 24. 

Past values in this dataset has shape [64,61]
Past time features [64,61,2]
Future values [64,24]
and Future time features [64,24,2]
'''