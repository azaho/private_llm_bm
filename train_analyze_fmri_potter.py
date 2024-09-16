# %%
import scipy, scipy.io, scipy.signal, scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
import argparse
import os
import hashlib

parser = argparse.ArgumentParser(description='Process LLM and fMRI data')
parser.add_argument('--llm_model_index', type=int, default=1, help='Index of the LLM model to use')
parser.add_argument('--n_delay_embedding_llm', type=int, default=10, help='Number of delay embeddings for LLM')
parser.add_argument('--n_top_pc_llm', type=int, default=-1, help='Number of top principal components for LLM features. Set to -1 to skip PCA.')
parser.add_argument('--convolve_size', type=int, default=10, help='Size of convolution')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimization')
parser.add_argument('--random', type=str, default="X", help='Random string for setting seed')
args = parser.parse_args()

random_seed = int(hashlib.sha1(args.random.encode("utf-8")).hexdigest(),
                                         16) % 10 ** 8  # random initialization seed (for reproducibility)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# editable parameters
llm_model_index = args.llm_model_index
n_delay_embedding_llm = args.n_delay_embedding_llm
n_top_pc_llm = args.n_top_pc_llm
convolve_size = args.convolve_size
weight_decay = args.weight_decay

llm_model_name = ["GPT2_XL", "LLAMA_3.1_70b"][llm_model_index]
llm_model_activations_dir = ["data_gpt2_xl", "data_llama70b"][llm_model_index]
llm_layer_i = [24, 40][llm_model_index]
llm_d_model = [1600, 8192][llm_model_index]
llm_n_ctx = 1024

# Usage
n_delay_embedding_fmri = 0
n_top_pc_fmri = -1  # Skip PCA for fMRI signals
keep_dims_llm = True
keep_dims_fmri = False

num_epochs = 200
train_data_percentage = 0.7
test_data_percentage = 0.2
validation_data_percentage = 1 - train_data_percentage - test_data_percentage
replace_x_with_noise = False # for control testing
test_sequential = True  # New parameter

figure_prefix = f"figures/{llm_model_activations_dir}_nde{n_delay_embedding_llm}_npc{n_top_pc_llm}_cs{convolve_size}_wd{weight_decay:.3f}_r{args.random}"
os.makedirs('figures', exist_ok=True) # Create the 'figures' directory if it doesn't exist

# %%
fmri_num_subjects = 8
fmri_data_subjects = []
for i in range(fmri_num_subjects):
    fmri_data_subjects.append(scipy.io.loadmat(f'data_fmri_subjects/subject_{i+1}.mat', simplify_cells=True))
words = np.load('data_fmri_newer_version/words_fmri.npy')

# getting the beginning index of every word in string. this is important for then aligning with the tokens.
# note that the data cleaning process must be the same as was performed when running all the text through the model. So, there is different code for different models.
if llm_model_index == 0: #GPT2XL
    words_text_string = str(words[0])
    words_locations_in_text = [0]
    for word in words[1:]:
        words_locations_in_text.append(len(words_text_string))
        words_text_string += ' '+ word
elif llm_model_index == 1: #Llama70B
    words_text_string = str(words[0].replace('+', '').replace('@', '') )
    words_locations_in_text = [0]
    for word in words[1:]:
        word_ = word.replace('+', '').replace('@', '') # removing these symbols from the text
        words_locations_in_text.append(len(words_text_string))
        if len(word_)>0: words_text_string += ' '+ word_
words_locations_in_text = np.array(words_locations_in_text)

# %%
tokenized_input_text = np.load(f'{llm_model_activations_dir}/tokenized_input_text.npy', allow_pickle=True).item()
token_locations_in_text = tokenized_input_text['offset_mapping'].double().mean(axis=2)

non_padded_indexes = (token_locations_in_text != 0)
token_locations_in_text = token_locations_in_text[non_padded_indexes]
layer_activations = np.load(f'{llm_model_activations_dir}/activations_layer{llm_layer_i}.npy')[non_padded_indexes]

token_word_index = np.array([np.where(words_locations_in_text<=token_location.item())[0][-1] for token_location in token_locations_in_text])

layer_activations.shape, token_word_index.shape

# %%
def extract_relevant_brain_area_signal(subject_i):
    area_nums = [roi_num for roi_num, roi_name in enumerate(fmri_data_subjects[subject_i]['meta']['ROInumToName']) if "Temporal" in roi_name]
    all_voxels_data = fmri_data_subjects[subject_i]['data']
    all_voxels_roi_num = fmri_data_subjects[subject_i]['meta']['colToROInum']
    return all_voxels_data[:, np.isin(all_voxels_roi_num, area_nums)]
    #return subject_fmri_data.mean(axis=1)  # simply take the mean of ALL brain voxels' signals
fmri_signal_subjects = [extract_relevant_brain_area_signal(subject_i) for subject_i in range(fmri_num_subjects)]

# %%
subject_runs_llm_features = []
subject_runs_fMRI_signal = []
for subject_i in range(fmri_num_subjects):
    fmri_words_meta = fmri_data_subjects[subject_i]['words']

    current_time = -1
    last_TR_i = -1
    for token_i, word_i in enumerate(token_word_index):
        word_start_time = fmri_words_meta[word_i]['start']
        if word_start_time-current_time>2: 
            # if there have been more than 2 seconds between words, initiate new run
            subject_runs_llm_features.append([]) 
            subject_runs_fMRI_signal.append([])
        
        TR_i = int(word_start_time / 2) # index of the repetition with the data
        if last_TR_i != TR_i: 
            # if onto new TR, add its fMRI signal to the run and also make a new list of all of the tokens in this TR
            subject_runs_llm_features[-1].append([])
            subject_runs_fMRI_signal[-1].append(fmri_signal_subjects[subject_i][TR_i])
        layer_activations[token_i]
        subject_runs_llm_features[-1][-1].append(layer_activations[token_i]) # to this run and this TR, add the feature
        
        current_time = word_start_time
        last_TR_i = TR_i

    n_runs = len(subject_runs_llm_features)
    # converting fmri signal arrays to np arrays
    for run_i in range(n_runs): subject_runs_fMRI_signal[run_i] = np.array(subject_runs_fMRI_signal[run_i])
    # taking a mean of all tokens' features for every TR and making the whole thing a np array
    subject_runs_mean_llm_features = []
    for run_i in range(n_runs):
        subject_run_mean_llm_features = np.array([np.array(subject_runs_llm_features[run_i][tr_i]).mean(axis=0) for tr_i in range(len(subject_runs_llm_features[run_i]))])
        subject_runs_mean_llm_features.append(subject_run_mean_llm_features)

def high_pass_filter(data, cutoff_freq, tr):
    nyquist = 0.5 / tr
    normal_cutoff = cutoff_freq / nyquist
    b, a = scipy.signal.butter(5, normal_cutoff, btype='high', analog=False)
    return scipy.signal.filtfilt(b, a, data, axis=0)

def low_pass_filter(data, cutoff_freq, tr):
    nyquist = 0.5 / tr
    normal_cutoff = cutoff_freq / nyquist
    
    # Ensure the normalized cutoff frequency is between 0 and 1
    normal_cutoff = np.clip(normal_cutoff, 0, 0.99)
    
    b, a = scipy.signal.butter(5, normal_cutoff, btype='low', analog=False)
    return scipy.signal.filtfilt(b, a, data, axis=0)

subject_runs_fMRI_signal_normalized = []
subject_runs_mean_llm_features_normalized = []
for data_i, data in enumerate(subject_runs_fMRI_signal):
    filtered_data = high_pass_filter(data.mean(axis=1), cutoff_freq=0.005, tr=2)
    #filtered_data = low_pass_filter(filtered_data, cutoff_freq=0.25, tr=2)

    filtered_data = np.convolve(filtered_data, np.ones(convolve_size)/convolve_size, mode='same')

    z_scored_data = (filtered_data - filtered_data.mean()) / filtered_data.std()
    
    subject_runs_fMRI_signal_normalized.append(z_scored_data)
    subject_runs_mean_llm_features_normalized.append(subject_runs_mean_llm_features[data_i])

# %%
import numpy as np
from sklearn.decomposition import PCA

def apply_pca_to_runs(runs_data, n_components, keep_dims=False):
    # If n_components is negative, return original data without PCA
    if n_components < 0:
        return runs_data, None
    
    # Concatenate all runs
    all_data = np.concatenate(runs_data, axis=0)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(all_data)
    
    if keep_dims:
        # Project back to original space
        transformed_data = pca.inverse_transform(transformed_data)
    
    # Split back into runs
    transformed_runs = np.split(transformed_data, np.cumsum([len(run) for run in runs_data[:-1]]))
    
    return transformed_runs, pca

def create_training_dataset(subject_runs_mean_llm_features, subject_runs_fMRI_signal, 
                            n_delay_embedding_llm, n_delay_embedding_fmri, 
                            n_top_pc_llm, n_top_pc_fmri, 
                            keep_dims_llm=False, keep_dims_fmri=False):
    # Apply PCA to LLM features (or not, if n_top_pc_llm < 0)
    llm_pca_runs, llm_pca = apply_pca_to_runs(subject_runs_mean_llm_features, n_top_pc_llm, keep_dims_llm)

    # Apply PCA to fMRI signals (or not, if n_top_pc_fmri < 0)
    fmri_pca_runs, fmri_pca = apply_pca_to_runs(subject_runs_fMRI_signal, n_top_pc_fmri, keep_dims_fmri)
    
    X_llm_all = []
    X_fmri_all = []
    y_all = []
    
    for run_llm, run_fMRI in zip(llm_pca_runs, fmri_pca_runs):
        X_llm_run = []
        X_fmri_run = []
        y_run = []
        
        # Determine the maximum delay embedding
        max_delay = max(n_delay_embedding_llm, n_delay_embedding_fmri + 1)
        
        # Ensure we have enough data points in this run
        if len(run_llm) <= max_delay:
            print(f"Warning: Run with {len(run_llm)} points is too short for max delay embedding {max_delay}. Skipping.")
            continue
        
        # Create X_llm, X_fmri, and y for this run
        for i in range(max_delay, len(run_llm)):
            # For X_llm: take the last n_delay_embedding_llm vectors
            X_llm_run.append(run_llm[i-n_delay_embedding_llm:i])
            
            # For X_fmri: take the last n_delay_embedding_fmri vectors, excluding the current
            X_fmri_run.append(run_fMRI[i-n_delay_embedding_fmri-1:i-1])
            
            # For y: calculate the difference between current and previous fMRI signal
            y_run.append(run_fMRI[i] - run_fMRI[i-1])
        
        X_llm_all.append(np.array(X_llm_run))
        X_fmri_all.append(np.array(X_fmri_run))
        y_all.append(np.array(y_run))
    
    # Concatenate results from all runs
    X_llm = np.concatenate(X_llm_all, axis=0)
    X_fmri = np.concatenate(X_fmri_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    
    return X_llm, X_fmri, y, llm_pca, fmri_pca

X_llm, X_fmri, y, llm_pca, fmri_pca = create_training_dataset(
    subject_runs_mean_llm_features_normalized, 
    subject_runs_fMRI_signal_normalized, 
    n_delay_embedding_llm, 
    n_delay_embedding_fmri, 
    n_top_pc_llm, 
    n_top_pc_fmri,
    keep_dims_llm,
    keep_dims_fmri
)

# %%
# Define Linear Regression Model
class DelayedEmbeddingLinearModel(nn.Module):
    def __init__(self, X_llm, X_fmri):
        super(DelayedEmbeddingLinearModel, self).__init__()
        self.linear_llm1 = nn.Linear(X_llm.shape[-1], 1)  # Output size is 1 per timestep of delay embedding
        self.linear_llm2 = nn.Linear(X_llm.shape[1], 1) # final output
        self.linear_fmri = nn.Linear(X_fmri.shape[-1], 1)

    def forward(self, X):
        X_llm, X_fmri = X
        n_llm_batches, n_llm_timesteps, n_llm_features = X_llm.shape
        return self.linear_llm2(self.linear_llm1(X_llm).squeeze()) + self.linear_fmri(X_fmri)
        return self.linear_llm2(torch.maximum(self.linear_llm1(X_llm).squeeze(), torch.tensor(0))) + self.linear_fmri(X_fmri)
    

total_data_n = len(X_llm)
train_data_n = int(total_data_n * train_data_percentage)
test_data_n = int(total_data_n * test_data_percentage)
if test_sequential:
    train_indices = np.arange(train_data_n)
    test_indices = np.arange(train_data_n, train_data_n+test_data_n)
else:
    train_indices = np.random.choice(np.arange(total_data_n), size=train_data_n)
    test_indices = [x for x in range(len(X_llm)) if x not in train_indices]

X = torch.tensor(X_llm, dtype=torch.float32), torch.tensor(X_fmri, dtype=torch.float32)
X_train = torch.tensor(X_llm[train_indices], dtype=torch.float32), torch.tensor(X_fmri[train_indices], dtype=torch.float32)
y_train = torch.tensor(y[train_indices], dtype=torch.float32)
X_test = torch.tensor(X_llm[test_indices], dtype=torch.float32), torch.tensor(X_fmri[test_indices], dtype=torch.float32)
y_test = torch.tensor(y[test_indices], dtype=torch.float32)

# %%
# Initialize the model, loss function, and optimizer
regression_model = DelayedEmbeddingLinearModel(X_llm, X_fmri)
#regression_model = LinearRegressionModel(input_size=X.shape[1])
optimizer = optim.Adam(regression_model.parameters(), lr=0.01, weight_decay=weight_decay)
criterion = nn.MSELoss()

# Training loop with loss tracking
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    regression_model.train()
    
    # Forward pass
    outputs = regression_model(X_train).squeeze()
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Record training loss
    train_losses.append(loss.item())
    
    # Compute test loss
    regression_model.eval()
    with torch.no_grad():
        test_outputs = regression_model(X_test).squeeze()
        test_loss = criterion(test_outputs, y_test)
        test_losses.append(test_loss.item())
    
    # Print epoch progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

# Plot the training and testing loss
plt.figure(figsize=(8, 4))
plt.plot(range(0, num_epochs, 1), train_losses, label='Training Loss')
plt.plot(range(0, num_epochs, 1), test_losses, label='Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Testing Loss vs. Epochs')
plt.legend()
plt.grid(True)
plt.savefig(figure_prefix + "_trainingdynamics.pdf", bbox_inches="tight")

# %%
regression_model.eval()
model_predictions_train = regression_model(X_train).detach().cpu().numpy().reshape(-1)
model_predictions_test = regression_model(X_test).detach().cpu().numpy().reshape(-1)
model_predictions_all = regression_model(X).detach().cpu().numpy().reshape(-1)

N = 15

# %%
r_value_train, p_value_train = scipy.stats.pearsonr(y_train, model_predictions_train)

plt.figure(figsize=(4, 4))
plt.scatter(y_train, model_predictions_train, 10, color="k", alpha=.1)
plt.xlabel('real difference in fMRI signal')
plt.ylabel('model predicted')
plt.title(f'r = {r_value_train:.4f}, p = {p_value_train:.4f}')
plt.grid(True)
plt.savefig(figure_prefix + "_correlation_train.pdf", bbox_inches="tight")

# %%
r_value_test, p_value_test = scipy.stats.pearsonr(model_predictions_test, y_test)

plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})
plt.figure(figsize=(4, 4))
plt.scatter(y_test, model_predictions_test, 10, color="k", alpha=.1)
plt.xlabel('real difference in fMRI signal')
plt.ylabel('model predicted')
plt.title(f'r = {r_value_test:.4f}, p = {p_value_test:.4f}')
plt.grid(True)
plt.savefig(figure_prefix + "_correlation_test.pdf", bbox_inches="tight")

# %%
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Calculate mean across subjects
mean_data = y
# Calculate SEM using scipy
error_data = y*0
error_label = '---'
# Create time axis
time_points = len(mean_data)
time_axis = np.arange(time_points)

# Split data into train and test
train_data = mean_data[:train_data_n]
test_data = mean_data[train_data_n:]
train_error = error_data[:train_data_n]
test_error = error_data[train_data_n:]
train_time = time_axis[:train_data_n]
test_time = time_axis[train_data_n:]

# Split model predictions
train_predictions = model_predictions_all[:train_data_n]
test_predictions = model_predictions_all[train_data_n:]

# Calculate the ratio for subplot heights
train_ratio = train_data_n / time_points
test_ratio = 1 - train_ratio

# Create the plot
plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})
fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])

# Top plot (train data)
ax1.plot(train_time, train_data, color='k', label='mean', linewidth=1.5)
ax1.plot(train_time, train_predictions, color='red', linewidth=1.5, label='model prediction')
ax1.fill_between(train_time, train_data - train_error, train_data + train_error, 
                 color='k', alpha=0.2, label=error_label)
ax1.set_ylabel('fMRI Signal (z-scored)')
ax1.set_ylim(-1.5, 1.5)
ax1.grid(True, linestyle='--', alpha=0.7)
#ax1.legend(loc='upper right')
ax1.set_title('train')

# Bottom plot (test data)
ax2.plot(test_time, test_data, color='k', label='mean', linewidth=1.5)
ax2.plot(test_time, test_predictions, color='red', linewidth=1.5, label='model prediction')
ax2.fill_between(test_time, test_data - test_error, test_data + test_error, 
                 color='k', alpha=0.2, label=error_label)
ax2.set_xlabel('token #')
ax2.set_ylabel('fMRI Signal (z-scored)')
ax2.set_ylim(-1.5, 1.5)
ax2.grid(True, linestyle='--', alpha=0.7)
#ax2.legend(loc='upper right')
ax2.set_title('test')

# Adjust x-limits to make the plots continuous
ax1.set_xlim(train_data_n//16, train_data_n//8)
ax2.set_xlim(train_data_n, train_data_n+train_data_n//16)
plt.tight_layout()
plt.savefig(figure_prefix + "_fit.pdf", bbox_inches="tight")

# %%
import json

# Create a dictionary with all the parameters and results
data_to_save = {
    "r_value_train": float(r_value_train),
    "p_value_train": float(p_value_train),
    "r_value_test": float(r_value_test),
    "p_value_test": float(p_value_test),
    "llm_model_name": llm_model_name,
    "llm_layer_i": llm_layer_i,
    "llm_d_model": llm_d_model,
    "llm_n_ctx": llm_n_ctx,
    "n_delay_embedding_llm": n_delay_embedding_llm,
    "n_delay_embedding_fmri": n_delay_embedding_fmri,
    "n_top_pc_llm": n_top_pc_llm,
    "n_top_pc_fmri": n_top_pc_fmri,
    "keep_dims_llm": keep_dims_llm,
    "keep_dims_fmri": keep_dims_fmri,
    "convolve_size": convolve_size,
    "weight_decay": weight_decay,
    "train_data_percentage": train_data_percentage,
    "replace_x_with_noise": replace_x_with_noise,
    "test_sequential": test_sequential
}

# Save the data to a JSON file
with open(figure_prefix + "_data.json", "w") as f:
    json.dump(data_to_save, f, indent=4)

print(f"Data saved to {figure_prefix}_data.json")

# %%
