from DataLoader.dataloader import data_loader
import torch
import multiprocessing as mp


from model.shazamgpt import GPTLanguageModel
from run import Trainer

mp.set_start_method('spawn', force=True)
batch_size = 32
block_size = 128
max_iters = 200
learning_rate = [1e-4,3e-4,1e-5,1e-6,2e-4]
eval_iters = 500
n_embd = 400
n_head = 3
n_layer = 3
dropout = 0.2
vocab_size = 100256
gpu_selector  = torch.cuda.device_count() 
device = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > gpu_selector else 'cpu'
if device == 'cuda':
    device = torch.device(f"cuda:")
else:
    device = torch.device('cpu')

splits = [0.7,0.8,0.9]
model = GPTLanguageModel(vocab_size,device)
filename ='Stock_Exchange.txt'

data_split_1 = data_loader(batch_size=batch_size, block_size=block_size, device=device,splits=splits[:1],filename=filename)
data_split_2 = data_loader(batch_size=batch_size, block_size=block_size, device=device,splits=splits[:2],filename=filename)
data_split_3 = data_loader(batch_size=batch_size, block_size=block_size, device=device,splits=splits[:3],filename=filename)


data = [data_split_1, data_split_2,data_split_3]
gpu_usage_cache = {i: False for i in range(gpu_selector)}
gpu_count = torch.cuda.device_count()

# Dictionary to track GPU usage
gpu_usage_cache = {i: False for i in range(gpu_count)}

class TrainerProcess(mp.Process):
    def __init__(self, model, device, learning_rate, data, splits, eval_iters, max_iters, gpu_id):
        super(TrainerProcess, self).__init__()
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.data = data
        self.splits = splits
        self.eval_iters = eval_iters
        self.max_iters = max_iters
        self.gpu_id = gpu_id
        
    def run(self):
        torch.cuda.set_device(self.gpu_id)  # Set the GPU for this process
        trainer = Trainer(self.model, self.device, self.learning_rate, self.data, self.splits,
                          self.eval_iters, self.max_iters, self.gpu_id)
        trainer.train()

# Create and start a TrainerProcess for each data split, assigning a GPU to each process
processes = []
for i, data in enumerate(data):
    # Find an available GPU
    gpu_id = None
    for j in range(gpu_count):
        if not gpu_usage_cache[j]:
            gpu_id = j
            gpu_usage_cache[j] = True
            break
    if gpu_id is not None:
        process = TrainerProcess(model, device, learning_rate, data, splits, eval_iters, max_iters, gpu_id)
        processes.append(process)
        process.start()
    else:
        print("No available GPU for data split ", i)
        break

# Wait for all processes to finish
for process in processes:
    process.join()

# Reset GPU usage cache
gpu_usage_cache = {i: False for i in range(gpu_count)}