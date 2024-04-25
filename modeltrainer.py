import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse
from torch.nn.parallel import DataParallel

import tiktoken
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
parser = argparse.ArgumentParser(description='This is a demonstration program')

# Define command-line arguments
parser.add_argument('--tokenizer', type=str, required=True, help='Please provide a tokenizer', default="cl100k_base")
parser.add_argument('--GPU', type=int, required=False, help='Please specify a GPU index', default=0)

# Parse the command-line arguments
args = parser.parse_args()

# Access the arguments using the args object
tokenizer = args.tokenizer
gpu_selector = args.GPU

# args = parser.parse_args()

# Now we can use the argument value in our program.
# print(f'batch size: {args.batch_size}')
device = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > gpu_selector else 'cpu'
if device == 'cuda':
    device = torch.device(f"cuda:{gpu_selector}")
else:
    device = torch.device('cpu')

# batch_size = args.batch_size # to use the batch_size cmd arg -> python file_name.py -batch_size 32
batch_size = 32
block_size = 128
max_iters = 200
learning_rate = 3e-4
eval_iters = 5000
n_embd = 384
n_head = 4
n_layer = 4
dropout = 0.2

print(device)
print(f"Using tokenizer: {args.tokenizer}")
print(f"Using GPU index: {args.GPU}")

tokenizer = args.tokenizer
chars = ""
with open("vocab.txt", 'r', encoding='utf-8') as f:
        text = f.read()
        chars = sorted(list(set(text)))
        
vocab_size = {"gpt-4": 32000, "gpt-3":60000,"cl100k_base":100256}
vocab_size = vocab_size[tokenizer]

string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }
# encode = lambda s: [string_to_int[c] for c in s]
# decode = lambda l: ''.join([int_to_string[i] for i in l])


# enc = tiktoken.encoding_for_model(tokenizer)
enc = tiktoken.get_encoding(tokenizer)

# n = int(0.8*len(data))
# train_data = data[:n]
# val_data = data[n:]

# memory map for using small snippets of text from a single file of any size
def get_random_chunk(split):
    filename = "Stock_Exchange.txt" if split == 'train' else "Stock_Exchange.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)

            # Seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)

            # Decode the block to a string, ignoring any invalid byte sequences
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            
            # Train and test splits
            data = torch.tensor(enc.encode(decoded_block), dtype=torch.long)
            n = int(0.8*len(data))
            if split == 'train':
                data[:n]
            else:
                data[n:]
            
    return data


def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

# [1, 0, 0]
# [1, 0.6, 0]
# [1, 0.6, 0.4]
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, F) -> (B, T, [h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3])
        out = self.dropout(self.proj(out))
        return out
    

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x
    
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        
        
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(index) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            index_cond = index[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(index_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1)
        return index, index_next

model = GPTLanguageModel(vocab_size)
# model = DataParallel(model)
# print('loading model parameters...')
# with open('model-01.pkl', 'rb') as f:
#     model = pickle.load(f)
# print('loaded successfully!')
m = model.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def generated_char():
    prompt = 'The market-place'
    context = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device)
    output,index_next = model.generate(context.unsqueeze(0), max_new_tokens=100)
    generated_chars = enc.decode(output[0].tolist())
    predictions = enc.decode(index_next[0].tolist())
    # generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=100)[0].tolist())
    return generated_chars, predictions
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Train the model
print("Started Training")
for iter in range(555000):
    
    # Print the iteration number
    # print(iter)
    
    # Evaluate the model and print the losses at the specified interval
    if iter % eval_iters == 0:
        
        with open(f'model-{tokenizer}.pkl', 'wb') as f:
            pickle.dump(model, f)
            print('Model saved at iteration', iter)

        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}",flush=True)
       
        # Write the losses to tensorboard
        writer.add_scalar('Train Loss', losses['train'], iter)
        writer.add_scalar('Validation Loss', losses['val'], iter)
        output, next_prediction = generated_char()
        # Generate characters using the model
        print("-----------------------------------------------------")
        print(f'Ouput of model after {iter}: {output}',flush=True)
        print("-----------------------------------------------------")
        print(f'Next prediction of model after {iter}: {next_prediction}',flush=True)
        print("-----------------------------------------------------")
    
    # Sample a batch of data
    xb, yb = get_batch('train')
    
    # Evaluate the loss
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    # Write the training loss to tensorboard
    writer.add_scalar('Training Loss', loss.item(), iter)

# Save the model

    # Your training code here
with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)


# Close the tensorboard writer
writer.close()
