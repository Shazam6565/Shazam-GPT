import torch
import tiktoken
import random 
import mmap




class data_loader():
    def __init__(self, batch_size, block_size, device,splits,filename):
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.filename = filename
        self.splits = splits 
        self.data = None
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.x = []
        self.y = []
        
    def load_data(self):
        with open(self.filename, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                # Determine the file size and a random position to start reading
                file_size = len(mm)
                start_pos = random.randint(0, (file_size) - self.block_size*self.batch_size)

                # Seek to the random position and read the block of text
                mm.seek(start_pos)
                block = mm.read(self.block_size*self.batch_size-1)

                # Decode the block to a string, ignoring any invalid byte sequences
                decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
                
                # Train and test splits
                self.data = torch.tensor(self.enc.encode(decoded_block), dtype=torch.long)
    def get_random_chunk(self,split):
        # filename = self.filename if split == 'train' else "Stock_Exchange.txt"
        
                # split = [0.8,0.7,0.9]
                
        n = int(self.splits*len(self.data))
        if split == 'train':
            self.data = self.data[:n]
        else:
            self.data = self.data[n:]
                
        return self.data


    def get_batch(self,split):
        self.data = self.get_random_chunk(split)
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        self.x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        self.y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        self.x, self.y = self.x.to(self.device), self.y.to(self.device)
        return self.x, self.y