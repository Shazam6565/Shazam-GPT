

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.multiprocessing as mp
import dill
from datetime import datetime
import tiktoken
import dill

current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# batch_size = 32
# block_size = 128
# max_iters = 200
# learning_rate = [1e-4,3e-4,1e-5,1e-6,2e-4]
# self.eval_iters = 500
# n_embd = 400
# n_head = 3
# n_layer = 3
# dropout = 0.2
# vocab_size = 100256

# device = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > gpu_selector else 'cpu'
# if device == 'cuda':
#     device = torch.device(f"cuda{}")
# else:
#     device = torch.device('cpu')

# model = GPTLanguageModel(vocab_size)

class Trainer():
    def __init__(self, model, device, learning_rate,data,splits,eval_iters,max_iters,gpu_id):
        self.model = model
        device = torch.device(f'cuda:{gpu_id}')
        self.device = device
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        # self.get_batch = data.get_batch
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.writer = SummaryWriter()
        self.splits = splits
        self.eval_iters = eval_iters
        self.max_iters= max_iters
        self.data = data
        
        
    

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.data.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def generated_char(self):
        prompt = 'The market-place'
        context = torch.tensor(self.enc.encode(prompt), dtype=torch.long, device=device)
        output,index_next = self.model.generate(context.unsqueeze(0), max_new_tokens=100)
        generated_chars = self.enc.decode(output[0].tolist())
        predictions = self.enc.decode(index_next[0].tolist())
        # generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=100)[0].tolist())
        return generated_chars, predictions
    # create a PyTorch optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    def calculate_perplexity(self,loss):
        return torch.exp(loss)

    def modeltraining(self):
        for iter in range(self.max_iters):
            
            # Print the iteration number
            # print(iter)
            
            # Evaluate the model and print the losses at the specified interval
            if iter % self.eval_iters == 0:
                filename = f"model-2_{current_date}_{self.splits}_{self.learning_rate}.pkl"
                with open(filename, 'wb') as f:
                    dill.dump(self.model, f)
                    print('Model saved at iteration', iter)

                losses = self.estimate_loss()
                print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}",flush=True)
                train_perplexity = self.calculate_perplexity(losses['train'])
                val_perplexity = self.calculate_perplexity(losses['val'])
                print(f'perplexity{train_perplexity},{val_perplexity}')
                # Write the losses to tensorboard
                self.writer.add_scalar('Train Loss', losses['train'], iter)
                self.writer.add_scalar('Validation Loss', losses['val'], iter)
                output, next_prediction = self.generated_char()
                # Generate characters using the model
                print("-----------------------------------------------------")
                print(f'Ouput of model after {iter}: {output}',flush=True)
                print("-----------------------------------------------------")
                print(f'Next prediction of model after {iter}: {next_prediction}',flush=True)
                print("-----------------------------------------------------")
            
            # Sample a batch of data
            xb, yb = self.data.get_batch('train')
            
            # Evaluate the loss
            logits, loss = self.model.forward(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            
            # Write the training loss to tensorboard
            self.writer.add_scalar('Training Loss', loss.item(), iter)

        # Save the model

            # Your training code here
        # with open('model-01.pkl', 'wb') as f:
        #     dill.dump(model, f)


        # Close the tensorboard self.writer
        self.writer.close()
