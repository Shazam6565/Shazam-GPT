
from ..model.trainer import GPTLanguageModel()
import dill
batch_size = 32
block_size = 128
max_iters = 200
learning_rate = [1e-4,3e-4,1e-5,1e-6,2e-4]
eval_iters = 500
n_embd = 400
n_head = 3
n_layer = 3
dropout = 0.2


device = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > gpu_selector else 'cpu'
if device == 'cuda':
    device = torch.device(f"cuda")
else:
    device = torch.device('cpu')




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
def calculate_perplexity(loss):
    return torch.exp(loss)


for iter in range(555000):
    
    # Print the iteration number
    # print(iter)
    
    # Evaluate the model and print the losses at the specified interval
    if iter % eval_iters == 0:
        filename = f"model-2_{current_date}_{tokenizer}.pkl"
        with open(filename, 'wb') as f:
            dill.dump(model, f)
            print('Model saved at iteration', iter)

        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}",flush=True)
        train_perplexity = calculate_perplexity(losses['train'])
        val_perplexity = calculate_perplexity(losses['val'])
        print(f'perplexity{train_perplexity},{val_perplexity}')
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
# with open('model-01.pkl', 'wb') as f:
#     pickle.dump(model, f)


# Close the tensorboard writer
writer.close()
