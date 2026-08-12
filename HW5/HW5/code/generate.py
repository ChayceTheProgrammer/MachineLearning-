import torch
from datasets import load_dataset
from tqdm import tqdm

from model import GPT, GPTConfig
from HW5.HW5.code.tokenizer import build_tokenizer


def load_model(model_path, config):
    model = GPT(config)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model

def generate_sample(model, tokenizer, conditions, max_length, verbose=True):
    model.eval()
    input_ids = tokenizer.generation_encode(conditions)
    input_ids = torch.tensor([input_ids], dtype=torch.long) #.cuda()
    len_conditions = len(input_ids[0])

    if verbose:
        print(f"Input command: '{conditions}'")
        print(f"Tokenized input: {[tokenizer.reverse_vocab.get(id, '<unk>') for id in input_ids[0]]}")
        print(f"Token IDs: {input_ids[0].tolist()}")

    with torch.no_grad():
        for _ in range(max_length - len_conditions):
            # Generate one token at a time, and append it to the input to do generation iteratively until </s> is generated
            ### YOUR CODE HERE ###

            logits, _, _ = model(input_ids)  # Get model predictions
            next_token_logits = logits[:, -1, :]  # Get logits for the next token position
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # Choose the most likely token
            input_ids = torch.cat((input_ids, next_token), dim=1)  # Append to input sequence

            # Stop if we generated the end token
            if next_token.item() == tokenizer.vocab["</s>"]:
                if verbose:
                    print("End token generated. Stopping generation.")
                break

            ### YOUR CODE HERE ###
        # Check if any tokens were generated
    if len(input_ids[0][len_conditions:]) == 0:
        print("No tokens were generated beyond the input conditions.")
        return ""

    generated_text = tokenizer.decode(input_ids[0][len_conditions:])
    return generated_text


def generate(args):

    data_SCAN = load_dataset("scan", args.data_split)

    max_len = args.max_len
    tokenizer, vocab_size = build_tokenizer(args, data_SCAN, max_len, args.output_tokenizer_dir)

    mconf = GPTConfig(vocab_size, max_len,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      isconditional=True)

    # Load model and tokenizer
    print("loading model")
    model = load_model(args.ckpt_path, mconf)#.cuda()
    print('total params:', sum(p.numel() for p in model.parameters()))


    # Sample generation
    test_data = data_SCAN['test']
    correct_count = 0
    pbar = tqdm(enumerate(test_data), total=len(test_data))
    for i, data in pbar:
        generated_actions = generate_sample(model, tokenizer, data['commands'], max_len)
        if generated_actions == data['actions']:
            correct_count += 1
        pbar.set_description(f'Accuracy: {correct_count / (i + 1):.4f}')
    print(f'Test accuracy: {correct_count / len(test_data)}')
