import torch
from model import GPT, GPTConfig
from HW5.HW5.code.tokenizer import SimpleTokenizer
from datasets import load_dataset

# Path to your trained model
model_path = '..\cond_gpt\weights\jump_split_addprim_jumpsplit_2layer_2head_16embd_32bs.pt'

# Fix the generate_sample function to not require GPU
def generate_sample(model, tokenizer, conditions, max_length):
    model.eval()
    input_ids = tokenizer.generation_encode(conditions)
    input_ids = torch.tensor([input_ids], dtype=torch.long).cuda()
    len_conditions = len(input_ids[0])

    # Print the tokenized input for demonstration
    print(f"Input command: '{conditions}'")
    print(f"Tokenized input: {[tokenizer.reverse_vocab.get(id, '<unk>') for id in input_ids[0]]}")
    print(f"Token IDs: {input_ids[0].tolist()}\n")

    # Step-by-step generation
    with torch.no_grad():
        for step in range(max_length - len_conditions):
            # Forward pass through the model
            logits, _, _ = model(input_ids)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Get the token name
            next_token_name = tokenizer.reverse_vocab.get(next_token.item(), '<unk>')

            # Print this step in a generation
            print(f"Step {step + 1}:")
            print(f"  Generated token: {next_token_name} (Token ID: {next_token.item()})")

            # Add the new token to the sequence
            input_ids = torch.cat((input_ids, next_token), dim=1)

            # Print the updated sequence
            print(f"  Updated sequence: {[tokenizer.reverse_vocab.get(id, '<unk>') for id in input_ids[0]]}\n")

            # Stop if we generated the end token
            if next_token.item() == tokenizer.vocab["</s>"]:
                print("End token generated. Stopping generation.\n")
                break

    # Final output
    generated_text = tokenizer.decode(input_ids[0][len_conditions:])
    return generated_text


def main():
    print("Loading SCAN dataset and tokenizer...")
    data_SCAN = load_dataset("scan", "simple", trust_remote_code=True)

    # Load tokenizer from saved vocabulary
    tokenizer = SimpleTokenizer(max_length=128)
    tokenizer_path = './tokenizer/simple_vocab.json'
    tokenizer.load_vocab(tokenizer_path)
    tokenizer.reverse_vocab = {v: k for k, v in tokenizer.vocab.items()}

    # Configure and load model
    print("Loading model...")
    vocab_size = len(tokenizer.vocab)
    mconf = GPTConfig(vocab_size, 128, n_layer=2, n_head=2, n_embd=16, isconditional=True)
    model = GPT(mconf)

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test with concrete example
    print("\n--- CONCRETE EXAMPLE DEMONSTRATION ---")
    example = "jump right"
    generated = generate_sample(model, tokenizer, example, 128)

    print(f"Final generated action sequence: '{generated}'")
    print(f"Expected action sequence:        'I_JUMP I_TURN_RIGHT'")

    if generated == "I_JUMP I_TURN_RIGHT":
        print("\nSuccess! The model correctly translated the command.")
    else:
        print("\nNote: The generated sequence doesn't match the expected output.")


if __name__ == "__main__":
    main()
