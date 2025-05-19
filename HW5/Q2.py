import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class TrigramAttention(nn.Module):
    """
    Modified self-attention mechanism that implements trigram constraints.
    Each token can only attend to itself and the previous two tokens.
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        assert n_embd % n_head == 0, "embedding dimension must be divisible by number of heads"
        self.head_size = n_embd // n_head

        # Key, query, value projections
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # Output projection
        self.proj = nn.Linear(n_embd, n_embd)

        # Register a buffer for the trigram attention mask
        self.register_buffer("trigram_mask", None)

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality

        # If trigram mask doesn't exist or has wrong size, create it
        if self.trigram_mask is None or self.trigram_mask.shape[1] != T:
            # Create trigram mask (each position can only attend to itself and 2 previous positions)
            trigram_mask = torch.tril(torch.ones(T, T), diagonal=0)

            # Zero out attention to positions beyond the trigram window
            for i in range(T):
                for j in range(T):
                    if i - j > 2:  # If position j is more than 2 steps before position i
                        trigram_mask[i, j] = 0

            self.register_buffer("trigram_mask", trigram_mask.view(1, 1, T, T))

        # Linear projections
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)

        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))  # (B, nh, T, T)

        # Apply trigram attention mask
        att = att.masked_fill(self.trigram_mask == 0, float('-inf'))

        # Softmax attention weights
        att = torch.softmax(att, dim=-1)

        # Apply attention to values
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        # Output projection
        y = self.proj(y)

        return y


class TrigramTransformerBlock(nn.Module):
    """Transformer block modified to implement trigram constraints"""

    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        # Use trigram attention instead of standard causal attention
        self.attn = TrigramAttention(n_embd, n_head)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class TrigramGPT(nn.Module):
    """GPT-like model restricted to trigram context window"""

    def __init__(
            self,
            vocab_size,
            n_embd=128,
            n_head=4,
            n_layer=4,
            max_seq_len=256,
            dropout=0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.max_seq_len = max_seq_len

        # Token and positional embeddings
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, n_embd))

        # Transformer blocks with trigram attention
        self.transformer_blocks = nn.ModuleList([
            TrigramTransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.max_seq_len, "Sequence length exceeds model capacity"

        # Get token embeddings
        token_embeddings = self.tok_emb(idx)  # (B, T, C)

        # Add positional embeddings
        position_embeddings = self.pos_emb[:, :T, :]  # (1, T, C)
        x = self.dropout(token_embeddings + position_embeddings)  # (B, T, C)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            # Reshape logits and compute cross entropy loss
            B, T, C = logits.size()
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Generate new tokens autoregressively"""
        for _ in range(max_new_tokens):
            # Get last three tokens (trigram context)
            idx_cond = idx[:, -3:] if idx.size(1) > 3 else idx

            # Forward pass to get logits
            with torch.no_grad():
                logits, _ = self.forward(idx_cond)

            # Get logits for the last token
            logits = logits[:, -1, :] / temperature

            # Apply softmax to get probabilities
            probs = nn.functional.softmax(logits, dim=-1)

            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to the sequence
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


# Example usage and visualization
def visualize_attention_mask():
    """Visualize the difference between standard causal mask and trigram mask"""
    seq_len = 10

    # Standard causal mask (lower triangular)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))

    # Trigram mask
    trigram_mask = torch.tril(torch.ones(seq_len, seq_len))
    for i in range(seq_len):
        for j in range(seq_len):
            if i - j > 2:  # If position j is more than 2 steps before position i
                trigram_mask[i, j] = 0

    # Visualize the masks
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].matshow(causal_mask, cmap='viridis')
    axes[0].set_title('Standard Causal Mask')
    axes[0].set_xlabel('Key position')
    axes[0].set_ylabel('Query position')

    axes[1].matshow(trigram_mask, cmap='viridis')
    axes[1].set_title('Trigram Mask')
    axes[1].set_xlabel('Key position')
    axes[1].set_ylabel('Query position')

    plt.tight_layout()
    plt.savefig('attention_masks_comparison.png')

    return fig


def train_and_evaluate_trigram_model():
    """Demo function to train and evaluate the trigram model on a toy task"""
    # Toy vocabulary and dataset for demonstration
    vocab_size = 100
    seq_length = 20
    batch_size = 32

    # Create a toy dataset (random sequences)
    def get_batch():
        # Generate random token indices
        x = torch.randint(1, vocab_size, (batch_size, seq_length))
        y = torch.roll(x, shifts=-1, dims=1)
        y[:, -1] = 0  # padding
        return x, y

    # Initialize the model
    model = TrigramGPT(vocab_size=vocab_size, max_seq_len=seq_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    num_epochs = 10

    # Training loop
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 100

        for _ in range(num_batches):
            x, y = get_batch()
            x, y = x.to(device), y.to(device)

            # Forward pass
            logits, loss = model(x, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('trigram_model_training_loss.png')

    # Generate text using the model
    print("\nText generation example:")

    start_tokens = torch.tensor([[1, 2, 3]]).to(device)  # Start with some seed tokens
    generated = model.generate(start_tokens, max_new_tokens=20)

    print("Generated sequence:")
    print(generated.cpu().numpy())

    return model


if __name__ == "__main__":
    print("Visualizing attention masks to compare standard causal attention vs trigram attention...")
    visualize_attention_mask()

    print("\nTraining and evaluating trigram language model...")
    model = train_and_evaluate_trigram_model()

    print("\nDifferences between standard transformer and trigram transformer:")
    print("1. In standard transformer: each token can attend to all previous tokens")
    print("2. In trigram transformer: each token can only attend to itself and previous two tokens")
    print("3. Implementation difference: modified attention mask restricts the context window")
