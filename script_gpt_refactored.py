"""
================================================================================
MINIMAL GPT IMPLEMENTATION IN PURE PYTHON
================================================================================

A complete GPT (Generative Pre-trained Transformer) implementation in 
dependency-free Python, demonstrating the core algorithmic concepts of 
modern language models.

This implementation includes:
- Automatic differentiation (Autograd) for gradient computation
- Multi-head self-attention mechanism
- Transformer architecture with RMSNorm
- Adam optimizer with learning rate decay
- Training and inference pipelines

Based on Andrej Karpathy's minimal GPT implementation.

Architecture Overview:
--------------------------------------------------------------------------------

```mermaid
graph TB
    A[Input Text] --> B[Tokenizer]
    B --> C[Token Embedding]
    B --> D[Position Embedding]
    C --> E[GPT Model]
    D --> E
    E --> F[Softmax]
    F --> G[Predicted Tokens]
    E --> H[Loss Calculation]
    H --> I[Backward Pass]
    I --> J[Adam Optimizer]
    J --> E
```

Training Pipeline:
--------------------------------------------------------------------------------

```mermaid
graph LR
    A[Dataset] --> B[Sample Document]
    B --> C[Tokenize]
    C --> D[Forward Pass]
    D --> E[Compute Loss]
    E --> F[Backward Pass]
    F --> G[Compute Gradients]
    G --> H[Update Parameters]
    H --> D
```

Author: Based on work by Andrej Karpathy
"""

# =============================================================================
# IMPORTS AND CONFIGURATION
# =============================================================================

import os       # os.path.exists - for file operations
import math     # math.log, math.exp - for mathematical operations
import random   # random.seed, random.choices, random.gauss, random.shuffle

# Set random seed for reproducibility
random.seed(42)

print("=" * 80)
print("MINIMAL GPT IMPLEMENTATION")
print("=" * 80)

# =============================================================================
# HYPERPARAMETERS AND MODEL CONFIGURATION
# =============================================================================

"""
Model Architecture Parameters:
------------------------------
- VOCAB_SIZE: Number of unique tokens in the vocabulary
- N_EMBD: Dimension of token embeddings (size of each token's vector representation)
- N_HEAD: Number of attention heads (parallel attention mechanisms)
- N_LAYER: Number of transformer layers (depth of the network)
- BLOCK_SIZE: Maximum sequence length (context window)
- HEAD_DIM: Dimension of each attention head = N_EMBD / N_HEAD

Training Hyperparameters:
-------------------------
- LEARNING_RATE: Initial learning rate for optimizer
- BETA1, BETA2: Adam optimizer momentum parameters
- EPS_ADAM: Small constant to prevent division by zero
- NUM_STEPS: Number of training iterations
- TEMPERATURE: Controls randomness in generation (lower = more deterministic)
"""

# Model architecture
N_EMBD = 16          # Embedding dimension (each token becomes a 16D vector)
N_HEAD = 4           # Number of attention heads (parallel processing)
N_LAYER = 1          # Number of transformer layers (can be increased for more capacity)
BLOCK_SIZE = 16      # Maximum sequence length (context window)
HEAD_DIM = N_EMBD // N_HEAD  # Dimension per head = 16 / 4 = 4

# Training hyperparameters
LEARNING_RATE = 0.01
BETA1 = 0.85         # Adam momentum parameter (first moment)
BETA2 = 0.99         # Adam momentum parameter (second moment)
EPS_ADAM = 1e-8      # Small constant to prevent division by zero
NUM_STEPS = 1000     # Number of training iterations
TEMPERATURE = 0.5    # Sampling temperature (0.5 = moderately creative)

# Initialization
WEIGHT_INIT_STD = 0.08  # Standard deviation for weight initialization

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_dataset(filename='input.txt', url=None):
    """
    Load or download the training dataset.
    
    Args:
        filename: Path to save/load the dataset
        url: URL to download dataset if file doesn't exist
    
    Returns:
        list: List of documents (strings)
    """
    # Download dataset if it doesn't exist
    if not os.path.exists(filename):
        print(f"\n[Data] Downloading dataset to {filename}...")
        import urllib.request
        if url is None:
            # Default: names dataset from makemore repository
            url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
        urllib.request.urlretrieve(url, filename)
    
    # Load and parse documents
    print(f"\n[Data] Loading dataset from {filename}...")
    docs = [
        line.strip() 
        for line in open(filename).read().strip().split('\n') 
        if line.strip()
    ]
    
    # Shuffle for randomness during training
    random.shuffle(docs)
    
    print(f"[Data] Loaded {len(docs)} documents")
    return docs

# Load the dataset
docs = load_dataset()

# =============================================================================
# TOKENIZER
# =============================================================================

class Tokenizer:
    """
    Converts between text strings and integer token IDs.
    
    The tokenizer creates a vocabulary of unique characters from the training
    data and adds a special BOS (Beginning of Sequence) token.
    
    Tokenization Process:
    ---------------------
    
    ```mermaid
    graph LR
        A[Raw Text: hello] --> B[Split into chars: h,e,l,l,o]
        B --> C[Map to IDs: 8,5,12,12,15]
        C --> D[Add BOS: 26,8,5,12,12,15,26]
    ```
    
    Attributes:
        uchars: Sorted list of unique characters in the dataset
        bos_token_id: Token ID for the special BOS token
        vocab_size: Total number of unique tokens including BOS
    """
    
    def __init__(self, docs):
        """
        Initialize tokenizer by building vocabulary from documents.
        
        Args:
            docs: List of training documents
        """
        # Extract all unique characters from the dataset
        self.uchars = sorted(set(''.join(docs)))
        
        # BOS (Beginning of Sequence) token ID
        # This special token marks the start and end of sequences
        self.bos_token_id = len(self.uchars)
        
        # Total vocabulary size includes all characters + BOS token
        self.vocab_size = len(self.uchars) + 1
        
        print(f"\n[Tokenizer] Vocabulary size: {self.vocab_size}")
        print(f"[Tokenizer] Characters: {''.join(self.uchars)}")
        print(f"[Tokenizer] BOS token ID: {self.bos_token_id}")
    
    def encode(self, text):
        """
        Convert text string to list of token IDs.
        
        Args:
            text: String to encode
        
        Returns:
            list: Token IDs with BOS tokens at start and end
        """
        # Convert each character to its token ID
        tokens = [self.uchars.index(ch) for ch in text]
        
        # Add BOS tokens at start and end
        return [self.bos_token_id] + tokens + [self.bos_token_id]
    
    def decode(self, token_ids):
        """
        Convert list of token IDs back to text string.
        
        Args:
            token_ids: List of token IDs to decode
        
        Returns:
            str: Decoded text string (excludes BOS tokens)
        """
        # Convert token IDs back to characters, excluding BOS tokens
        return ''.join([
            self.uchars[tid] 
            for tid in token_ids 
            if tid != self.bos_token_id
        ])

# Initialize tokenizer
tokenizer = Tokenizer(docs)

# =============================================================================
# AUTOGRAD ENGINE (AUTOMATIC DIFFERENTIATION)
# =============================================================================

class Value:
    """
    A scalar value that supports automatic differentiation.
    
    This class implements a computation graph where each operation creates
    a new node. During the backward pass, gradients are computed using the
    chain rule by traversing the graph in reverse topological order.
    
    Computation Graph Example:
    -------------------------
    
    ```mermaid
    graph TD
        A[a] --> C[c = a + b]
        B[b] --> C
        C --> D[e = c * d]
        D[d] --> D
        D --> E[Loss]
        E -->|Backward Pass| D
        D -->|∂L/∂e| C
        C -->|∂L/∂c| A
        C -->|∂L/∂c| B
        D -->|∂L/∂e| D
    ```
    
    Attributes:
        data: The scalar value of this node (computed during forward pass)
        grad: Gradient of loss with respect to this node (computed during backward)
        _children: Child nodes in the computation graph
        _local_grads: Local gradients for each child (derivative of operation)
    
    Note:
        __slots__ is used for memory optimization in Python
    """
    
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    
    def __init__(self, data, children=(), local_grads=()):
        """
        Initialize a Value node.
        
        Args:
            data: Scalar value
            children: Tuple of child nodes this node depends on
            local_grads: Tuple of local gradients (derivatives) for each child
        """
        self.data = data                # Forward pass value
        self.grad = 0                   # Backward pass gradient
        self._children = children       # Dependencies in computation graph
        self._local_grads = local_grads # Local derivatives
    
    def __add__(self, other):
        """
        Addition operation: z = x + y
        Local gradients: ∂z/∂x = 1, ∂z/∂y = 1
        """
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))
    
    def __mul__(self, other):
        """
        Multiplication operation: z = x * y
        Local gradients: ∂z/∂x = y, ∂z/∂y = x
        """
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
    
    def __pow__(self, other):
        """
        Power operation: z = x ** n
        Local gradient: ∂z/∂x = n * x^(n-1)
        """
        return Value(self.data ** other, (self,), (other * self.data ** (other - 1),))
    
    def log(self):
        """
        Natural logarithm: z = log(x)
        Local gradient: ∂z/∂x = 1/x
        """
        return Value(math.log(self.data), (self,), (1 / self.data,))
    
    def exp(self):
        """
        Exponential: z = exp(x)
        Local gradient: ∂z/∂x = exp(x)
        """
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    
    def relu(self):
        """
        ReLU activation: z = max(0, x)
        Local gradient: ∂z/∂x = 1 if x > 0 else 0
        """
        return Value(max(0, self.data), (self,), (float(self.data > 0),))
    
    # Operator overloads for reverse operations
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other ** -1
    def __rtruediv__(self, other): return other * self ** -1
    
    def backward(self):
        """
        Compute gradients using backpropagation.
        
        This performs a topological sort of the computation graph and
        propagates gradients backward using the chain rule.
        
        Chain Rule: ∂L/∂x = ∂L/∂y * ∂y/∂x
        
        Algorithm:
        ---------
        1. Build topological ordering of all nodes
        2. Set gradient of loss node to 1 (∂L/∂L = 1)
        3. Traverse nodes in reverse order
        4. For each node, propagate gradient to children
        
        Backward Pass Flow:
        -------------------
        
        ```mermaid
        graph TD
            A[Loss] -->|1.0| B[Layer 2]
            B -->|∂L/∂out| C[Layer 1]
            C -->|∂L/∂mid| D[Input]
        ```
        """
        # Step 1: Build topological ordering
        topo = []
        visited = set()
        
        def build_topological(v):
            """Recursively build topological order."""
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topological(child)
                topo.append(v)
        
        build_topological(self)
        
        # Step 2: Initialize gradient of loss (∂L/∂L = 1)
        self.grad = 1
        
        # Step 3: Backpropagate through graph
        for node in reversed(topo):
            for child, local_grad in zip(node._children, node._local_grads):
                # Chain rule: ∂L/∂child = ∂L/∂node * ∂node/∂child
                child.grad += local_grad * node.grad

print(f"\n[Autograd] Value class initialized for automatic differentiation")

# =============================================================================
# NEURAL NETWORK PRIMITIVES
# =============================================================================

def linear(x, w):
    """
    Linear transformation (matrix multiplication): y = xW
    
    Args:
        x: Input vector (list of Values)
        w: Weight matrix (list of lists of Values)
    
    Returns:
        list: Output vector (list of Values)
    
    Computation:
        y[j] = sum(x[i] * w[j][i] for i in range(len(x)))
    """
    return [sum(wi * xi for wi, xi in zip(w_row, x)) for w_row in w]

def softmax(logits):
    """
    Softmax activation: converts logits to probabilities.
    
    Softmax(x[i]) = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
    
    The max subtraction is for numerical stability (prevents overflow).
    
    Args:
        logits: List of Value objects (unnormalized log-probabilities)
    
    Returns:
        list: Probability distribution (sums to 1)
    
    Softmax Flow:
    -------------
    
    ```mermaid
    graph LR
        A[Logits: 2.0, 1.0, 0.1] --> B[Subtract Max: 1.9, 0.9, 0.0]
        B --> C[Exp: 6.7, 2.5, 1.0]
        C --> D[Divide by Sum: 0.66, 0.24, 0.10]
    ```
    """
    # Subtract max for numerical stability
    max_val = max(val.data for val in logits)
    
    # Exponentiate shifted values
    exps = [(val - max_val).exp() for val in logits]
    
    # Normalize to sum to 1
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    """
    RMS Normalization: normalizes vectors to unit variance.
    
    RMSNorm(x) = x * sqrt(mean(x^2) + eps)^(-1)
    
    This is a simpler alternative to LayerNorm with similar performance.
    
    Args:
        x: Input vector (list of Values)
    
    Returns:
        list: Normalized vector
    
    RMS Normalization:
    -----------------
    
    ```mermaid
    graph LR
        A[Input: 2, 4, 6] --> B[Square: 4, 16, 36]
        B --> C[Mean: 18.67]
        C --> D[Sqrt: 4.32]
        D --> E[Scale: 0.23]
        E --> F[Output: 0.46, 0.92, 1.39]
    ```
    """
    # Compute mean of squares
    mean_square = sum(xi * xi for xi in x) / len(x)
    
    # Compute scaling factor
    scale = (mean_square + 1e-5) ** -0.5
    
    # Scale all values
    return [xi * scale for xi in x]

print(f"[Primitives] Neural network functions defined")

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

def initialize_parameters(vocab_size, n_embd, n_head, n_layer):
    """
    Initialize all model parameters with Gaussian random values.
    
    The GPT model has the following parameter groups:
    1. Token embeddings (wte): Map token IDs to vectors
    2. Position embeddings (wpe): Add position information
    3. Attention weights (attn_wq, attn_wk, attn_wv, attn_wo): Query, Key, Value, Output
    4. MLP weights (mlp_fc1, mlp_fc2): Feed-forward network
    5. Language model head (lm_head): Output projection
    
    Model Architecture:
    ------------------
    
    ```mermaid
    graph TB
        A[Token IDs] --> B[wte: Token Embeddings]
        C[Positions] --> D[wpe: Position Embeddings]
        B --> E[Add + RMSNorm]
        D --> E
        E --> F[Transformer Layer 1]
        F --> G[Transformer Layer N]
        G --> H[lm_head: Output Projection]
        H --> I[Logits]
        
        F --> J[Multi-Head Attention]
        F --> K[Feed Forward Network]
    ```
    
    Args:
        vocab_size: Size of vocabulary
        n_embd: Embedding dimension
        n_head: Number of attention heads
        n_layer: Number of transformer layers
    
    Returns:
        dict: Dictionary of parameter matrices
    """
    # Helper function to create weight matrices
    def matrix(n_out, n_in, std=WEIGHT_INIT_STD):
        """Create a weight matrix with Gaussian initialization."""
        return [
            [Value(random.gauss(0, std)) for _ in range(n_in)]
            for _ in range(n_out)
        ]
    
    # Initialize parameters dictionary
    state_dict = {
        'wte': matrix(vocab_size, n_embd),      # Token embeddings
        'wpe': matrix(BLOCK_SIZE, n_embd),      # Position embeddings
        'lm_head': matrix(vocab_size, n_embd),  # Output projection
    }
    
    # Initialize transformer layers
    for layer_idx in range(n_layer):
        state_dict[f'layer{layer_idx}.attn_wq'] = matrix(n_embd, n_embd)  # Query projection
        state_dict[f'layer{layer_idx}.attn_wk'] = matrix(n_embd, n_embd)  # Key projection
        state_dict[f'layer{layer_idx}.attn_wv'] = matrix(n_embd, n_embd)  # Value projection
        state_dict[f'layer{layer_idx}.attn_wo'] = matrix(n_embd, n_embd)  # Output projection
        state_dict[f'layer{layer_idx}.mlp_fc1'] = matrix(4 * n_embd, n_embd)  # MLP expansion
        state_dict[f'layer{layer_idx}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)  # MLP contraction
    
    # Flatten all parameters into a single list for optimization
    params = [
        param 
        for matrix in state_dict.values() 
        for row in matrix 
        for param in row
    ]
    
    print(f"\n[Model] Initialized {len(params)} parameters")
    print(f"[Model] Architecture: {n_layer} layers × {n_head} heads × {n_embd} dimensions")
    
    return state_dict, params

# Initialize model parameters
state_dict, params = initialize_parameters(
    tokenizer.vocab_size, N_EMBD, N_HEAD, N_LAYER
)

def gpt_forward(token_id, pos_id, keys, values):
    """
    Forward pass through the GPT model.
    
    This function implements the complete GPT architecture including:
    1. Token and position embeddings
    2. Multi-head self-attention
    3. Feed-forward networks (MLP)
    4. Residual connections and normalization
    
    Transformer Layer Structure:
    ---------------------------
    
    ```mermaid
    graph TB
        A[Input x] --> B[RMSNorm]
        B --> C[Multi-Head Attention]
        C --> D[Residual Add: x + attn]
        D --> E[RMSNorm]
        E --> F[MLP: Linear → ReLU → Linear]
        F --> G[Residual Add: x + mlp]
        G --> H[Output]
    ```
    
    Multi-Head Attention:
    --------------------
    
    ```mermaid
    graph TB
        A[Input x] --> B[Linear: Wq → Q]
        A --> C[Linear: Wk → K]
        A --> D[Linear: Wv → V]
        E[Stored Keys] --> F[Attention Scores]
        B --> F
        F --> G[Attention Weights]
        G --> H[Weighted Values]
        D --> H
        H --> I[Linear: Wo → Output]
    ```
    
    Args:
        token_id: Current token ID
        pos_id: Current position in sequence
        keys: List of key vectors from previous positions (one list per layer)
        values: List of value vectors from previous positions (one list per layer)
    
    Returns:
        list: Logits over vocabulary for next token prediction
    """
    # 1. EMBEDDING LAYER
    # ------------------
    # Look up token embedding for current token
    tok_emb = state_dict['wte'][token_id]  # Shape: [n_embd]
    
    # Look up position embedding for current position
    pos_emb = state_dict['wpe'][pos_id]    # Shape: [n_embd]
    
    # Combine token and position embeddings
    x = [t + p for t, p in zip(tok_emb, pos_emb)]  # Shape: [n_embd]
    
    # Normalize embeddings
    x = rmsnorm(x)
    
    # 2. TRANSFORMER LAYERS
    # ---------------------
    for layer_idx in range(N_LAYER):
        # === RESIDUAL CONNECTION ===
        x_residual = x
        
        # === MULTI-HEAD SELF-ATTENTION ===
        # ---------------------------------
        # Normalize input
        x = rmsnorm(x)
        
        # Project to Query, Key, Value
        q = linear(x, state_dict[f'layer{layer_idx}.attn_wq'])  # Query
        k = linear(x, state_dict[f'layer{layer_idx}.attn_wk'])  # Key
        v = linear(x, state_dict[f'layer{layer_idx}.attn_wv'])  # Value
        
        # Store key and value for future positions to attend to
        keys[layer_idx].append(k)
        values[layer_idx].append(v)
        
        # Process each attention head
        x_attn = []
        for head_idx in range(N_HEAD):
            # Calculate head boundaries
            head_start = head_idx * HEAD_DIM
            head_end = head_start + HEAD_DIM
            
            # Extract head-specific Q, K, V
            q_head = q[head_start:head_end]  # Head query
            
            # Get all keys and values for this head from previous positions
            k_head = [k_vec[head_start:head_end] for k_vec in keys[layer_idx]]
            v_head = [v_vec[head_start:head_end] for v_vec in values[layer_idx]]
            
            # Compute attention scores
            # score[j][t] = (q_head · k_head[t]) / sqrt(d_k)
            attn_logits = [
                sum(
                    q_head[h] * k_head[t][h]
                    for h in range(HEAD_DIM)
                ) / (HEAD_DIM ** 0.5)
                for t in range(len(k_head))
            ]
            
            # Convert scores to probabilities (attention weights)
            attn_weights = softmax(attn_logits)
            
            # Compute weighted sum of values
            # out[h] = sum(attention_weights[t] * value[t][h])
            head_out = [
                sum(
                    attn_weights[t] * v_head[t][h]
                    for t in range(len(v_head))
                )
                for h in range(HEAD_DIM)
            ]
            
            # Concatenate head outputs
            x_attn.extend(head_out)
        
        # Project attention output
        x = linear(x_attn, state_dict[f'layer{layer_idx}.attn_wo'])
        
        # Residual connection
        x = [a + b for a, b in zip(x, x_residual)]
        
        # === FEED-FORWARD NETWORK (MLP) ===
        # ----------------------------------
        x_residual = x
        
        # Normalize
        x = rmsnorm(x)
        
        # First linear layer (expand by 4x)
        x = linear(x, state_dict[f'layer{layer_idx}.mlp_fc1'])
        
        # ReLU activation
        x = [xi.relu() for xi in x]
        
        # Second linear layer (contract back)
        x = linear(x, state_dict[f'layer{layer_idx}.mlp_fc2'])
        
        # Residual connection
        x = [a + b for a, b in zip(x, x_residual)]
    
    # 3. OUTPUT PROJECTION
    # --------------------
    # Project to vocabulary logits
    logits = linear(x, state_dict['lm_head'])
    
    return logits

print(f"[Model] GPT architecture defined")

# =============================================================================
# ADAM OPTIMIZER
# =============================================================================

class AdamOptimizer:
    """
    Adam optimizer with learning rate decay.
    
    Adam combines two gradient descent methods:
    1. Momentum: Keeps moving average of gradients (first moment)
    2. RMSprop: Keeps moving average of squared gradients (second moment)
    
    Adam Update Rule:
    -----------------
    m_t = β1 * m_{t-1} + (1 - β1) * g_t       # First moment (mean)
    v_t = β2 * v_{t-1} + (1 - β2) * g_t^2     # Second moment (variance)
    m̂_t = m_t / (1 - β1^t)                   # Bias correction
    v̂_t = v_t / (1 - β2^t)                   # Bias correction
    θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)     # Parameter update
    
    Adam Algorithm:
    --------------
    
    ```mermaid
    graph TD
        A[Gradient g_t] --> B[Update m_t: β1*m_{t-1} + 1-β1*g_t]
        A --> C[Update v_t: β2*v_{t-1} + 1-β2*g_t^2]
        B --> D[Bias correction: m̂_t = m_t / 1-β1^t]
        C --> E[Bias correction: v̂_t = v_t / 1-β2^t]
        D --> F[Compute update: m̂_t / √v̂_t + ε]
        E --> F
        F --> G[Apply update: θ_t = θ_{t-1} - lr * update]
    ```
    
    Attributes:
        params: List of model parameters to optimize
        lr: Learning rate
        beta1: First moment decay rate
        beta2: Second moment decay rate
        eps: Small constant for numerical stability
        m: First moment buffer (momentum)
        v: Second moment buffer (RMSprop)
    """
    
    def __init__(self, params, lr=LEARNING_RATE, beta1=BETA1, beta2=BETA2, eps=EPS_ADAM):
        """
        Initialize Adam optimizer.
        
        Args:
            params: List of parameters to optimize
            lr: Learning rate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            eps: Small constant for numerical stability
        """
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [0.0] * len(params)  # First moment
        self.v = [0.0] * len(params)  # Second moment
        self.t = 0  # Timestep counter
        
        print(f"\n[Optimizer] Adam initialized")
        print(f"[Optimizer] Learning rate: {lr}")
        print(f"[Optimizer] Beta1: {beta1}, Beta2: {beta2}")
    
    def step(self, step_num, total_steps):
        """
        Perform one optimization step.
        
        Args:
            step_num: Current step number (for learning rate decay)
            total_steps: Total number of training steps
        """
        self.t += 1
        
        # Linear learning rate decay
        lr_t = self.lr * (1 - step_num / total_steps)
        
        # Update each parameter
        for i, param in enumerate(self.params):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            
            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)
            
            # Compute bias-corrected estimates
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameter
            param.data -= lr_t * m_hat / (v_hat ** 0.5 + self.eps)
            
            # Reset gradient for next iteration
            param.grad = 0

# Initialize optimizer
optimizer = AdamOptimizer(params)

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def compute_loss(model_outputs, targets):
    """
    Compute cross-entropy loss for language modeling.
    
    Cross-entropy loss measures the difference between predicted
    probability distribution and true distribution.
    
    Loss = -log(probability of correct token)
    
    Args:
        model_outputs: List of predicted probability distributions
        targets: List of true target token IDs
    
    Returns:
        Value: Average loss over sequence
    """
    losses = []
    for output_probs, target_id in zip(model_outputs, targets):
        # Loss for this position: -log(prob[target_id])
        losses.append(-output_probs[target_id].log())
    
    # Average loss over sequence
    return (1 / len(losses)) * sum(losses)

def train_step(docs, tokenizer, step_num):
    """
    Perform one training step.
    
    Training Step Flow:
    -----------------
    
    ```mermaid
    graph TD
        A[Sample Document] --> B[Tokenize with BOS]
        B --> C[Forward Pass through Model]
        C --> D[Compute Probabilities]
        D --> E[Compute Loss]
        E --> F[Backward Pass]
        F --> G[Compute Gradients]
        G --> H[Update Parameters with Adam]
    ```
    
    Args:
        docs: Training dataset
        tokenizer: Tokenizer instance
        step_num: Current step number
    
    Returns:
        float: Loss value for this step
    """
    # Sample document (cycle through dataset)
    doc = docs[step_num % len(docs)]
    
    # Tokenize document
    tokens = tokenizer.encode(doc)
    
    # Limit sequence length
    seq_len = min(BLOCK_SIZE, len(tokens) - 1)
    
    # Initialize key and value caches for each layer
    keys = [[] for _ in range(N_LAYER)]
    values = [[] for _ in range(N_LAYER)]
    
    # Forward pass through sequence
    outputs = []
    for pos_id in range(seq_len):
        token_id = tokens[pos_id]
        target_id = tokens[pos_id + 1]
        
        # Forward pass
        logits = gpt_forward(token_id, pos_id, keys, values)
        
        # Convert to probabilities
        probs = softmax(logits)
        
        outputs.append(probs)
    
    # Compute loss
    targets = [tokens[pos_id + 1] for pos_id in range(seq_len)]
    loss = compute_loss(outputs, targets)
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    optimizer.step(step_num, NUM_STEPS)
    
    return loss

print(f"\n[Training] Training functions defined")

# =============================================================================
# TRAINING LOOP
# =============================================================================

print("\n" + "=" * 80)
print("TRAINING LOOP")
print("=" * 80)

for step in range(NUM_STEPS):
    # Perform training step
    loss = train_step(docs, tokenizer, step)
    
    # Print progress
    if (step + 1) % 100 == 0 or step == 0:
        print(f"Step {step+1:4d} / {NUM_STEPS:4d} | Loss: {loss.data:.4f}")

print("\n[Training] Training completed!")

# =============================================================================
# INFERENCE / TEXT GENERATION
# =============================================================================

def generate_text(tokenizer, max_length=BLOCK_SIZE, temperature=TEMPERATURE):
    """
    Generate text by sampling from the model.
    
    Text Generation Process:
    -----------------------
    
    ```mermaid
    graph TD
        A[Start with BOS token] --> B[Forward Pass]
        B --> C[Get Logits]
        C --> D[Apply Temperature]
        D --> E[Softmax → Probabilities]
        E --> F[Sample Token]
        F --> G{Is BOS?}
        G -->|Yes| H[Stop]
        G -->|No| I[Append Token]
        I --> B
    ```
    
    Args:
        tokenizer: Tokenizer instance
        max_length: Maximum generation length
        temperature: Sampling temperature (lower = more deterministic)
    
    Returns:
        str: Generated text
    """
    # Initialize key and value caches
    keys = [[] for _ in range(N_LAYER)]
    values = [[] for _ in range(N_LAYER)]
    
    # Start with BOS token
    token_id = tokenizer.bos_token_id
    generated_tokens = []
    
    # Generate tokens autoregressively
    for pos_id in range(max_length):
        # Forward pass
        logits = gpt_forward(token_id, pos_id, keys, values)
        
        # Apply temperature (lower temperature → more deterministic)
        scaled_logits = [l / temperature for l in logits]
        
        # Convert to probabilities
        probs = softmax(scaled_logits)
        
        # Sample token from probability distribution
        token_id = random.choices(
            range(tokenizer.vocab_size),
            weights=[p.data for p in probs]
        )[0]
        
        # Stop if we generated BOS token (end marker)
        if token_id == tokenizer.bos_token_id:
            break
        
        # Append token
        generated_tokens.append(token_id)
    
    # Decode tokens to text
    return tokenizer.decode(generated_tokens)

# =============================================================================
# GENERATE SAMPLES
# =============================================================================

print("\n" + "=" * 80)
print("GENERATING SAMPLES")
print("=" * 80)
print(f"\n[Generation] Temperature: {TEMPERATURE}")
print("[Generation] Generating 20 samples...\n")

for sample_idx in range(20):
    generated = generate_text(tokenizer)
    print(f"Sample {sample_idx + 1:2d}: {generated}")

print("\n" + "=" * 80)
print("GENERATION COMPLETE")
print("=" * 80)