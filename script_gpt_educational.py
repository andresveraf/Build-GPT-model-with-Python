"""
================================================================================
DEEP LEARNING FROM SCRATCH: Building a GPT in Pure Python
================================================================================

📚 EDUCATIONAL IMPLEMENTATION OF A GPT MODEL

This code is designed as a complete learning journey through deep learning.
Every concept is explained with intuition, mathematics, and implementation.

🎯 LEARNING OBJECTIVES:
    After studying this code, you will understand:
    1. How neural networks compute (forward pass)
    2. How they learn (backward pass & automatic differentiation)
    3. How to optimize them (Adam optimizer & hyperparameters)
    4. The transformer architecture (attention is all you need)
    5. Language modeling and text generation

👨‍🏫 PROFESSOR'S TEACHING PHILOSOPHY:
    "The best way to learn deep learning is to understand the WHY behind 
     every decision. We don't just implement algorithms - we understand 
     the intuition, the mathematics, and the practical trade-offs."

Based on Andrej Karpathy's minimal GPT implementation
Enhanced with educational explanations and workflow visualization
"""

print("=" * 80)
print("DEEP LEARNING FROM SCRATCH: GPT Implementation")
print("=" * 80)
print("\n📚 Welcome! Let's build a GPT model step by step, understanding every detail.")
print("   This implementation includes detailed workflow prints to help you")
print("   visualize what's happening at each stage of training and inference.\n")

# =============================================================================
# PART 1: FOUNDATIONS - IMPORTS AND RANDOM SEED
# =============================================================================

import os       # For file operations
import math     # For mathematical functions (log, exp, sqrt)
import random   # For randomness in initialization and sampling

"""
🎲 WHY SET A RANDOM SEED?
━━━━━━━━━━━━━━━━━━━━━━━━━━
Reproducibility! In research and debugging, we want the same results
every time we run the code. This is crucial for:
  1. Debugging: If there's a bug, we can reproduce it
  2. Comparing experiments: Same initialization = fair comparison
  3. Educational consistency: Everyone sees the same learning process

The number 42 is arbitrary (a reference to "The Hitchhiker's Guide to the Galaxy")
but any fixed number works.
"""

random.seed(42)

print(f"✓ Random seed set to 42 for reproducibility")

# =============================================================================
# PART 2: HYPERPARAMETERS - THE "KNOBS" OF DEEP LEARNING
# =============================================================================

print("\n" + "=" * 80)
print("PART 2: CONFIGURING THE MODEL - HYPERPARAMETERS")
print("=" * 80)

"""
📐 ARCHITECTURE HYPERPARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
These define the STRUCTURE of our neural network. Think of them as the
blueprint for building the model.

Why N_EMBD = 16?
  - In practice: 512, 1024, or more for real models
  - Here: Small for educational purposes (faster training)
  - Each token becomes a 16-dimensional vector
  - Higher = more capacity to learn patterns, but slower

Why N_HEAD = 4?
  - Multi-head attention lets the model focus on different aspects
  - Head 1 might learn: "pay attention to previous letter"
  - Head 2 might learn: "pay attention to letter position"
  - Head 3 might learn: "detect consonant patterns"
  - Head 4 might learn: "detect vowel patterns"
  - Each head is independent, so they can learn different things

Why N_LAYER = 1?
  - More layers = deeper understanding (more abstraction)
  - Layer 1: Simple patterns (bigrams)
  - Layer 2: Medium patterns (trigrams, phonemes)
  - Layer 3+: Complex patterns (morphology, semantics)
  - We use 1 for simplicity, but GPT-2 uses 12, GPT-3 uses 96!

Why BLOCK_SIZE = 16?
  - Maximum context window: how many previous tokens we can "remember"
  - Shorter = faster training, but limited context
  - Longer = richer context, but more computation (quadratic!)
  - Real models use 1024, 2048, or more

📊 TRAINING HYPERPARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━
These control HOW the model learns.

Why LEARNING_RATE = 0.01?
  - Learning rate is the "step size" in gradient descent
  - Too large (0.1): Might overshoot the minimum, diverge
  - Too small (0.0001): Takes forever to learn
  - 0.01 is moderate for this small model
  - In practice: Use learning rate scheduling (warmup + decay)

Why BETA1 = 0.85 and BETA2 = 0.99?
  - These are for the Adam optimizer's momentum
  - β1 controls the "momentum" (first moment - mean of gradients)
    * 0.85 = moderately fast adaptation
    * Closer to 1 = more smoothing, but slower to adapt
    * Standard is 0.9, we use 0.85 for faster learning
  - β2 controls the "adaptive learning rate" (second moment - variance)
    * 0.99 = very slow adaptation (moving average of squared gradients)
    * This makes the learning rate more stable
    * Standard is 0.999, we use 0.99 for faster adaptation

Why EPS_ADAM = 1e-8?
  - Tiny value to prevent division by zero
  - In Adam, we divide by sqrt(variance + eps)
  - Without eps: If variance is 0, we'd crash!
  - 1e-8 is small enough not to affect learning but prevents crashes

Why WEIGHT_INIT_STD = 0.08?
  - Standard deviation for random weight initialization
  - Small enough to prevent exploding activations
  - Large enough to prevent vanishing gradients
  - Rule of thumb: 1/sqrt(fan_in) = 1/sqrt(256) ≈ 0.06
  - 0.08 is slightly larger for better gradient flow

Why TEMPERATURE = 0.5?
  - Controls randomness in text generation
  - Temperature = 1.0: Sample from model's exact distribution
  - Temperature < 1.0: More deterministic, conservative
  - Temperature > 1.0: More random, creative (but risky)
  - 0.5 gives us reasonable diversity without too much nonsense
"""

# Model Architecture
N_EMBD = 16          # Embedding dimension
N_HEAD = 4           # Number of attention heads
N_LAYER = 1          # Number of transformer layers
BLOCK_SIZE = 16      # Maximum sequence length (context window)
HEAD_DIM = N_EMBD // N_HEAD  # Dimension per head = 4

# Training Configuration
LEARNING_RATE = 0.01
BETA1 = 0.85         # Adam momentum parameter (first moment)
BETA2 = 0.99         # Adam momentum parameter (second moment)
EPS_ADAM = 1e-8      # Numerical stability constant
NUM_STEPS = 1000     # Training iterations
TEMPERATURE = 0.5    # Sampling temperature

# Initialization
WEIGHT_INIT_STD = 0.08

print(f"\n📐 MODEL ARCHITECTURE:")
print(f"   • Embedding dimension: {N_EMBD}")
print(f"   • Attention heads: {N_HEAD} (each with {HEAD_DIM} dimensions)")
print(f"   • Transformer layers: {N_LAYER}")
print(f"   • Context window: {BLOCK_SIZE} tokens")

print(f"\n📊 TRAINING CONFIGURATION:")
print(f"   • Learning rate: {LEARNING_RATE}")
print(f"   • Adam β1 (momentum): {BETA1}")
print(f"   • Adam β2 (variance): {BETA2}")
print(f"   • Training steps: {NUM_STEPS}")
print(f"   • Sampling temperature: {TEMPERATURE}")

# =============================================================================
# PART 3: DATA LOADING - THE TRAINING SET
# =============================================================================

print("\n" + "=" * 80)
print("PART 3: DATA LOADING - WHERE KNOWLEDGE COMES FROM")
print("=" * 80)

def load_dataset(filename='input.txt', url=None):
    """
    Load or download the training dataset.
    
    📚 WHY THIS DATASET?
    ━━━━━━━━━━━━━━━━━━━
    We're using a dataset of names (makemore). This is perfect for learning because:
    1. Small enough to train quickly
    2. Clear structure (names follow patterns)
    3. Fun to see what the model generates
    
    The model will learn:
    - Common letter combinations (th, ch, sh)
    - Name patterns (vowel-consonant structure)
    - Probabilistic rules (q is almost always followed by u)
    """
    # Download if needed
    if not os.path.exists(filename):
        print(f"\n📥 Dataset not found. Downloading from GitHub...")
        import urllib.request
        if url is None:
            url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
        urllib.request.urlretrieve(url, filename)
        print(f"✓ Downloaded to {filename}")
    
    # Load and parse
    print(f"\n📖 Loading dataset from {filename}...")
    docs = [
        line.strip() 
        for line in open(filename).read().strip().split('\n') 
        if line.strip()
    ]
    
    # Shuffle for randomness
    random.shuffle(docs)
    
    # Show examples
    print(f"✓ Loaded {len(docs)} names")
    print(f"\n📝 Example names from dataset:")
    for i, name in enumerate(docs[:5]):
        print(f"   {i+1}. {name}")
    
    return docs

docs = load_dataset()

# =============================================================================
# PART 4: TOKENIZATION - CONVERTING TEXT TO NUMBERS
# =============================================================================

print("\n" + "=" * 80)
print("PART 4: TOKENIZATION - BRIDGING HUMANS AND MACHINES")
print("=" * 80)

class Tokenizer:
    """
    Converts between text strings and integer token IDs.
    
    🔄 WHY TOKENIZE?
    ━━━━━━━━━━━━━━━━
    Neural networks work with NUMBERS, not text. We need to convert:
    1. Text → Tokens: "hello" → [8, 5, 12, 12, 15]
    2. Tokens → Text: [8, 5, 12, 12, 15] → "hello"
    
    🔢 CHARACTER-LEVEL TOKENIZATION
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    We use character-level tokenization (not word-level or subword):
    - Simpler for educational purposes
    - Each character becomes a unique ID
    - Vocabulary size = number of unique characters + 1
    
    🏷️ SPECIAL TOKEN: BOS
    ━━━━━━━━━━━━━━━━━━━━━━━
    BOS = Beginning of Sequence
    - Marks the start and end of sequences
    - Helps the model learn when to stop generating
    - Also serves as EOS (End of Sequence) token
    
    Example:
    Input: "emma"
    With BOS: 26, 5, 13, 13, 1, 26
    Decode: "emma" (removes BOS tokens)
    """
    
    def __init__(self, docs):
        """Build vocabulary from training data."""
        # Extract unique characters
        self.uchars = sorted(set(''.join(docs)))
        
        # BOS token ID (last one)
        self.bos_token_id = len(self.uchars)
        
        # Total vocabulary size
        self.vocab_size = len(self.uchars) + 1
        
        print(f"\n📚 BUILT VOCABULARY:")
        print(f"   • Vocabulary size: {self.vocab_size} tokens")
        print(f"   • Characters: {''.join(self.uchars)}")
        print(f"   • BOS token ID: {self.bos_token_id} (special marker)")
        
        # Show encoding examples
        print(f"\n🔄 TOKENIZATION EXAMPLES:")
        example_names = docs[:3]
        for name in example_names:
            encoded = self.encode(name)
            print(f"   '{name}' → {encoded}")
    
    def encode(self, text):
        """Convert text to token IDs with BOS markers."""
        tokens = [self.uchars.index(ch) for ch in text]
        return [self.bos_token_id] + tokens + [self.bos_token_id]
    
    def decode(self, token_ids):
        """Convert token IDs back to text (removes BOS)."""
        return ''.join([
            self.uchars[tid] 
            for tid in token_ids 
            if tid != self.bos_token_id
        ])

tokenizer = Tokenizer(docs)

# =============================================================================
# PART 5: AUTOGRAD - THE MAGIC OF AUTOMATIC DIFFERENTIATION
# =============================================================================

print("\n" + "=" * 80)
print("PART 5: AUTOGRAD - HOW NEURAL NETWORKS LEARN")
print("=" * 80)

print("""
🧠 THE CORE IDEA: COMPUTATION GRAPHS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every computation builds a graph. During backpropagation, we traverse
this graph backwards, computing gradients using the chain rule.

Example: Computing loss for a single prediction
    Input x → Linear layer → Activation → Softmax → Cross-entropy → Loss
    
Forward pass (compute output):
    We calculate the loss by going left to right
    
Backward pass (compute gradients):
    We calculate ∂Loss/∂∂weight by going right to left
    Chain rule: ∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w

🎯 WHY THIS MATTERS:
    Without autograd, we'd have to manually compute derivatives for
    millions of parameters. Autograd automates this perfectly!

📊 CHAIN RULE IN PRACTICE:
    If y = f(x) and L = g(y), then:
    ∂L/∂x = ∂L/∂y × ∂y/∂x
    
    This lets us propagate gradients backward through any computation!
""")

class Value:
    """
    A scalar value that supports automatic differentiation.
    
    🎮 THINK OF IT AS:
    Each Value is a node in a computation graph. It knows:
    1. Its value (data)
    2. Which operations created it (_children)
    3. How to compute gradients (_local_grads)
    
    💡 KEY INSIGHT:
    When we compute z = x + y, we create a new Value z that:
    - Stores the result: z.data = x.data + y.data
    - Remembers its parents: z._children = (x, y)
    - Knows local gradients: z._local_grads = (1, 1)
      (because ∂z/∂x = 1 and ∂z/∂y = 1)
    
    Later, when we call z.backward(), we can compute gradients for x and y!
    """
    
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    
    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # Forward pass value
        self.grad = 0                   # Backward pass gradient (initialized to 0)
        self._children = children       # Dependencies
        self._local_grads = local_grads # Local derivatives
    
    def __add__(self, other):
        """
        Addition: z = x + y
        Local gradients: ∂z/∂x = 1, ∂z/∂y = 1
        
        💡 WHY 1?
        If z = x + y, then:
        ∂z/∂x = 1 (derivative of x is 1, y is constant)
        ∂z/∂y = 1 (derivative of y is 1, x is constant)
        """
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))
    
    def __mul__(self, other):
        """
        Multiplication: z = x * y
        Local gradients: ∂z/∂x = y, ∂z/∂y = x
        
        💡 WHY y AND x?
        If z = x * y, then:
        ∂z/∂x = y (derivative of x*y with respect to x is y)
        ∂z/∂y = x (derivative of x*y with respect to y is x)
        
        This is why we need to remember the values during forward pass!
        """
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
    
    def __pow__(self, other):
        """
        Power: z = x^n
        Local gradient: ∂z/∂x = n * x^(n-1)
        
        📐 DERIVATIVE RULE:
        d/dx(x^n) = n * x^(n-1)
        Example: d/dx(x^2) = 2x, d/dx(x^3) = 3x^2
        """
        return Value(self.data ** other, (self,), (other * self.data ** (other - 1),))
    
    def log(self):
        """
        Natural logarithm: z = log(x)
        Local gradient: ∂z/∂x = 1/x
        
        📐 DERIVATIVE RULE:
        d/dx(log(x)) = 1/x
        
        💡 WHY LOG IN LOSS?
        Log converts products to sums: log(a*b) = log(a) + log(b)
        This makes gradients more stable and prevents underflow.
        Also: -log(p) penalizes low probability heavily
        """
        return Value(math.log(self.data), (self,), (1 / self.data,))
    
    def exp(self):
        """
        Exponential: z = exp(x)
        Local gradient: ∂z/∂x = exp(x)
        
        📐 DERIVATIVE RULE:
        d/dx(exp(x)) = exp(x)
        
        The exponential function is its own derivative!
        """
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    
    def relu(self):
        """
        ReLU activation: z = max(0, x)
        Local gradient: ∂z/∂x = 1 if x > 0 else 0
        
        💡 WHY RELU?
        ReLU = Rectified Linear Unit
        - Simple: max(0, x)
        - Non-linear: Allows learning complex patterns
        - Sparse: Many neurons output 0 (efficient)
        - Gradient: 1 for positive inputs (no vanishing gradient!)
        
        📉 DERIVATIVE:
        If x > 0: ∂/∂x(max(0, x)) = 1
        If x < 0: ∂/∂x(max(0, x)) = 0
        At x = 0: Undefined (we use 0 by convention)
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
        
        🔄 THE ALGORITHM:
        1. Build topological order of all nodes (parents before children)
        2. Set gradient of loss node to 1 (∂L/∂L = 1, trivial)
        3. Traverse backwards (children to parents)
        4. For each node, propagate gradient to its children using chain rule
        
        📐 CHAIN RULE:
        ∂L/∂child = ∂L/∂node × ∂node/∂child
        
        Example:
        If L = f(g(h(x))), then:
        ∂L/∂x = ∂L/∂f × ∂f/∂g × ∂g/∂h × ∂h/∂x
        
        We compute this by traversing the graph backwards!
        """
        # Step 1: Build topological ordering
        topo = []
        visited = set()
        
        def build_topological(v):
            """Recursive DFS to build order."""
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topological(child)
                topo.append(v)
        
        build_topological(self)
        
        # Step 2: Initialize gradient (∂L/∂L = 1)
        self.grad = 1
        
        # Step 3: Backpropagate
        for node in reversed(topo):
            for child, local_grad in zip(node._children, node._local_grads):
                # Chain rule: accumulate gradients
                child.grad += local_grad * node.grad

print(f"✓ Autograd engine ready - can compute gradients automatically!")

# =============================================================================
# PART 6: NEURAL NETWORK PRIMITIVES
# =============================================================================

print("\n" + "=" * 80)
print("PART 6: BUILDING BLOCKS - LINEAR, SOFTMAX, NORMALIZATION")
print("=" * 80)

def linear(x, w):
    """
    Linear transformation: y = xW (matrix multiplication)
    
    📐 MATHEMATICAL DEFINITION:
        y[j] = Σ(i) x[i] × W[j][i]
    
    💡 WHY "LINEAR"?
        - It's a linear transformation (scaling + rotation)
        - No non-linearity yet (that comes later with ReLU)
        - This is the fundamental operation in neural networks
    
    🎯 PURPOSE:
        - Transform input features to output features
        - Learn patterns through weights W
        - Each weight represents connection strength
    
    📊 EXAMPLE:
        If x = [1, 2, 3] and W = [[4, 5, 6], [7, 8, 9]]:
        y[0] = 1×4 + 2×5 + 3×6 = 32
        y[1] = 1×7 + 2×8 + 3×9 = 50
        y = [32, 50]
    """
    return [sum(wi * xi for wi, xi in zip(w_row, x)) for w_row in w]

def softmax(logits):
    """
    Convert logits to probabilities using softmax.
    
    📐 MATHEMATICAL DEFINITION:
        softmax(x[i]) = exp(x[i]) / Σ(j) exp(x[j])
    
    💡 WHY SOFTMAX?
        - Converts any numbers to probabilities (sum to 1)
        - Preserves order: larger input → larger output
        - Exaggerates differences: winner gets higher probability
    
    🔢 NUMERICAL STABILITY TRICK:
        We subtract max before exponentiation:
        softmax(x[i]) = exp(x[i] - max(x)) / Σ exp(x[j] - max(x))
        
        WHY? Because exp(1000) = ∞ (overflow!), but exp(0) = 1
        By subtracting max, the largest value becomes exp(0) = 1
        This prevents overflow while preserving probabilities!
    
    📊 EXAMPLE:
        Input: [2.0, 1.0, 0.1]
        Max: 2.0
        Shifted: [0.0, -1.0, -1.9]
        Exp: [1.0, 0.368, 0.150]
        Sum: 1.518
        Output: [0.659, 0.242, 0.099]  (sums to 1.0)
    """
    # Numerical stability: subtract max
    max_val = max(val.data for val in logits)
    
    # Exponentiate (safe now because values are ≤ 0)
    exps = [(val - max_val).exp() for val in logits]
    
    # Normalize to sum to 1
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    """
    RMS (Root Mean Square) Normalization.
    
    📐 MATHEMATICAL DEFINITION:
        RMSNorm(x) = x × (mean(x²) + ε)^(-1/2)
    
    💡 WHY NORMALIZE?
        Problem: Without normalization, activations can grow or shrink
                 exponentially through layers, causing:
                 - Exploding gradients (too large)
                 - Vanishing gradients (too small)
        
        Solution: Normalize to unit variance at each layer
                 - Keeps values in reasonable range
                 - Stabilizes training
                 - Allows deeper networks
    
    🎯 RMS vs LAYER NORM:
        RMSNorm is simpler (doesn't subtract mean) but works similarly.
        It's preferred because it's faster and empirically works as well.
    
    📊 EXAMPLE:
        Input: [2, 4, 6]
        Squared: [4, 16, 36]
        Mean: 18.67
        Scale: (18.67 + 0.00001)^(-0.5) = 0.231
        Output: [0.462, 0.924, 1.386]  (now normalized!)
    """
    # Compute mean of squares
    mean_square = sum(xi * xi for xi in x) / len(x)
    
    # Compute scaling factor (add tiny epsilon to prevent div/0)
    scale = (mean_square + 1e-5) ** -0.5
    
    # Scale all values
    return [xi * scale for xi in x]

print(f"✓ Neural network primitives defined")
print(f"   • Linear: Matrix multiplication")
print(f"   • Softmax: Logits → Probabilities")
print(f"   • RMSNorm: Normalize activations")

# =============================================================================
# PART 7: MODEL ARCHITECTURE - THE TRANSFORMER
# =============================================================================

print("\n" + "=" * 80)
print("PART 7: THE TRANSFORMER - ATTENTION IS ALL YOU NEED")
print("=" * 80)

print("""
🧠 THE TRANSFORMER REVOLUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Transformers (2017) revolutionized AI because:
1. They can process all positions in parallel (unlike RNNs)
2. Self-attention lets them learn relationships between any positions
3. They scale incredibly well (GPT-3 has 175 BILLION parameters)

🎯 KEY INNOVATION: SELF-ATTENTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Traditional: Process left-to-right (like reading)
Transformer: Process all at once, with attention to what matters

Example: "The cat sat on the mat"
- When processing "cat", attention looks at "The" (article)
- When processing "sat", attention looks at "cat" (subject)
- When processing "mat", attention looks at "sat" (action)
- When processing "the" (second one), attention looks at "mat" (object)

This lets the model learn grammar and semantics!

🏗️ TRANSFORMER BLOCK STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input → RMSNorm → Multi-Head Attention → Residual → RMSNorm → MLP → Residual → Output

💡 RESIDUAL CONNECTIONS:
    x = x + Sublayer(x)
    
    WHY? They provide "gradient highways" for backpropagation.
    Without residuals, deep networks can't train (vanishing gradients).
    With residuals, gradients flow straight through!

📊 MULTI-HEAD ATTENTION:
    - Multiple "heads" process input in parallel
    - Each head learns different attention patterns
    - Results are concatenated and projected
    
    Analogy: Multiple experts looking at the same data,
              each focusing on different aspects.
""")

def initialize_parameters(vocab_size, n_embd, n_head, n_layer):
    """
    Initialize model parameters.
    
    🔢 PARAMETER GROUPS:
    ━━━━━━━━━━━━━━━━━━━━━
    1. Token embeddings (wte): Map token IDs → vectors
       - Each token gets a unique vector representation
       - Model learns semantic relationships (similar tokens ≈ similar vectors)
    
    2. Position embeddings (wpe): Add position information
       - "I am" vs "am I" - same words, different meaning!
       - Tells the model "this token is at position 5"
    
    3. Attention weights (attn_wq, attn_wk, attn_wv, attn_wo):
       - Wq: Query projection (what am I looking for?)
       - Wk: Key projection (what do I contain?)
       - Wv: Value projection (what information do I offer?)
       - Wo: Output projection (combine heads)
    
    4. MLP weights (mlp_fc1, mlp_fc2):
       - Feed-forward network for non-linear transformations
       - Expands by 4x, then contracts back (bottleneck)
       - Learns complex patterns
    
    5. Language model head (lm_head):
       - Projects back to vocabulary size
       - Predicts probability of next token
    
    🎲 INITIALIZATION STRATEGY:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━
    Gaussian with std=0.08
    - Small enough: Prevents exploding activations
    - Large enough: Prevents vanishing gradients
    - Random: Breaks symmetry (all neurons learn different things)
    """
    def matrix(n_out, n_in, std=WEIGHT_INIT_STD):
        """Create weight matrix with Gaussian initialization."""
        return [
            [Value(random.gauss(0, std)) for _ in range(n_in)]
            for _ in range(n_out)
        ]
    
    # Initialize all parameters
    state_dict = {
        'wte': matrix(vocab_size, n_embd),
        'wpe': matrix(BLOCK_SIZE, n_embd),
        'lm_head': matrix(vocab_size, n_embd),
    }
    
    for layer_idx in range(n_layer):
        state_dict[f'layer{layer_idx}.attn_wq'] = matrix(n_embd, n_embd)
        state_dict[f'layer{layer_idx}.attn_wk'] = matrix(n_embd, n_embd)
        state_dict[f'layer{layer_idx}.attn_wv'] = matrix(n_embd, n_embd)
        state_dict[f'layer{layer_idx}.attn_wo'] = matrix(n_embd, n_embd)
        state_dict[f'layer{layer_idx}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
        state_dict[f'layer{layer_idx}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
    
    # Flatten into single list
    params = [
        param 
        for matrix in state_dict.values() 
        for row in matrix 
        for param in row
    ]
    
    print(f"\n🎲 INITIALIZED MODEL PARAMETERS:")
    print(f"   • Total parameters: {len(params):,}")
    print(f"   • Token embeddings: {vocab_size} × {n_embd} = {vocab_size * n_embd:,}")
    print(f"   • Position embeddings: {BLOCK_SIZE} × {n_embd} = {BLOCK_SIZE * n_embd:,}")
    print(f"   • Per-layer attention: 4 × {n_embd} × {n_embd} = {4 * n_embd * n_embd:,}")
    print(f"   • Per-layer MLP: {4 * n_embd} × {n_embd} + {n_embd} × {4 * n_embd} = {8 * n_embd * n_embd:,}")
    
    return state_dict, params

state_dict, params = initialize_parameters(
    tokenizer.vocab_size, N_EMBD, N_HEAD, N_LAYER
)

def gpt_forward(token_id, pos_id, keys, values, verbose=False):
    """
    Forward pass through the GPT model.
    
    🌊 THE FORWARD PASS FLOW:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. Embedding: Look up token and position vectors
    2. Process through transformer layers
       a. Multi-head self-attention (learn relationships)
       b. Feed-forward MLP (process information)
    3. Project to vocabulary (predict next token)
    
    📍 MULTI-HEAD ATTENTION IN DETAIL:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    For each head h:
        1. Project input to Q, K, V (Query, Key, Value)
        2. Compute attention scores: score = Q · K / √d
        3. Convert to probabilities: weights = softmax(score)
        4. Weight values: output = weights · V
    
    💡 WHY Q, K, V?
    ━━━━━━━━━━━━━━━━
    Analogy: Library search
    - Query (Q): What you're looking for
    - Key (K): What each book contains (title, tags)
    - Value (V): The book's content
    
    Attention score = similarity(Query, Key)
    If score is high → retrieve that Value
    
    📏 WHY DIVIDE BY √d?
    ━━━━━━━━━━━━━━━━━━
    Prevents dot products from growing too large with dimension.
    Large scores → softmax saturates (all near 0 or 1) → no gradients!
    Dividing by √d keeps scores in reasonable range.
    
    Args:
        token_id: Current token to process
        pos_id: Current position in sequence
        keys: Cached keys from previous positions
        values: Cached values from previous positions
        verbose: Print detailed information (for learning)
    
    Returns:
        Logits over vocabulary for next token prediction
    """
    if verbose and pos_id == 0:
        print(f"\n🔍 FORWARD PASS - Position {pos_id}")
        print(f"   Token ID: {token_id} ('{tokenizer.decode([token_id])}')")
    
    # 1. EMBEDDING LAYER
    # ──────────────────
    # Look up embeddings
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    
    # Combine them (addition merges information)
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    
    # Normalize
    x = rmsnorm(x)
    
    if verbose and pos_id == 0:
        print(f"   Embeddings combined and normalized")
    
    # 2. TRANSFORMER LAYERS
    # ─────────────────────
    for layer_idx in range(N_LAYER):
        # === RESIDUAL CONNECTION ===
        x_residual = x
        
        # === MULTI-HEAD ATTENTION ===
        x = rmsnorm(x)
        
        # Project to Q, K, V
        q = linear(x, state_dict[f'layer{layer_idx}.attn_wq'])
        k = linear(x, state_dict[f'layer{layer_idx}.attn_wk'])
        v = linear(x, state_dict[f'layer{layer_idx}.attn_wv'])
        
        # Store for future positions
        keys[layer_idx].append(k)
        values[layer_idx].append(v)
        
        # Process each head
        x_attn = []
        for head_idx in range(N_HEAD):
            head_start = head_idx * HEAD_DIM
            head_end = head_start + HEAD_DIM
            
            # Extract head-specific Q, K, V
            q_head = q[head_start:head_end]
            k_head = [k_vec[head_start:head_end] for k_vec in keys[layer_idx]]
            v_head = [v_vec[head_start:head_end] for v_vec in values[layer_idx]]
            
            # Attention scores
            attn_logits = [
                sum(
                    q_head[h] * k_head[t][h]
                    for h in range(HEAD_DIM)
                ) / (HEAD_DIM ** 0.5)
                for t in range(len(k_head))
            ]
            
            # Attention weights
            attn_weights = softmax(attn_logits)
            
            if verbose and pos_id == 0 and head_idx == 0:
                weights_list = [w.data for w in attn_weights[:5]]
                print(f"   Head {head_idx} attention weights: {weights_list}")
            
            # Weighted sum of values
            head_out = [
                sum(
                    attn_weights[t] * v_head[t][h]
                    for t in range(len(v_head))
                )
                for h in range(HEAD_DIM)
            ]
            
            x_attn.extend(head_out)
        
        # Project attention output
        x = linear(x_attn, state_dict[f'layer{layer_idx}.attn_wo'])
        
        # Residual
        x = [a + b for a, b in zip(x, x_residual)]
        
        # === MLP ===
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{layer_idx}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{layer_idx}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
    
    # 3. OUTPUT PROJECTION
    # ────────────────────
    logits = linear(x, state_dict['lm_head'])
    
    if verbose and pos_id == 0:
        print(f"   Output logits computed")
    
    return logits

print(f"✓ GPT architecture defined")

# =============================================================================
# PART 8: THE ADAM OPTIMIZER
# =============================================================================

print("\n" + "=" * 80)
print("PART 8: THE ADAM OPTIMIZER - ADAPTIVE LEARNING")
print("=" * 80)

print("""
🎯 WHY ADAM? (ADAPTIVE MOMENT ESTIMATION)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Plain SGD (Stochastic Gradient Descent) problems:
1. Fixed learning rate: Too slow in some directions, too fast in others
2. No momentum: Gets stuck in local minima, slow in flat regions

Adam solves both:
1. Momentum: Like a heavy ball rolling down a hill
   - Accumulates past gradients
   - Helps escape local minima
   - Speeds up convergence

2. Adaptive learning rates: Different LR for each parameter
   - Frequently updated parameters → smaller LR
   - Rarely updated parameters → larger LR
   - Automatically balances learning

📊 ADAM VS SGD EXAMPLE:
━━━━━━━━━━━━━━━━━━━━━━━
Problem: Minimize f(x,y) = 1000x² + y²
- SGD gets stuck (needs tiny LR for x, but then y learns slowly)
- Adam adapts: large LR for y, small LR for x
- Adam converges faster!

🧮 THE ALGORITHM:
━━━━━━━━━━━━━━━━━
For each parameter θ with gradient g:

1. Update first moment (momentum):
   m_t = β1 × m_{t-1} + (1-β1) × g_t
   
   β1 = 0.85 means: keep 85% of past momentum, add 15% of current gradient

2. Update second moment (uncentered variance):
   v_t = β2 × v_{t-1} + (1-β2) × g_t²
   
   β2 = 0.99 means: keep 99% of past variance, add 1% of current squared gradient

3. Bias correction (important early in training):
   m̂_t = m_t / (1 - β1^t)
   v̂_t = v_t / (1 - β2^t)
   
   WHY? Early in training, m and v are initialized to 0
        They're biased toward 0, so we correct this bias

4. Update parameter:
   θ_t = θ_{t-1} - α × m̂_t / (√v̂_t + ε)
   
   α is learning rate (we decay it linearly)
   ε = 1e-8 prevents division by zero

💡 KEY INSIGHTS:
━━━━━━━━━━━━━━━━━
- m̂_t / √v̂_t is like: signal / noise
  - High gradient (high variance) → small step (noisy)
  - Consistent gradient (low variance) → large step (trustworthy)
  
- This is why Adam works so well in practice!
""")

class AdamOptimizer:
    """
    Adam optimizer with momentum and adaptive learning rates.
    
    🎯 HYPERPARAMETER EXPLANATIONS:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    lr = 0.01:
        Base learning rate. Adam multiplies this by m̂_t/√v̂_t for each parameter.
        Smaller = more stable but slower
        Larger = faster but might diverge
    
    beta1 = 0.85:
        Momentum decay rate. Controls how fast we forget past gradients.
        Higher (0.9, 0.99) = smoother momentum, slower to adapt
        Lower (0.8, 0.85) = faster adaptation, less smooth
        Standard is 0.9, we use 0.85 for faster learning
    
    beta2 = 0.99:
        Variance decay rate. Controls how fast we forget past squared gradients.
        Should be close to 1 (0.999 in paper, 0.99 here for speed)
        Higher = more stable adaptive LR, slower to adapt
    
    eps = 1e-8:
        Tiny constant to prevent division by zero.
        In practice: gradients are rarely exactly 0, so this is a safety net.
    
    📊 WHY BIAS CORRECTION?
    ━━━━━━━━━━━━━━━━━━━━━━
    m_t = β1 × m_{t-1} + (1-β1) × g_t
    
    If β1 = 0.85 and m_0 = 0:
    m_1 = 0.85 × 0 + 0.15 × g_1 = 0.15 × g_1
    m_2 = 0.85 × 0.15×g_1 + 0.15 × g_2 ≈ 0.1275×g_1 + 0.15×g_2
    
    Early m values are biased toward 0 (because we started at 0)
    Bias correction: m̂_t = m_t / (1 - β1^t)
    
    For t=1: m̂_1 = 0.15×g_1 / (1 - 0.85) = 0.15×g_1 / 0.15 = g_1 ✓
    
    This makes early training more accurate!
    """
    
    def __init__(self, params, lr=LEARNING_RATE, beta1=BETA1, beta2=BETA2, eps=EPS_ADAM):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [0.0] * len(params)  # First moment (momentum)
        self.v = [0.0] * len(params)  # Second moment (variance)
        self.t = 0  # Timestep
        
        print(f"\n📊 ADAM OPTIMIZER CONFIGURED:")
        print(f"   • Learning rate (α): {lr}")
        print(f"   • Momentum (β1): {beta1}")
        print(f"   • Variance (β2): {beta2}")
        print(f"   • Epsilon (ε): {eps}")
    
    def step(self, step_num, total_steps, verbose=False):
        """
        Perform one optimization step.
        
        🔄 WHAT HAPPENS:
        1. Update timestep (for bias correction)
        2. Compute decaying learning rate (linear decay)
        3. For each parameter:
           a. Update first moment (momentum)
           b. Update second moment (variance)
           c. Apply bias correction
           d. Compute adaptive learning rate
           e. Update parameter
           f. Reset gradient to 0
        """
        self.t += 1
        
        # Linear learning rate decay
        lr_t = self.lr * (1 - step_num / total_steps)
        
        if verbose:
            print(f"\n⚙️  OPTIMIZER STEP {step_num}:")
            print(f"   • Decayed learning rate: {lr_t:.6f}")
        
        # Update each parameter
        grad_magnitudes = []
        param_updates = []
        
        for i, param in enumerate(self.params):
            # Skip parameters with no gradient
            if param.grad == 0:
                continue
            
            # Update first moment (momentum)
            m_old = self.m[i]
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            
            # Update second moment (variance)
            v_old = self.v[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Compute update
            update = lr_t * m_hat / (v_hat ** 0.5 + self.eps)
            
            # Store old value
            old_value = param.data
            
            # Apply update
            param.data -= update
            
            # Reset gradient
            param.grad = 0
            
            # Track statistics (sample a few parameters)
            if i % 500 == 0:
                grad_magnitudes.append(abs(param.grad))
                param_updates.append(abs(old_value - param.data))
        
        if verbose and grad_magnitudes:
            avg_grad = sum(grad_magnitudes) / len(grad_magnitudes)
            avg_update = sum(param_updates) / len(param_updates)
            print(f"   • Avg gradient magnitude: {avg_grad:.6f}")
            print(f"   • Avg parameter update: {avg_update:.6f}")

# Initialize optimizer
optimizer = AdamOptimizer(params)

# =============================================================================
# PART 9: TRAINING - WHERE THE MODEL LEARNS
# =============================================================================

print("\n" + "=" * 80)
print("PART 9: TRAINING - LEARNING FROM DATA")
print("=" * 80)

print("""
🎓 THE TRAINING LOOP
━━━━━━━━━━━━━━━━━━━━━
Training is an iterative process:

FOR each training step:
    1. Sample a document from dataset
    2. Tokenize it (convert to numbers)
    3. Forward pass: predict next token for each position
    4. Compute loss: how wrong were our predictions?
    5. Backward pass: compute gradients (how to improve)
    6. Update parameters: move in direction that reduces loss

📉 CROSS-ENTROPY LOSS:
━━━━━━━━━━━━━━━━━━━━━━
For each prediction, we compute:
    Loss = -log(probability of correct token)

Example: If we predict 'a' (probability 0.7) but correct answer is 'e' (probability 0.1):
    Loss = -log(0.1) = 2.303
    
If we predict correctly (probability 0.9):
    Loss = -log(0.9) = 0.105  (much better!)

Average loss over all positions tells us how well we're doing.

🎯 WHAT THE MODEL LEARNS:
━━━━━━━━━━━━━━━━━━━━━━━━━
Through training, the model learns:
- Character patterns: 'q' is usually followed by 'u'
- Name structures: vowels and consonants alternate
- Phonetics: certain letter combinations make sense
- Semantics (for larger models): meaning and context

🔄 WHY CYCLE THROUGH DATASET?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
We use step % len(docs) to cycle through the dataset.
This ensures:
- Each document is seen multiple times (multiple epochs)
- Randomness from initial shuffle prevents overfitting to order
- More exposure to data = better learning
""")

def compute_loss(model_outputs, targets):
    """
    Compute cross-entropy loss.
    
    📐 MATHEMATICAL DEFINITION:
        Loss = -(1/N) × Σ(i) log(p_i[target_i])
    
    Where:
        - N: number of predictions
        - p_i: predicted probability distribution
        - target_i: correct token
    
    💡 WHY NEGATIVE LOG LIKELIHOOD?
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    We want to MAXIMIZE probability of correct answers.
    But optimization algorithms MINIMIZE functions.
    
    Solution: MINIMIZE negative log probability
    -max(log(p)) = min(-log(p))
    
    Also: -log(p) heavily penalizes low probabilities
          -log(0.01) = 4.6 (very bad)
          -log(0.5) = 0.69 (okay)
          -log(0.99) = 0.01 (great!)
    
    This matches our intuition: we want high confidence in correct answers!
    """
    losses = []
    for output_probs, target_id in zip(model_outputs, targets):
        # Loss for this position: -log(probability of correct token)
        losses.append(-output_probs[target_id].log())
    
    # Average over sequence
    return (1 / len(losses)) * sum(losses)

def train_step(docs, tokenizer, step_num, verbose=False):
    """
    Perform one training step with detailed workflow prints.
    
    📊 WORKFLOW:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    STEP 1: SAMPLE DATA
    ───────────────────
    Get a document from the dataset
    Example: "emma"
    
    STEP 2: TOKENIZE
    ────────────────
    Convert to token IDs with BOS markers
    [26, 5, 13, 13, 1, 26]
    
    STEP 3: FORWARD PASS (position by position)
    ────────────────────────────────────────────
    Position 0: Input BOS → Predict 'e'
    Position 1: Input 'e' → Predict 'm'
    Position 2: Input 'm' → Predict 'm'
    Position 3: Input 'm' → Predict 'a'
    Position 4: Input 'a' → Predict BOS
    
    STEP 4: COMPUTE LOSS
    ─────────────────────
    Compare predictions to actual next tokens
    High probability of correct token → Low loss
    Low probability of correct token → High loss
    
    STEP 5: BACKWARD PASS
    ──────────────────────
    Compute gradients: ∂Loss/∂∂parameter
    Uses chain rule through computation graph
    
    STEP 6: UPDATE PARAMETERS
    ─────────────────────────
    Adam optimizer updates all parameters
    Move in direction that reduces loss
    """
    # Sample document
    doc = docs[step_num % len(docs)]
    
    # Tokenize
    tokens = tokenizer.encode(doc)
    seq_len = min(BLOCK_SIZE, len(tokens) - 1)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"TRAINING STEP {step_num + 1}")
        print(f"{'='*60}")
        print(f"\n📄 Sample document: '{doc}'")
        print(f"   Tokens: {tokens[:seq_len+1]}")
        print(f"   Sequence length: {seq_len}")
        print(f"\n🎯 Training task:")
        print(f"   Given: {tokenizer.decode(tokens[:seq_len])}")
        print(f"   Predict: {tokenizer.decode([tokens[seq_len]])}")
    
    # Initialize caches
    keys = [[] for _ in range(N_LAYER)]
    values = [[] for _ in range(N_LAYER)]
    
    # Forward pass through sequence
    outputs = []
    for pos_id in range(seq_len):
        token_id = tokens[pos_id]
        target_id = tokens[pos_id + 1]
        
        # Forward pass
        logits = gpt_forward(token_id, pos_id, keys, values, verbose=verbose and pos_id == 0)
        
        # Convert to probabilities
        probs = softmax(logits)
        
        outputs.append(probs)
    
    # Compute loss
    targets = [tokens[pos_id + 1] for pos_id in range(seq_len)]
    loss = compute_loss(outputs, targets)
    
    if verbose:
        print(f"\n📊 Loss Computation:")
        print(f"   • Final loss value: {loss.data:.4f}")
        print(f"   • Perplexity: {math.exp(loss.data):.2f}")
        print(f"     (Perplexity = exp(loss), lower is better)")
        
        # Show some prediction examples
        print(f"\n🔮 Prediction Examples:")
        for i in range(min(3, seq_len)):
            pred_probs = outputs[i]
            target = targets[i]
            target_prob = pred_probs[target].data
            top_tokens = sorted(range(len(pred_probs)), key=lambda t: pred_probs[t].data, reverse=True)[:3]
            top_tokens_str = ', '.join([f"'{tokenizer.decode([t])}' ({pred_probs[t].data:.3f})" for t in top_tokens if t != tokenizer.bos_token_id])
            print(f"   Position {i}: Predict '{tokenizer.decode([target])}' with prob {target_prob:.3f}")
            print(f"               Top predictions: {top_tokens_str}")
    
    # Backward pass
    loss.backward()
    
    if verbose:
        # Show gradient statistics
        non_zero_grads = [p for p in params if p.grad != 0]
        if non_zero_grads:
            avg_grad = sum(abs(p.grad) for p in non_zero_grads) / len(non_zero_grads)
            max_grad = max(abs(p.grad) for p in non_zero_grads)
            print(f"\n📈 Gradient Statistics:")
            print(f"   • Parameters with gradients: {len(non_zero_grads)}")
            print(f"   • Average gradient magnitude: {avg_grad:.6f}")
            print(f"   • Maximum gradient magnitude: {max_grad:.6f}")
    
    # Update parameters
    optimizer.step(step_num, NUM_STEPS, verbose=verbose)
    
    return loss

# =============================================================================
# PART 10: THE TRAINING LOOP
# =============================================================================

print("\n" + "=" * 80)
print("PART 10: TRAINING THE MODEL")
print("=" * 80)

# Show detailed workflow for first few steps
verbose_steps = [0, 1, 50, 100, 500, 999]

for step in range(NUM_STEPS):
    verbose = step in verbose_steps
    
    # Train
    loss = train_step(docs, tokenizer, step, verbose=verbose)
    
    # Regular progress
    if not verbose:
        if (step + 1) % 100 == 0 or step == 0:
            print(f"Step {step+1:4d} / {NUM_STEPS:4d} | Loss: {loss.data:.4f} | Perplexity: {math.exp(loss.data):.2f}")
    elif step == verbose_steps[-1]:
        print(f"\n✓ Training completed!")

# =============================================================================
# PART 11: INFERENCE - GENERATING TEXT
# =============================================================================

print("\n" + "=" * 80)
print("PART 11: TEXT GENERATION - CREATIVITY IN ACTION")
print("=" * 80)

print("""
🎲 GENERATION STRATEGY: TEMPERATURE SAMPLING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
We don't just pick the most likely token (that's boring!).
We sample from the probability distribution with temperature.

🌡️ TEMPERATURE EFFECTS:
━━━━━━━━━━━━━━━━━━━━━━━
Temperature = 0.5 (what we use):
    - More conservative
    - Higher probabilities get boosted
    - Lower probabilities get suppressed
    - Result: Mostly sensible names, some variety
    
Temperature = 1.0:
    - Pure sampling from model's distribution
    - More random, less filtered
    - Result: More creative, but more nonsense
    
Temperature = 0.1:
    - Almost deterministic (pick most likely)
    - Result: Very conservative, repetitive

📊 MATHEMATICS:
━━━━━━━━━━━━━━━
p[i] = exp(logits[i] / T) / Σ exp(logits[j] / T)

When T < 1: Large logits become larger (amplify confidence)
When T > 1: Large logits become smaller (dampen confidence)

🎯 GENERATION ALGORITHM:
━━━━━━━━━━━━━━━━━━━━━━━
1. Start with BOS token (beginning marker)
2. Loop:
   a. Forward pass: get logits for next token
   b. Apply temperature
   c. Sample from probability distribution
   d. If sampled BOS: stop (end marker)
   e. Else: append and continue
3. Decode tokens to text
""")

def generate_text(tokenizer, max_length=BLOCK_SIZE, temperature=TEMPERATURE, show_first=False, sample_idx=0):
    """
    Generate text by sampling from the model.
    
    🎲 THE GENERATION PROCESS:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Step 1: Start
    ─────────────
    Initialize with BOS token
    Empty key/value caches
    
    Step 2: Autoregressive Generation Loop
    ────────────────────────────────────────
    FOR each position:
        a. Forward pass (model predicts next token)
        b. Get probability distribution (apply softmax)
        c. Sample token from distribution (weighted random choice)
        d. If BOS: STOP (end marker)
        e. Else: append and continue
    
    Step 3: Decode
    ───────────────
    Convert token IDs back to text
    
    💡 WHY AUTOREGRESSIVE?
    ━━━━━━━━━━━━━━━━━━━━━━
    Each token depends on ALL previous tokens.
    This is how the model learns context and structure!
    
    The key/value cache stores all previous positions,
    allowing the model to "remember" the entire sequence.
    
    🎲 SAMPLING VS GREEDY:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Greedy: Always pick highest probability
            Pros: Consistent, sensible
            Cons: Repetitive, boring
    
    Sampling: Pick randomly from distribution (weighted by probability)
            Pros: Creative, diverse
            Cons: Can generate nonsense
    
    Temperature sampling gives us the best of both!
    """
    # Initialize
    keys = [[] for _ in range(N_LAYER)]
    values = [[] for _ in range(N_LAYER)]
    token_id = tokenizer.bos_token_id
    generated_tokens = []
    
    if show_first and sample_idx == 0:
        print(f"\n🎲 GENERATION SAMPLE {sample_idx + 1} (VERBOSE):")
        print(f"   Temperature: {temperature}")
        print(f"   Max length: {max_length}")
    
    # Generate
    for pos_id in range(max_length):
        # Forward pass
        logits = gpt_forward(token_id, pos_id, keys, values)
        
        # Apply temperature
        scaled_logits = [l / temperature for l in logits]
        
        # Softmax to get probabilities
        probs = softmax(scaled_logits)
        
        # Sample
        token_id = random.choices(
            range(tokenizer.vocab_size),
            weights=[p.data for p in probs]
        )[0]
        
        # Show first sample in detail
        if show_first and sample_idx == 0 and pos_id < 5:
            top_probs = sorted(
                [(i, p.data) for i, p in enumerate(probs) if i != tokenizer.bos_token_id],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            top_str = ', '.join([f"'{tokenizer.decode([t])}'({p:.3f})" for t, p in top_probs])
            if token_id == tokenizer.bos_token_id:
                print(f"   Position {pos_id}: Generated BOS (stop)")
            else:
                print(f"   Position {pos_id}: Generated '{tokenizer.decode([token_id])}'")
                print(f"               Top probs: {top_str}")
        
        # Check for end
        if token_id == tokenizer.bos_token_id:
            if show_first and sample_idx == 0:
                print(f"   ✓ Generated BOS token - stopping")
            break
        
        generated_tokens.append(token_id)
    
    # Decode
    text = tokenizer.decode(generated_tokens)
    
    if show_first and sample_idx == 0:
        print(f"\n   Final result: '{text}'")
        print(f"   Length: {len(generated_tokens)} tokens")
    
    return text

# Generate samples
print(f"\n🎲 GENERATING {20} SAMPLES:")
print(f"   Temperature: {TEMPERATURE}")
print(f"   This will take a moment...\n")

# Show first sample in detail
samples = []
for i in range(20):
    text = generate_text(tokenizer, show_first=(i == 0), sample_idx=i)
    samples.append(text)
    if i > 0:  # First one already printed
        print(f"Sample {i+1:2d}: {text}")

print(f"\n{'='*80}")
print(f"GENERATION COMPLETE")
print(f"{'='*80}")

# Show summary statistics
print(f"\n📊 GENERATION STATISTICS:")
lengths = [len(s) for s in samples]
print(f"   • Average length: {sum(lengths)/len(lengths):.1f} characters")
print(f"   • Min length: {min(lengths)} characters")
print(f"   • Max length: {max(lengths)} characters")
print(f"   • Unique samples: {len(set(samples))} / {len(samples)}")

print(f"\n🎓 EDUCATIONAL SUMMARY:")
print(f"   You've just seen a complete GPT implementation!")
print(f"   Key takeaways:")
print(f"   1. Neural networks learn through gradient descent")
print(f"   2. Transformers use attention to learn relationships")
print(f"   3. Adam optimizer adapts learning rates per parameter")
print(f"   4. Temperature sampling balances creativity and quality")
print(f"   5. Every decision (hyperparameters) has a reason!")

print(f"\n📚 FURTHER LEARNING:")
print(f"   • 'Attention Is All You Need' (Vaswani et al., 2017)")
print(f"   • 'Language Models are Few-Shot Learners' (GPT-3 paper)")
print(f"   • Andrej Karpathy's YouTube channel (excellent tutorials)")

print(f"\n{'='*80}")
print(f"END OF EDUCATIONAL GPT IMPLEMENTATION")
print(f"{'='*80}\n")