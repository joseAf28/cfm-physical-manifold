import torch 
import torch.nn as nn
import math


def build_model(config):
    model_name = config.get('model_name')
    model_cfg = config.get('model', {})
    
    # Extract dimensions (defaulting to your specific dataset dimensions)
    x_dim = model_cfg.get('x_dim', 3)
    y_dim = model_cfg.get('y_dim', 17)
    
    if model_name == 'CFM_t':
        return CFM_t(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dim=model_cfg.get('hidden_dim', 40),
            cond_dim=model_cfg.get('cond_dim', 32),
            num_blocks=model_cfg.get('num_blocks', 2)
        )
        
    elif model_name == 'CFM_0':
        return CFM_0(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dim=model_cfg.get('hidden_dim', 40),
            cond_dim=model_cfg.get('cond_dim', 32),
            num_blocks=model_cfg.get('num_blocks', 2)
        )
        
    elif model_name in ['NN', 'NN_Soft']:
        # The baselines share the Regressor architecture, but NN_Soft applies physics penalties during training
        return Regressor(
            input_size=x_dim,
            hidden_sizes=model_cfg.get('hidden_size', [130, 130]),
            output_size=y_dim
        )
        
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4, min_epochs=350):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, current_loss, epoch):
        if epoch < self.min_epochs:
            if current_loss + self.min_delta < self.best_loss:
                self.best_loss = current_loss
                self.counter = 0
            return False
        if current_loss + self.min_delta < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def step(self):
        """Update shadow weights with current model weights."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].sub_((1.0 - self.decay) * (self.shadow[name] - param.data))

    def apply_shadow(self):
        """Swap current model weights with EMA shadow weights for evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original model weights after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])



def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 16, max_freq: float = 1e4):
        super().__init__()
        self.embed_dim = embed_dim
        half = embed_dim // 2
        self.freqs = nn.Parameter(
            torch.exp(torch.linspace(math.log(1.0), math.log(max_freq), half)), requires_grad=False
        )
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, t: torch.Tensor):
        args = t * self.freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(emb)


class AdaLNResidualBlock(nn.Module):
    """
    Residual block with Adaptive Layer Normalization (FiLM).
    """
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.act1 = nn.SiLU()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.act2 = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.film1 = nn.Linear(cond_dim, hidden_dim * 2)
        self.film2 = nn.Linear(cond_dim, hidden_dim * 2)

    def forward(self, h: torch.Tensor, c: torch.Tensor):
        scale1, shift1 = self.film1(c).chunk(2, dim=-1)
        res = h
        h = self.norm1(h)
        h = h * (1.0 + scale1) + shift1
        h = self.fc1(self.act1(h))
        
        scale2, shift2 = self.film2(c).chunk(2, dim=-1)
        h = self.norm2(h)
        h = h * (1.0 + scale2) + shift2
        h = self.fc2(self.act2(h))
        
        return h + res

# =============================================================================
# 1. TIME-DEPENDENT FLOW MATCHING MODEL (CFM-t)
# =============================================================================
class CFM_t(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dim: int = 40,
        cond_dim: int = 32,
        num_blocks: int = 2
    ):
        super().__init__()
        self.y_dim = y_dim
        
        self.x_proj = nn.Sequential(
            nn.Linear(x_dim, cond_dim // 2),
            nn.SiLU(),
            nn.Linear(cond_dim // 2, cond_dim // 2)
        )
        self.t_emb = SinusoidalTimeEmbedding(cond_dim // 2)
        
        # y_dim * 2 : [state_y, S_mask]
        self.y_proj = nn.Linear(y_dim * 2, hidden_dim)
        
        self.blocks = nn.ModuleList([
            AdaLNResidualBlock(hidden_dim, cond_dim) for _ in range(num_blocks)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, y_dim)

    def forward(self, x: torch.Tensor, I_t: torch.Tensor, t: torch.Tensor, S: torch.Tensor):
        x_emb = self.x_proj(x)
        t_emb = self.t_emb(t)
        c = torch.cat([x_emb, t_emb], dim=-1) 
        
        # concatenation: [I_t, S]
        I_t_masked = torch.cat([I_t, S], dim=-1)
        h = self.y_proj(I_t_masked)
        
        for block in self.blocks:
            h = block(h, c)
            
        h = self.final_norm(h)
        y_pred = self.decoder(h)
        
        return y_pred, h

# =============================================================================
# 2. AUTONOMOUS FLOW MATCHING MODEL (CFM-0)
# =============================================================================
class CFM_0(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dim: int = 40,
        cond_dim: int = 32,
        num_blocks: int = 2
    ):
        super().__init__()
        self.y_dim = y_dim
        
        self.x_proj = nn.Sequential(
            nn.Linear(x_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        
        # CHANGED: Input dimension is now y_dim * 2
        self.y_proj = nn.Linear(y_dim * 2, hidden_dim)
        
        self.blocks = nn.ModuleList([
            AdaLNResidualBlock(hidden_dim, cond_dim) for _ in range(num_blocks)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, y_dim)

    def forward(self, x: torch.Tensor, y_k: torch.Tensor, S: torch.Tensor):
        c = self.x_proj(x)
        
        # EARLY CONCATENATION
        y_k_masked = torch.cat([y_k, S], dim=-1)
        h = self.y_proj(y_k_masked)
        
        for block in self.blocks:
            h = block(h, c)
            
        h = self.final_norm(h)
        y_pred = self.decoder(h)
        
        return y_pred, h

# =============================================================================
# 3. BASELINE REGRESSOR (NN / NN+Soft)
# =============================================================================
class Regressor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        
        self.layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        activation_fn = nn.LeakyReLU() # replication of Valente2025
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(activation_fn)
                
        self.init_weights()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)

