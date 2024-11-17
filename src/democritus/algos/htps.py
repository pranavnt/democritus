import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import torch
import torch.nn.functional as F
import math

from democritus.model.model import ModelArgs, Transformer
from democritus.model.tokenizer import Tokenizer
from democritus.lean.verifier import verify

@dataclass
class HTPSConfig:
    max_seq_len: int = 4096
    max_batch_size: int = 32
    max_tokens: int = 100000  # Total tokens across all nodes
    c_puct: float = 1.0  # Exploration constant
    search_budget: int = 1000  # Maximum search steps
    temperature: float = 0.6
    top_p: float = 0.9

@dataclass
class ProofState:
    """Represents a goal state in the proof"""
    tokens: List[int]
    tactic_state: str  # Lean state string
    node_id: str

class HyperEdge:
    """Represents a tactic application and its resulting subgoals"""
    def __init__(self, source_id: str, tactic: str):
        self.source_id = source_id
        self.tactic = tactic
        self.target_ids: List[str] = []  # Subgoal node IDs
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.mean_value: float = 0.0

    @property
    def Q(self) -> float:
        return self.mean_value if self.visit_count > 0 else 0.0

class ProofNode:
    def __init__(self, node_id: str, state: ProofState):
        self.node_id = node_id
        self.state = state
        self.edges: List[HyperEdge] = []
        self.parent_edges: List[HyperEdge] = []
        self.is_expanded: bool = False
        self.is_solved: bool = False
        self.is_invalid: bool = False
        self.value_estimate: float = 0.0

class HTPS:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int = 4096,
        max_batch_size: int = 32,
        device: str = "auto",
        seed: int = 1,
    ) -> "HTPS":
        """Build HTPS instance from checkpoint"""
        assert os.path.isdir(ckpt_dir), f"Checkpoint directory '{ckpt_dir}' does not exist"
        assert os.path.isfile(tokenizer_path), f"Tokenizer file '{tokenizer_path}' does not exist"

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        torch.manual_seed(seed)

        # Load checkpoint files
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"No checkpoint files found in {ckpt_dir}"

        checkpoint = {}
        for ckpt_file in checkpoints:
            checkpoint.update(torch.load(ckpt_file, map_location='cpu'))

        # Load model parameters
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params
        )

        # Initialize tokenizer and model
        tokenizer = Tokenizer(tokenizer_path)
        assert model_args.vocab_size == tokenizer.n_words

        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        model.to(device)
        model.eval()

        # Create config
        config = HTPSConfig(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            max_tokens=max_seq_len * max_batch_size * 10  # Reasonable buffer
        )

        return HTPS(model, tokenizer, config)

    def __init__(self, model: Transformer, tokenizer: Tokenizer, config: HTPSConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Tree state
        self.nodes: Dict[str, ProofNode] = {}
        self.root_id: Optional[str] = None
        self.next_node_id: int = 0

        # Cache handling - accessed by model layers
        for layer in self.model.layers:
            cache_shape = (
                config.max_batch_size,
                config.max_seq_len,
                layer.attention.n_local_kv_heads,
                layer.attention.head_dim
            )
            layer.attention.cache_k = torch.zeros(cache_shape, device=model.device, dtype=torch.float16)
            layer.attention.cache_v = torch.zeros(cache_shape, device=model.device, dtype=torch.float16)

        # Node -> cache position mapping
        self.cache_positions: Dict[str, int] = {}
        self.next_cache_pos: int = 0

    def _get_node_id(self) -> str:
        node_id = f"node_{self.next_node_id}"
        self.next_node_id += 1
        return node_id

    def _init_root(self, initial_state: ProofState):
        """Initialize root node of proof tree"""
        self.root_id = self._get_node_id()
        root_node = ProofNode(self.root_id, initial_state)
        self.nodes[self.root_id] = root_node
        return root_node

    def _select_node(self) -> ProofNode:
        """Select a node for expansion using PUCT"""
        if not self.root_id:
            raise ValueError("Tree not initialized")

        current = self.nodes[self.root_id]
        path = [current]

        while current.edges and not current.is_solved:
            best_value = float("-inf")
            best_edge = None

            N_parent = sum(edge.visit_count for edge in current.edges)

            for edge in current.edges:
                Q = edge.Q
                U = self.config.c_puct * math.sqrt(N_parent) / (1 + edge.visit_count)
                value = Q + U

                if value > best_value:
                    best_value = value
                    best_edge = edge

            if not best_edge:
                break

            # Select unsolved child
            unsolved_children = [
                self.nodes[child_id]
                for child_id in best_edge.target_ids
                if not self.nodes[child_id].is_solved
            ]

            if not unsolved_children:
                current.is_solved = True
                break

            current = unsolved_children[0]
            path.append(current)

        return current

    @torch.inference_mode()
    def _generate_tactics(self, state: ProofState, temperature: float = 0.6, top_p: float = 0.9) -> List[str]:
        """Generate candidate tactics using the model"""
        input_ids = torch.tensor([state.tokens], device=next(self.model.parameters()).device)

        # Forward pass using cached KV
        logits = self.model.forward(input_ids, start_pos=0)

        # Sample tactics
        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            # Sample using nucleus sampling
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1)
            next_token = torch.gather(probs_idx, -1, next_token)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        # Decode to tactic string
        tactic = self.tokenizer.decode(next_token[0].tolist())
        return [tactic]  # In practice, you might want to generate multiple candidates

    def _apply_tactic(self, state: ProofState, tactic: str, lean_workspace: str) -> Optional[List[ProofState]]:
        """Apply tactic using Lean verifier"""
        # Construct Lean code with tactic
        code = f"{state.tactic_state}\n{tactic}"

        # Verify using Lean
        result = verify(code, lean_workspace)

        if not result['pass']:
            return None

        # Extract new goals from result
        # This would need to be implemented based on your Lean output format
        new_states = []
        for goal in result.get('goals', []):
            new_state = ProofState(
                tokens=self.tokenizer.encode(goal, bos=True, eos=False),
                tactic_state=goal,
                node_id=self._get_node_id()
            )
            new_states.append(new_state)

        return new_states

    def _expand_node(self, node: ProofNode, lean_workspace: str) -> List[ProofNode]:
        """Expand node by generating and applying tactics"""
        if node.is_expanded:
            return []

        tactics = self._generate_tactics(
            node.state,
            temperature=self.config.temperature,
            top_p=self.config.top_p
        )

        new_nodes = []
        for tactic in tactics:
            subgoals = self._apply_tactic(node.state, tactic, lean_workspace)

            if subgoals is not None:  # Tactic was valid
                edge = HyperEdge(node.node_id, tactic)

                # Create nodes for subgoals
                for subgoal in subgoals:
                    child_node = ProofNode(subgoal.node_id, subgoal)
                    self.nodes[subgoal.node_id] = child_node
                    edge.target_ids.append(subgoal.node_id)
                    child_node.parent_edges.append(edge)
                    new_nodes.append(child_node)

                node.edges.append(edge)

        node.is_expanded = True
        return new_nodes

    def _backpropagate(self, node: ProofNode, value: float):
        """Update statistics back up the tree"""
        current = node
        while current.parent_edges:
            for edge in current.parent_edges:
                edge.visit_count += 1
                edge.total_value += value
                edge.mean_value = edge.total_value / edge.visit_count

                # Update parent node
                parent = self.nodes[edge.source_id]

                # Node value is product of children values
                parent_value = 1.0
                for child_id in edge.target_ids:
                    child = self.nodes[child_id]
                    if child.is_solved:
                        child_value = 1.0
                    elif child.is_invalid:
                        child_value = 0.0
                    else:
                        child_value = child.value_estimate
                    parent_value *= child_value

                parent.value_estimate = parent_value
            current = parent

    def _evaluate_node(self, node: ProofNode) -> float:
        """Evaluate node using value network"""
        # TODO: Implement proper value estimation
        return 0.5 if not node.is_invalid else 0.0

    def search(
        self,
        initial_state: ProofState,
        lean_workspace: str,
        max_steps: Optional[int] = None
    ) -> bool:
        """Run proof search"""
        if max_steps is None:
            max_steps = self.config.search_budget

        root = self._init_root(initial_state)

        for _ in range(max_steps):
            # Selection
            node = self._select_node()

            # Expansion
            if not node.is_expanded and not node.is_solved:
                children = self._expand_node(node, lean_workspace)
                if children:
                    node = children[0]

            # Evaluation
            if not node.is_solved and not node.is_invalid:
                value = self._evaluate_node(node)
                node.value_estimate = value

            # Backpropagation
            self._backpropagate(node, node.value_estimate)

            # Check for success
            if self.nodes[self.root_id].is_solved:
                return True

        return False

    def get_proof(self) -> Optional[List[str]]:
        """Extract proof from solved tree"""
        if not self.root_id or not self.nodes[self.root_id].is_solved:
            return None

        proof = []
        def extract_tactics(node_id: str):
            node = self.nodes[node_id]
            for edge in node.edges:
                if all(self.nodes[child_id].is_solved for child_id in edge.target_ids):
                    proof.append(edge.tactic)
                    for child_id in edge.target_ids:
                        extract_tactics(child_id)
                    break

        extract_tactics(self.root_id)
        return proof