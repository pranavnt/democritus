import dataclasses
from typing import List, Set, Dict, Optional
from enum import Enum

@dataclasses.dataclass
class ProofNode:
    """A node in the proof tree"""
    goals: List[str]  # Current goals to prove
    context: str      # Current proof context
    available_lemmas: Set[str]  # Available lemmas
    parent: Optional['ProofNode'] = None
    children: List['ProofNode'] = dataclasses.field(default_factory=list)

    # Search statistics
    visit_count: Dict[str, int] = dataclasses.field(default_factory=dict)  # N(g,t)
    total_value: Dict[str, float] = dataclasses.field(default_factory=dict) # W(g,t)
    is_solved: bool = False

class NodeType(Enum):
    """Types of nodes in proof tree"""
    SOLVED = "solved"       # Node has been proved
    UNSOLVED = "unsolved"  # Node has unexpanded tactics
    INVALID = "invalid"    # All tactics failed/invalid

class HyperTreeSearch:
    def __init__(self, lean_repl, language_model, value_model):
        self.repl = lean_repl
        self.model = language_model
        self.critic = value_model
        self.root = None
        self.nodes = set()  # All nodes in tree
        self.c_puct = 1.0  # Exploration constant

    def search(self, theorem: str, budget: int) -> Optional[List[str]]:
        """Main search loop"""
        self.root = ProofNode(goals=[theorem], context="", available_lemmas=set())
        self.nodes.add(self.root)

        for _ in range(budget):
            # 1. Selection phase
            tree = self.select()

            # 2. Expansion phase
            new_nodes = self.expand(tree)

            # 3. Backpropagation
            self.backpropagate(tree, new_nodes)

            # Check if theorem is proved
            if self.root.is_solved:
                return self.extract_proof()

        return None

    def select(self) -> List[ProofNode]:
        """Selection phase - returns tree of nodes to expand"""
        tree = []
        current = self.root

        while not self.is_leaf(current):
            # Get best tactic according to PUCT formula
            tactic = self.select_tactic(current)
            tree.append(current)

            # Follow tactic
            current = self.apply_tactic(current, tactic)

        tree.append(current)
        return tree

    def select_tactic(self, node: ProofNode) -> str:
        """Select tactic using PUCT formula"""
        def puct_score(tactic: str) -> float:
            # Q-value
            q_value = (node.total_value.get(tactic, 0) /
                      max(node.visit_count.get(tactic, 1), 1))

            # Prior from policy
            prior = self.model.get_prior(node.goals, tactic)

            # Visit count terms
            n_total = sum(node.visit_count.values())
            n_tactic = node.visit_count.get(tactic, 0)

            # PUCT formula
            score = (q_value +
                    self.c_puct * prior * (n_total ** 0.5) / (1 + n_tactic))
            return score

        valid_tactics = self.get_valid_tactics(node)
        return max(valid_tactics, key=puct_score)

    def expand(self, tree: List[ProofNode]) -> List[ProofNode]:
        """Expansion phase"""
        leaf = tree[-1]

        # Generate tactics from model
        tactics = self.model.generate_tactics(leaf.goals)
        new_nodes = []

        # Try each tactic
        for tactic in tactics:
            # Verify tactic
            success, new_goals = self.repl.verify_tactic(tactic, leaf.goals)

            if success:
                # Create new node for each subgoal
                child = ProofNode(
                    goals=new_goals,
                    context=leaf.context,
                    available_lemmas=leaf.available_lemmas,
                    parent=leaf
                )
                leaf.children.append(child)
                self.nodes.add(child)
                new_nodes.append(child)

        return new_nodes

    def backpropagate(self, tree: List[ProofNode], new_nodes: List[ProofNode]):
        """Backpropagation phase"""
        # Evaluate new nodes
        values = {}
        for node in new_nodes:
            if not node.goals:  # Empty goals = solved
                values[node] = 1.0
            else:
                values[node] = self.critic.evaluate(node.goals)

        # Backup values through tree
        for node in reversed(tree):
            # Value is product of child values (all subgoals must be proved)
            node_value = 1.0
            for child in node.children:
                if child in values:
                    node_value *= values[child]
            values[node] = node_value

            # Update statistics
            tactic = self.get_tactic(node)
            node.visit_count[tactic] = node.visit_count.get(tactic, 0) + 1
            node.total_value[tactic] = (node.total_value.get(tactic, 0) +
                                      values[node])

            # Update solved status
            self.update_status(node)

    def update_status(self, node: ProofNode):
        """Update node status (solved/unsolved/invalid)"""
        # Node is solved if:
        # 1. It has no goals (leaf node success)
        # 2. All children are solved for some tactic
        if not node.goals:
            node.is_solved = True
            return

        for tactic in self.get_valid_tactics(node):
            children = self.get_children(node, tactic)
            if children and all(c.is_solved for c in children):
                node.is_solved = True
                return

        # Otherwise node remains unsolved
        node.is_solved = False

    def extract_proof(self) -> List[str]:
        """Extract proof steps from solved tree"""
        if not self.root.is_solved:
            return None

        proof = []
        def extract_steps(node: ProofNode):
            if not node.goals:
                return

            # Find solving tactic
            for tactic in self.get_valid_tactics(node):
                children = self.get_children(node, tactic)
                if all(c.is_solved for c in children):
                    proof.append(tactic)
                    for child in children:
                        extract_steps(child)
                    break

        extract_steps(self.root)
        return proof

    # Helper methods
    def is_leaf(self, node: ProofNode) -> bool:
        """Check if node is a leaf in the tree"""
        return not node.children

    def get_valid_tactics(self, node: ProofNode) -> List[str]:
        """Get valid tactics for a node"""
        # Implementation depends on how tactics are stored
        pass

    def get_children(self, node: ProofNode, tactic: str) -> List[ProofNode]:
        """Get child nodes for a given tactic"""
        # Implementation depends on how children are stored
        pass

    def get_tactic(self, node: ProofNode) -> str:
        """Get tactic that led to this node"""
        # Implementation depends on tree structure
        pass