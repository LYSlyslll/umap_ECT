import collections
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.cluster import KMeans

from typing import Union, Tuple, Iterable, Dict, List, Any


class TreeNode:
    """
    A node in the DeepECT cluster tree.

    Attributes:
        id (int): A unique identifier for the node.
        parent (TreeNode, optional): The parent node in the tree.
        left (TreeNode, optional): The left child node.
        right (TreeNode, optional): The right child node.
        center (nn.Parameter): The learnable cluster center of the node.
        weight (torch.Tensor): The weight of the node, indicating its importance or occupancy.
        is_leaf (bool): True if the node is a leaf, False otherwise.
    """
    _next_id = 0

    def __init__(self, center: Union[torch.Tensor, nn.Parameter], parent: Union['TreeNode', None] = None, device: torch.device = 'cpu') -> None:
        self.id = TreeNode._next_id
        TreeNode._next_id += 1

        self.parent = parent
        self.left = None
        self.right = None

        if isinstance(center, torch.Tensor):
            self.center = nn.Parameter(center.clone().detach().to(device))
        else:
            self.center = center

        self.weight = torch.tensor(1.0, device=device)
        self.is_leaf = True

    def update_weight(self, assigned_ratio: float, alpha: float = 0.5) -> None:
        """
        Updates the node's weight using an exponential moving average.

        Args:
            assigned_ratio (float): The proportion of data points assigned to this node in the current batch.
            alpha (float): The smoothing factor for the moving average.
        """
        self.weight = (1 - alpha) * self.weight + alpha * assigned_ratio

    def __repr__(self) -> str:
        return f'Node(id={self.id}, leaf={self.is_leaf}, weight={self.weight:.2f})'


def cosine_distance_loss(input_tensor: torch.Tensor, target_tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Computes cosine-distance loss (1 - cosine similarity) between tensors."""
    if input_tensor.dim() == 1:
        input_tensor = input_tensor.unsqueeze(0)
    if target_tensor.dim() == 1:
        target_tensor = target_tensor.unsqueeze(0)

    if input_tensor.shape != target_tensor.shape:
        if target_tensor.size(0) == 1:
            target_tensor = target_tensor.expand(input_tensor.size(0), *target_tensor.shape[1:])
        elif input_tensor.size(0) == 1:
            input_tensor = input_tensor.expand(target_tensor.size(0), *input_tensor.shape[1:])
        else:
            raise ValueError("Input and target shapes are incompatible for cosine distance loss.")

    cosine_sim = F.cosine_similarity(input_tensor, target_tensor, dim=-1, eps=eps)
    return (1 - cosine_sim).mean()


class DeepECT(nn.Module):
    """
    Deep Embedded Clustering Tree (DeepECT) model.
    This variant operates directly on fixed low-dimensional embeddings produced
    externally (e.g., via UMAP) and optimizes only the tree structure.
    """
    def __init__(self, latent_dim: int, device: torch.device = 'cpu') -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device

        # Initialize the tree with a single root node at the origin
        initial_center = torch.zeros(latent_dim, device=self.device)
        self.root = TreeNode(initial_center, device=self.device)

        # Store nodes in dictionaries and lists for easy access
        self.nodes = {self.root.id: self.root}
        self.leaf_nodes = [self.root]

    def get_tree_parameters(self) -> list:
        """
        Gathers all learnable parameters (centers of leaf nodes) from the tree.

        Returns:
            list: A list of nn.Parameter objects corresponding to the leaf node centers.
        """
        params = []
        for node in self.nodes.values():
            if node.is_leaf:
                params.append(node.center)
        return params
    
    def initialize_tree_from_latents(self, latent_vectors: torch.Tensor) -> None:
        """Initializes the root center using pre-computed latent representations.

        Args:
            latent_vectors (torch.Tensor): Latent representations obtained from
                the pre-trained embedding model. Expected shape is
                ``(num_samples, latent_dim)``.
        """
        if latent_vectors is None:
            return

        if latent_vectors.ndim != 2:
            raise ValueError("latent_vectors must be a 2D tensor of shape (N, latent_dim).")

        if latent_vectors.size(1) != self.latent_dim:
            raise ValueError(
                f"latent_vectors have dimension {latent_vectors.size(1)}, expected {self.latent_dim}."
            )

        if latent_vectors.numel() == 0:
            return

        latent_vectors = latent_vectors.to(self.device)
        root_center = latent_vectors.mean(dim=0)
        self.root.center.data.copy_(root_center)
        self.root.weight = torch.tensor(1.0, device=self.device)
        self.root.is_leaf = True
        self.leaf_nodes = [self.root]
        self.nodes = {self.root.id: self.root}

    def initialize_tree_from_latents(self, latent_vectors: torch.Tensor) -> None:
        """Initializes the root center using pre-computed latent representations.

        Args:
            latent_vectors (torch.Tensor): Latent representations obtained from
                the pre-trained embedding model. Expected shape is
                ``(num_samples, latent_dim)``.
        """
        if latent_vectors is None:
            return

        if latent_vectors.ndim != 2:
            raise ValueError("latent_vectors must be a 2D tensor of shape (N, latent_dim).")

        if latent_vectors.size(1) != self.latent_dim:
            raise ValueError(
                f"latent_vectors have dimension {latent_vectors.size(1)}, expected {self.latent_dim}."
            )

        if latent_vectors.numel() == 0:
            return

        latent_vectors = latent_vectors.to(self.device)
        root_center = latent_vectors.mean(dim=0)
        self.root.center.data.copy_(root_center)
        self.root.weight = torch.tensor(1.0, device=self.device)
        self.root.is_leaf = True
        self.leaf_nodes = [self.root]
        self.nodes = {self.root.id: self.root}

    def _extract_batch(self, batch: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Normalizes a batch coming from a DataLoader into embeddings and metadata.

        Args:
            batch: The batch returned by a PyTorch DataLoader. It can be a tensor,
                a tuple/list (e.g., ``(embeddings, indices, ...)``), or a
                dictionary containing an ``"embedding"`` key.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: The embeddings tensor and a
                dictionary with any additional metadata (e.g., indices).
        """
        metadata: Dict[str, Any] = {}

        if isinstance(batch, torch.Tensor):
            embeddings = batch
        elif isinstance(batch, (list, tuple)):
            if len(batch) == 0:
                raise ValueError("Received an empty batch from the dataloader.")
            embeddings = batch[0]
            if len(batch) >= 2:
                metadata['idx'] = batch[1]
            if len(batch) > 2:
                metadata['extra'] = batch[2:]
        elif isinstance(batch, dict):
            if 'embedding' not in batch:
                raise ValueError("Dictionary batches must contain an 'embedding' key.")
            embeddings = batch['embedding']
            metadata = {k: v for k, v in batch.items() if k != 'embedding'}
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)

        return embeddings, metadata

    def _find_closest_leaf(self, z: torch.Tensor) -> torch.Tensor:
        """
        Finds the closest leaf node for each vector in the batch `z`.

        Args:
            z (torch.Tensor): The batch of latent vectors, shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: A tensor of indices corresponding to the closest leaf node for each vector.
        """
        leaf_centers = torch.stack([node.center for node in self.leaf_nodes])
        distances = torch.cdist(z, leaf_centers)
        assignments = torch.argmin(distances, dim=1)
        return assignments

    def _update_inner_node_centers(self) -> None:
        """
        Recursively updates the centers of internal nodes from the bottom up.
        The center of an internal node is the weighted average of its children's centers.
        """
        def post_order_traversal(node):
            if node is None or node.is_leaf:
                return

            post_order_traversal(node.left)
            post_order_traversal(node.right)

            # Update center based on children's weights and centers
            total_weight = node.left.weight + node.right.weight
            if total_weight > 1e-8: # Avoid division by zero
                w_l, w_r = node.left.weight, node.right.weight
                mu_l, mu_r = node.left.center, node.right.center
                node.center.data = (w_l * mu_l + w_r * mu_r) / total_weight

        post_order_traversal(self.root)

    def _grow_tree(self, data_loader: Iterable, max_leaves: int, split_count: Union[int, float] = 1) -> bool:
        """
        The tree growing procedure. It identifies leaf nodes with the highest variance
        and splits them using 2-means clustering.

        Args:
            data_loader: The data loader to evaluate variance on.
            max_leaves (int): The maximum number of leaf nodes allowed in the tree.
            split_count (int or float): The number of nodes to split.
                - If int: The exact number of nodes to split.
                - If float (0.0, 1.0): The fraction of current leaf nodes to split.

        Returns:
            bool: True if the tree was grown, False otherwise.
        """
        any_split_successful = False

        with torch.no_grad():
            # Get all latent representations from the data
            latent_batches = []
            for batch in data_loader:
                embeddings, _ = self._extract_batch(batch)
                embeddings = embeddings.to(self.device)
                latent_batches.append(embeddings)

            if not latent_batches:
                return False

            all_z = torch.cat(latent_batches, dim=0)
            assignments = self._find_closest_leaf(all_z)

            # Find split candidates based on intra-cluster variance
            split_candidates = []
            for i, leaf in enumerate(self.leaf_nodes):
                assigned_z = all_z[assignments == i]
                if len(assigned_z) > 1:
                    variance = torch.sum((assigned_z - leaf.center)**2)
                    split_candidates.append({'variance': variance, 'node': leaf, 'data': assigned_z})

        if not split_candidates:
            return False

        split_candidates.sort(key=lambda x: x['variance'], reverse=True)

        # Determine the number of nodes to split
        current_leaves_count = len(self.leaf_nodes)
        if isinstance(split_count, float):
            num_to_split = max(1, int(current_leaves_count * split_count))
        else:
            num_to_split = int(split_count)

        num_to_split = min(num_to_split, len(split_candidates))
        available_slots = max_leaves - current_leaves_count
        num_to_split = min(num_to_split, available_slots)
        
        if num_to_split <= 0:
            return False

        nodes_to_split_info = split_candidates[:num_to_split]

        for split_info in nodes_to_split_info:
            node_to_split = split_info['node']
            data_for_split = split_info['data']

            # Split the node's data into two new clusters using 2-means
            kmeans = KMeans(n_clusters=2, n_init='auto', random_state=42)
            kmeans.fit(data_for_split.cpu().numpy())
            new_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=self.device)

            # Update the tree structure
            node_to_split.is_leaf = False
            left_child = TreeNode(new_centers[0], parent=node_to_split, device=self.device)
            right_child = TreeNode(new_centers[1], parent=node_to_split, device=self.device)
            node_to_split.left = left_child
            node_to_split.right = right_child

            self.nodes[left_child.id] = left_child
            self.nodes[right_child.id] = right_child

            self.leaf_nodes.remove(node_to_split)
            self.leaf_nodes.extend([left_child, right_child])
            any_split_successful = True
        
        if any_split_successful:
            self._update_inner_node_centers()

        return any_split_successful

    def _prune_tree(self, threshold: float = 0.1) -> bool:
        """
        The tree pruning procedure. It finds and removes 'dead' leaf nodes with weights
        below a given threshold.

        Args:
            threshold (float): The weight threshold below which a node is considered 'dead'.

        Returns:
            bool: True if the tree was pruned, False otherwise.
        """
        tree_was_pruned = False
        # Iterate over a copy of the list as we modify it
        for leaf in list(self.leaf_nodes):
            if leaf.weight < threshold and leaf.parent is not None:
                # Identify the dead node, its parent, sibling, and grandparent
                dead_node = leaf
                parent = dead_node.parent
                sibling = parent.left if parent.right == dead_node else parent.right
                grandparent = parent.parent

                if grandparent is not None:
                    # Rewire the grandparent to point directly to the sibling
                    if grandparent.left == parent:
                        grandparent.left = sibling
                    else:
                        grandparent.right = sibling
                    sibling.parent = grandparent
                else:
                    # If the parent was the root, the sibling becomes the new root
                    self.root = sibling
                    sibling.parent = None

                # Clean up the removed nodes
                self.leaf_nodes.remove(dead_node)
                del self.nodes[dead_node.id]
                del self.nodes[parent.id]

                tree_was_pruned = True

        return tree_was_pruned

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_rec = torch.tensor(0.0, device=self.device)

        if not self.leaf_nodes:
            return loss_rec, loss_rec, torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        leaf_assignments = self._find_closest_leaf(z)

        # Loss 2: Node Center Loss (L_NC)
        # Pulls leaf centers towards the mean of their assigned data points.
        loss_nc = torch.tensor(0.0, device=self.device)
        num_leaves_with_data = 0
        for i, leaf in enumerate(self.leaf_nodes):
            assigned_indices = (leaf_assignments == i).nonzero(as_tuple=True)[0]
            if len(assigned_indices) > 0:
                mean_z = z[assigned_indices].mean(dim=0).detach()
                loss_nc += cosine_distance_loss(leaf.center, mean_z)
                num_leaves_with_data += 1
        if num_leaves_with_data > 0:
            loss_nc /= num_leaves_with_data

        # Loss 3: Node Data Compression Loss (L_DC)
        # Encourages data points within a subtree to lie along the direction
        # that separates the node from its sibling, improving cluster separation.
        loss_dc = torch.tensor(0.0, device=self.device)
        
        # Pre-compute ancestors for each leaf to speed up lookup
        leaf_to_ancestors = {}
        for leaf_idx, leaf_node in enumerate(self.leaf_nodes):
            ancestors = set()
            curr = leaf_node
            while curr is not None:
                ancestors.add(curr.id)
                curr = curr.parent
            leaf_to_ancestors[leaf_idx] = ancestors

        # Pre-compute projection vectors (direction between siblings)
        projection_vectors = {}
        nodes_to_process = [node for node in self.nodes.values() if node.parent is not None]
        for node in nodes_to_process:
            sibling = node.parent.left if node.parent.right == node else node.parent.right
            direction = node.center - sibling.center
            norm = torch.norm(direction)
            if norm > 1e-8:
                projection_vectors[node.id] = direction / norm

        # Map each data point to all its ancestor nodes in the tree
        points_assigned_to_internal_nodes = collections.defaultdict(list)
        for point_idx, assigned_leaf_idx in enumerate(leaf_assignments):
            ancestors_ids = leaf_to_ancestors[assigned_leaf_idx.item()]
            for ancestor_id in ancestors_ids:
                points_assigned_to_internal_nodes[ancestor_id].append(point_idx)
        
        # Calculate loss for each relevant internal node
        num_nodes_for_dc = 0
        for node in nodes_to_process:
            if node.id in points_assigned_to_internal_nodes and node.id in projection_vectors:
                assigned_z_for_node = z[points_assigned_to_internal_nodes[node.id]]
                projection_vector = projection_vectors[node.id]

                # Align projected vectors using cosine distance for direction consistency
                diff = assigned_z_for_node - node.center.detach()
                loss_dc += cosine_distance_loss(diff, projection_vector.unsqueeze(0))
                num_nodes_for_dc += 1
        if num_nodes_for_dc > 0:
            loss_dc /= num_nodes_for_dc

        total_loss = loss_nc + loss_dc

        # Update node weights and internal centers after loss calculation
        with torch.no_grad():
            for i, leaf in enumerate(self.leaf_nodes):
                num_assigned = (leaf_assignments == i).sum().float() / len(leaf_assignments)
                leaf.update_weight(num_assigned)
            self._update_inner_node_centers()

        return total_loss, loss_rec, loss_nc, loss_dc

    def train(self, dataloader: Iterable, iterations: int, lr: float, max_leaves: int, split_interval: int,
              pruning_threshold: float, split_count_per_growth: Union[int, float] = 1,
              lr_decay_step: int = 100, lr_decay_gamma: float = 0.95) -> Dict[str, List[float]]:
        """
        The main training loop for the DeepECT model.

        Args:
            dataloader: The data loader for training.
            iterations (int): The total number of training iterations.
            lr (float): The initial learning rate for the optimizer.
            max_leaves (int): The maximum number of leaf nodes in the tree.
            split_interval (int): The number of iterations between tree growing procedures.
            pruning_threshold (float): The weight threshold for pruning dead nodes.
            split_count_per_growth (int or float): The number or fraction of nodes to split during each growth phase.
            lr_decay_step (int): Number of iterations between StepLR updates.
            lr_decay_gamma (float): Multiplicative factor of learning rate decay.

        Returns:
            Dict[str, List[float]]: History of the total loss and its components collected at each iteration.
        """
        # Helper to re-create the optimizer when tree structure changes
        if lr_decay_step <= 0:
            raise ValueError("lr_decay_step must be a positive integer.")

        def create_optimizer(tree_params):
            if tree_params:
                return optim.Adam(tree_params, lr=lr)
            else:
                raise ValueError("Tree must have parameters to optimize.")

        optimizer = create_optimizer(self.get_tree_parameters())
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)
        bar = tqdm(range(iterations))
        data_iterator = iter(dataloader)
        iteration = 0

        loss_history: Dict[str, List[float]] = {
            'total': [],
            'reconstruction': [],
            'node_center': [],
            'node_compression': []
        }

        while iteration < iterations:
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(dataloader)
                batch = next(data_iterator)

            super().train()
            data, _ = self._extract_batch(batch)
            data = data.to(self.device)
            structure_changed = False

            # Prune the tree to remove inactive nodes
            if self._prune_tree(threshold=pruning_threshold):
                structure_changed = True

            # Grow the tree by splitting high-variance nodes
            if iteration > 0 and iteration % split_interval == 0 and len(self.leaf_nodes) < max_leaves:
                if self._grow_tree(dataloader, max_leaves=max_leaves, split_count=split_count_per_growth):
                    structure_changed = True

            # If tree structure changed, we need a new optimizer for the new set of parameters
            if structure_changed:
                optimizer = create_optimizer(self.get_tree_parameters())
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

            # Standard training step
            optimizer.zero_grad()
            loss, l_rec, l_nc, l_dc = self(data)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_history['total'].append(loss.item())
            loss_history['reconstruction'].append(l_rec.item())
            loss_history['node_center'].append(l_nc.item())
            loss_history['node_compression'].append(l_dc.item())

            # Update progress bar
            bar.update(1)
            bar.set_description(
                f'Iter {iteration} | '
                f'Loss: {loss.item():.3f} | '
                f'Rec: {l_rec.item():.3f} | NC: {l_nc.item():.3f} | DC: {l_dc.item():.3f} | '
                f'Leaves: {len(self.leaf_nodes)} | '
                f'LR: {scheduler.get_last_lr()[0]:.2e}'
            )
            iteration += 1

        bar.close()
        return loss_history
    
    def predict(self, data_loader: Iterable) -> Dict[str, Any]:
        """
        Assigns each data point from the data loader to a cluster (leaf node).

        Args:
            data_loader: The data loader containing the data to predict on.

        Returns:
            Dict[str, Any]: A dictionary containing the assignments tensor under
                the key ``"assignments"``. If the provided dataloader yields
                sample indices (under the ``"idx"`` key or as the second
                element of a tuple), they are returned under the key ``"idx"``
                in the same order as ``"assignments"``.
        """
        with torch.no_grad():
            all_z = []
            all_indices: List[Any] = []
            for batch in tqdm(data_loader, desc="Predicting"):
                data, metadata = self._extract_batch(batch)
                data = data.to(self.device)
                all_z.append(data)

                idx_values = metadata.get('idx')
                if idx_values is not None:
                    if isinstance(idx_values, torch.Tensor):
                        all_indices.extend(idx_values.detach().cpu().tolist())
                    elif isinstance(idx_values, (list, tuple)):
                        all_indices.extend(list(idx_values))
                    else:
                        all_indices.append(idx_values)

            all_z = torch.cat(all_z, dim=0)
            assignments = self._find_closest_leaf(all_z)

        result: Dict[str, Any] = {'assignments': assignments}
        if all_indices:
            result['idx'] = all_indices

        return result
    
    def get_state(self) -> Dict[str, Any]:
        """
        Serializes the tree state into a dictionary for saving.

        Returns:
            Dict[str, Any]: A dictionary containing the model's state.
        """
        tree_nodes_data = []
        for node in self.nodes.values():
            node_data = {
                'id': node.id,
                'center': node.center.detach().cpu(),
                'weight': node.weight,
                'is_leaf': node.is_leaf,
                'parent_id': node.parent.id if node.parent else None,
                'left_id': node.left.id if node.left else None,
                'right_id': node.right.id if node.right else None,
            }
            tree_nodes_data.append(node_data)
            
        tree_state = {
            'nodes_data': tree_nodes_data,
            'root_id': self.root.id if self.root else None
        }

        state = {
            'tree_state': tree_state,
        }
        return state

    def save_model(self, path: str) -> None:
        """
        Saves the complete model state to a file.

        Args:
            path (str): The path to the file where the model will be saved.
        """
        state = self.get_state()
        torch.save(state, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Loads the complete model state from a file.

        Args:
            path (str): The path to the file from which to load the model.
        """
        checkpoint = torch.load(path, map_location=self.device)

        tree_state = checkpoint['tree_state']
        if tree_state.get('root_id') is None:
            self.root = None
            self.nodes = {}
            self.leaf_nodes = []
            return

        self.nodes.clear()
        self.leaf_nodes.clear()
        
        # First pass: create all nodes
        for node_data in tree_state['nodes_data']:
            node = TreeNode(center=node_data['center'], device=self.device)
            node.id = node_data['id']
            node.weight = node_data['weight']
            node.is_leaf = node_data['is_leaf']
            self.nodes[node.id] = node

        # Second pass: link nodes together
        for node_data in tree_state['nodes_data']:
            node = self.nodes[node_data['id']]
            if node_data['parent_id'] is not None:
                node.parent = self.nodes[node_data['parent_id']]
            if node_data['left_id'] is not None:
                node.left = self.nodes[node_data['left_id']]
            if node_data['right_id'] is not None:
                node.right = self.nodes[node_data['right_id']]

        self.root = self.nodes[tree_state['root_id']]
        self.leaf_nodes = [node for node in self.nodes.values() if node.is_leaf]

        # Ensure new node IDs don't conflict with loaded ones
        max_id = max([d['id'] for d in tree_state['nodes_data']]) if tree_state['nodes_data'] else -1
        TreeNode._next_id = max_id + 1
        
        print(f"Model loaded from {path}. Tree has {len(self.nodes)} nodes ({len(self.leaf_nodes)} leaves).")

    def prune_subtree(self, node_id_to_prune: int) -> None:
        """
        Manually prunes all descendants of a given node, making it a leaf.

        Args:
            node_id_to_prune (int): The ID of the node whose subtree should be pruned.
        """
        if node_id_to_prune not in self.nodes:
            print(f"Node with id {node_id_to_prune} not found.")
            return

        target_node = self.nodes[node_id_to_prune]
        if target_node.is_leaf:
            print(f"Node {node_id_to_prune} is already a leaf.")
            return

        # Find all descendants using a queue (BFS)
        descendants_to_delete = []
        queue = collections.deque()
        if target_node.left:
            queue.append(target_node.left)
        if target_node.right:
            queue.append(target_node.right)
            
        while queue:
            current_node = queue.popleft()
            descendants_to_delete.append(current_node)
            if current_node.left:
                queue.append(current_node.left)
            if current_node.right:
                queue.append(current_node.right)

        print(f"Pruning subtree of node {target_node.id}. Deleting {len(descendants_to_delete)} descendants.")
        
        # Remove descendants from global lists
        for node in descendants_to_delete:
            del self.nodes[node.id]
            if not node.is_leaf:
                continue
            if node in self.leaf_nodes:
                self.leaf_nodes.remove(node)
        
        # Make the target node a leaf
        target_node.left = None
        target_node.right = None
        target_node.is_leaf = True
        if target_node not in self.leaf_nodes:
            self.leaf_nodes.append(target_node)
        
        # Update internal node centers after pruning
        self._update_inner_node_centers()
