import torch

def crop_to_workspace(pcd, feats, workspace_bounds, max_pcd_points = None):
    """
    Returns the indices of points within the specified workspace bounds, ensuring each batch has an equal number of points.

    Parameters:
    - pcd: A tensor of shape (B, N, 3) representing the point cloud.
    - feats: A tensor of shape (B, N, F) representing the features.
    - workspace_bounds: A list or tensor of shape (2, 3) specifying the min and max bounds for x, y, z.

    Returns:
    - trimmed_pcd: A list of tensors, where each tensor contains the selected points for a batch.
    - trimmed_feats: A list of tensors, where each tensor contains the feature values corresponding to the selected points for a batch.
    """
    batch_size = pcd.shape[0]
    batch_indices = []

    n_mcd_min = torch.inf
    for b in range(batch_size):
        mask = torch.ones(pcd[b].shape[0], dtype=torch.bool, device=pcd.device)
        for i in range(3):
            mask = torch.logical_and(mask, pcd[b, :, i] > workspace_bounds[0][i])
            mask = torch.logical_and(mask, pcd[b, :, i] < workspace_bounds[1][i])

        indices = torch.nonzero(mask, as_tuple=False).squeeze(1)  # Get the indices where mask is True

        n_mcd_min = min(n_mcd_min, len(indices))
        batch_indices.append(indices)

    if max_pcd_points is None:
        max_pcd_points = n_mcd_min

    for b in range(batch_size):
        # Randomly sample points if there are more than max_pcd_points
        indices = batch_indices[b]
        if len(indices) > max_pcd_points:
            batch_indices[b] = indices[torch.randperm(len(indices))[:max_pcd_points]]

    # Extract the corresponding points and RGB values
    cropped_pcd = [pcd[b, indices, :] for b, indices in enumerate(batch_indices)]
    cropped_feats = [feats[b, indices, :] for b, indices in enumerate(batch_indices)]

    cropped_pcd = torch.stack(cropped_pcd)
    cropped_feats = torch.stack(cropped_feats)

    return cropped_pcd, cropped_feats 
