# Much more elegant patching using torch.unfold
def patch_image(image, patch_size):
    """
    Split image into patches using torch.unfold - much more efficient!

    Args:
        image: torch.Tensor of shape (H, W, C)
        patch_size: int, size of square patches

    Returns:
        patches: torch.Tensor of shape (num_patches, patch_size, patch_size, C)
    """
    # Rearrange to (C, H, W) for unfold
    img_chw = image.permute(2, 0, 1)
    C, H, W = img_chw.shape

    # Use unfold to extract patches
    # unfold(dimension, size, step) - extracts sliding windows
    patches = img_chw.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # Shape: (C, num_patches_h, num_patches_w, patch_size, patch_size)

    # Reshape to (num_patches, patch_size, patch_size, C)
    num_patches_h, num_patches_w = patches.shape[1], patches.shape[2]
    patches = patches.permute(1, 2, 3, 4, 0).contiguous()
    patches = patches.view(-1, patch_size, patch_size, C)

    return patches, num_patches_h, num_patches_w