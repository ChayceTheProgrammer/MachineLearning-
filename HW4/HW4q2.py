def analyze_residual_block():
    """
    Analyze the residual block described in problem 2 of HW4.

    The block consists of:
    - Input with 128 feature maps
    - First conv layer: 3x3, stride=2, 256 feature maps
    - Second conv layer: 3x3, stride=1, 256 feature maps
    - Skip connection with addition to form output with 256 feature maps
    """

    print("Analysis of Residual Block from Problem 2:")
    print("=========================================\n")

    # Part (a) - What projection shortcut operations are required?
    print("(a) Projection Shortcut Analysis:")
    print("-------------------------------")
    print("The skip connection needs to match dimensions from input to output for addition.")
    print("Input: 128 feature maps")
    print("Output: 256 feature maps")
    print("First conv uses stride=2, which reduces spatial dimensions by half.")
    print("\nTwo issues that prevent direct addition:")
    print("1. Different number of feature maps (128 vs 256)")
    print("2. Different spatial dimensions due to stride=2")
    print("\nRequired projection shortcut:")
    print("A 1x1 convolution with stride=2 that projects 128 feature maps to 256 feature maps")
    print("while also reducing spatial dimensions by half to match the main path.\n")

    # Part (b) - Calculate total trainable parameters
    print("(b) Parameter Count Calculation:")
    print("------------------------------")

    # First convolutional layer
    first_conv_params = 3 * 3 * 128 * 256
    print(f"First conv layer (3x3, stride=2, 128→256):")
    print(f"  Filter size: 3x3x128")
    print(f"  Number of filters: 256")
    print(f"  Parameters: 3 × 3 × 128 × 256 = {first_conv_params:,}")

    # Second convolutional layer
    second_conv_params = 3 * 3 * 256 * 256
    print(f"\nSecond conv layer (3x3, stride=1, 256→256):")
    print(f"  Filter size: 3x3x256")
    print(f"  Number of filters: 256")
    print(f"  Parameters: 3 × 3 × 256 × 256 = {second_conv_params:,}")

    # Projection shortcut
    shortcut_params = 1 * 1 * 128 * 256
    print(f"\nProjection shortcut (1x1, stride=2, 128→256):")
    print(f"  Filter size: 1x1x128")
    print(f"  Number of filters: 256")
    print(f"  Parameters: 1 × 1 × 128 × 256 = {shortcut_params:,}")

    # Total parameters
    total_params = first_conv_params + second_conv_params + shortcut_params
    print(
        f"\nTotal trainable parameters: {first_conv_params:,} + {second_conv_params:,} + {shortcut_params:,} = {total_params:,}")


if __name__ == "__main__":
    analyze_residual_block()