def calculate_conv_output_size(input_size, filter_size, padding, stride):
    """
    Calculate the output dimensions of a convolutional layer.

    Args:
        input_size: tuple (height, width, depth)
        filter_size: tuple (height, width, depth)
        padding: padding size (same for all sides)
        stride: stride size (same in all directions)

    Returns:
        tuple: output dimensions (height, width, depth)
    """
    input_height, input_width, input_depth = input_size
    filter_height, filter_width, filter_depth = filter_size

    output_height = (input_height - filter_height + 2 * padding) // stride + 1
    output_width = (input_width - filter_width + 2 * padding) // stride + 1

    return output_height, output_width


def calculate_conv_parameters(filter_size, num_filters):
    """
    Calculate the number of parameters in a convolutional layer.

    Args:
        filter_size: tuple (height, width, depth)
        num_filters: number of filters in the layer

    Returns:
        int: total number of parameters including biases
    """
    filter_height, filter_width, filter_depth = filter_size

    # Parameters per filter: (height * width * depth) weights + 1 bias
    params_per_filter = (filter_height * filter_width * filter_depth) + 1

    # Total parameters: params_per_filter * num_filters
    total_params = params_per_filter * num_filters

    return total_params


def main():
    """
    Solve the convolutional neural network problem.
    """
    # Problem parameters
    input_size = (15, 15, 3)  # height, width, depth
    filter_size = (3, 3, 3)  # height, width, depth
    num_filters = 28
    padding = 1
    stride = 1

    # Calculate output size
    output_height, output_width = calculate_conv_output_size(input_size, filter_size, padding, stride)
    output_depth = num_filters  # depth equals number of filters
    output_size = (output_height, output_width, output_depth)

    # Calculate total parameters
    total_params = calculate_conv_parameters(filter_size, num_filters)

    # Display results
    print("Problem:")
    print("A single (15×15×3) image is passed through a convolutional layer with 28 filters,")
    print("each of size (3×3×3). The padding size is 1 (1 unit at top, bottom, left, and right)")
    print("and the stride size is also 1.")
    print("\nSolution:")
    print(f"1. Output Feature Map Size: {output_size[0]}×{output_size[1]}×{output_size[2]}")
    print(f"   Explanation: Using the formula (N-size − F-size + 2*(Padding) )/(Stride) + 1 for both height and width:")
    print(f"   Height: (15-3+2*1)/1+1 = 15")
    print(f"   Width: (15-3+2*1)/1+1 = 15")
    print(f"   Depth: Number of filters = 28")
    print(f"   Therefore, output volume is {output_size[0]}×{output_size[1]}×{output_size[2]}")
    print("\n2. Number of Parameters:")
    print(f"   Parameters per filter: (3×3×3) weights + 1 bias = 28 weights + 1 bias = 29")
    print(f"   Total parameters: 29 × 28 filters = {total_params}")
    print(f"   This can also be calculated as: (3×3×3×28) + 28 = 756 + 28 = {total_params}")


if __name__ == "__main__":
    main()