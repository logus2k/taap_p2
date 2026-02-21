def compute_genome_size(input_size, hidden1_size, hidden2_size, output_size):
    return (input_size * hidden1_size + hidden1_size +
            hidden1_size * hidden2_size + hidden2_size +
            hidden2_size * output_size + output_size)
