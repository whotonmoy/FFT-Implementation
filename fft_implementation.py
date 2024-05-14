import cmath

def bit_reverse(num, num_bits):
    reversed_num = 0
    for _ in range(num_bits):
        reversed_num = (reversed_num << 1) | (num & 1)
        num >>= 1
    return reversed_num

def one_step_fft_shuffling(data):
    N = len(data)
    num_bits = int(N.bit_length() - 1)
    
    # Perform bit-reversal permutation
    shuffled_data = [data[bit_reverse(i, num_bits)] for i in range(N)]
    
    return shuffled_data

def fft1d(x):
    N = len(x)
    
    if N <= 1:
        return x
    even = fft1d(x[0::2])
    odd = fft1d(x[1::2])
    
    T = [cmath.exp(-2j * cmath.pi * k / N) * odd[k] for k in range(N // 2)]
    
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

def fft2d(img):
    rows, cols = len(img), len(img[0])
    fft_img = [[0j] * cols for _ in range(rows)]
    
    # FFT along rows
    for i in range(rows):
        fft_img[i] = fft1d(one_step_fft_shuffling(img[i]))

    # FFT along columns
    for j in range(cols):
        col_data = [fft_img[i][j] for i in range(rows)]
        col_data = fft1d(one_step_fft_shuffling(col_data))
        for i in range(rows):
            fft_img[i][j] = col_data[i]

    return fft_img

def ifft2d(fft_img):
    rows, cols = len(fft_img), len(fft_img[0])
    ifft_img = [[0] * cols for _ in range(rows)]
    
    # IFFT along rows
    for i in range(rows):
        ifft_img[i] = [int(cmath.phase(val) * 255 / (2 * cmath.pi)) for val in fft1d(one_step_fft_shuffling(fft_img[i]))]

    # IFFT along columns
    for j in range(cols):
        col_data = [fft_img[i][j] for i in range(rows)]
        col_data = fft1d(one_step_fft_shuffling(col_data))
        for i in range(rows):
            ifft_img[i][j] = int(cmath.phase(col_data[i]) * 255 / (2 * cmath.pi))

    # Normalize pixel values to the range [0, 255]
    min_val = min(min(row) for row in ifft_img)
    max_val = max(max(row) for row in ifft_img)
    if max_val != min_val:
        scale_factor = 255 / (max_val - min_val)
        ifft_img = [[int((val - min_val) * scale_factor) for val in row] for row in ifft_img]

    return ifft_img

def read_raw_image(filename, size):
    with open(filename, 'rb') as file:
        raw_data = file.read()
        # Unpack the raw bytes into complex numbers (assuming 8-bit unsigned integers)
        image = [complex(raw_byte - 128, 0) for raw_byte in raw_data]
        return [image[i:i+size] for i in range(0, size*size, size)]

def write_raw_image(filename, data):
    with open(filename, 'wb') as file:
        for row in data:
            # Scale and normalize the values to the range [0, 255]
            normalized_row = [int((c.real + 128) / 2) for c in row]
            # Pack the integers into raw bytes
            raw_bytes = bytes(normalized_row)
            file.write(raw_bytes)

if __name__ == "__main__":
    # Read the raw images
    size = 256
    square_image = read_raw_image("square256.raw", size)
    car_image = read_raw_image("car.raw", size)

    # Perform FFT on the images
    fft_square = fft2d(square_image)
    fft_car = fft2d(car_image)

    # Perform IFFT to reconstruct the images
    reconstructed_square = ifft2d(fft_square)
    reconstructed_car = ifft2d(fft_car)

    # Save the reconstructed images as raw files
    write_raw_image("reconstructed_square.raw", reconstructed_square)
    write_raw_image("reconstructed_car.raw", reconstructed_car)
