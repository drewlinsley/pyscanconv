import sys

import numpy as np

from skimage import io
from skimage.transform import resize


from utils_scan import associative_scan
import torch.nn.functional as F
import torch

from matplotlib import pyplot as plt
from timeit import default_timer as timer


def torch_conv_binary_operator(q_i, q_j):
    """Assumes 1x1 kernels
       :inputs q_i an q_j are tuples containing (A_i, BU_i) and (A_j, BU_j)
       :inputs A_i and A_j are (P,)
       :inputs BU_i and BU_j are bszxH_UxW_UxP
       :returns tuple where first entry AA is (P,)
                and second entry is bszxH_UxW_UxP"""

    A_i, BU_i = q_i
    A_j, BU_j = q_j

    A_jBU_i = torch.func.vmap(F.conv2d)(BU_i, A_j, padding="same")
    # A_jBU_i = torch.nn.functional.layer_norm(A_jBU_i, A_jBU_i.shape[1:])
    A_jBU_i = normalize(A_jBU_i)
    # A_jBU_i = A_jBU_i_mag * torch.exp(1j * torch.angle(A_jBU_i))
    # AA = torch_vmap_conv(A_i, A_j, activity=False)
    AA = torch.func.vmap(F.conv2d)(A_i, A_j, padding="same")
    AA = normalize(AA)
    return AA, A_jBU_i + BU_j


def instance_normalize(x, eps=1e-4):
    return (x - x.mean((-3, -2, -1), keepdims=True)) / (x.std((-3, -2, -1), keepdims=True) + eps)


def layer_normalize(x, eps=1e-4):
    return (x - x.mean((-2, -1), keepdims=True)) / (x.std((-2, -1), keepdims=True) + eps)


if __name__ == '__main__':

    # timesteps = 120
    timesteps = int(sys.argv[1])
    repeats = int(sys.argv[2])
    normalize = layer_normalize
    normalize = instance_normalize

    kernel = np.load("gabors_for_contours_7.npy", allow_pickle=True, encoding="latin1").item()["s1"][0]
    kernel = kernel[..., :-1]  # Reduce to 24 channels. Final filter is a dot.

    # Make the kernel into an identity NxN kernel
    identity = torch.eye(kernel.shape[-1])[None, None]
    ones = torch.ones((kernel.shape[0], kernel.shape[1], 1, 1))
    kernel = (identity * ones) * torch.from_numpy(kernel)
    kernel = kernel / kernel.max()

    # Change kernel to In x Out x H x W
    kernel = kernel.permute(2, 3, 0, 1)

    # Load image
    image = io.imread("test.png")
    x = image[..., [0]] / 255.
    x = resize(x, (x.shape[0] // 4, x.shape[1] // 4), anti_aliasing=True)
    x = x[None].astype(np.float32)
    x = (x - x.mean()) / x.std()
    # x = normalize(x)
    # np.save("input_image2", x)
    # np.save("kernel2", kernel)
    x = torch.from_numpy(x)

    # Inflate channel dim
    x = x.repeat(1, 1, 1, kernel.shape[0])

    # Reshape to NxCxHxW
    x = x.permute(0, 3, 1, 2)

    # Sequential conv
    k_a = kernel[None].repeat(timesteps, 1, 1, 1, 1)
    x_a = x[None].repeat(timesteps, 1, 1, 1, 1)
    k_b = k_a.clone()
    x_b = x_a.clone()

    # Run scan conv
    timings = []
    Bu = F.conv2d(
        x_a[0],
        k_a[0],
        padding=k_a.shape[-1]//2)
    Bu = Bu[None].repeat(timesteps, 1, 1, 1, 1)
    Bu = normalize(Bu)
    for _ in range(repeats):
        scan_time = timer()

        # Run the full sequence conv each step
        a_k_fft, a_x_fft = associative_scan(
            torch_conv_binary_operator,
            (k_a, Bu),
            axis=0)
        scan_time = timer() - scan_time
        timings.append(scan_time)
    scan_time = np.mean(timings)
    afout = a_x_fft[-1]
    print("Timing")
    print("Scan Conv: {}".format(scan_time))

    #control with one conv per ts
    timings = []
    for i in range(repeats):
        conv_time = timer()
        u_conv = F.conv2d(
            x_b[i],
            k_b[i],
            padding=k_b.shape[-1]//2)
        uconv = normalize(u_conv)
        hidden = torch.zeros_like(uconv.clone())
        hiddens = []
        for t in range(timesteps):
            hidden = F.conv2d(hidden, k_b[t], padding="same")

            hidden = normalize(hidden)

            hidden = hidden + u_conv
            hiddens.append(hidden)

        timings.append(timer() - conv_time)
    conv_time = np.mean(timings)
    print("Conv Timing")
    print("Conv: {}".format(conv_time))

    diff = np.mean(np.sqrt((hidden.cpu().numpy() - afout.cpu().numpy()) ** 2))
    print("L2 diff between outputs: {}".format(diff))

    f = plt.figure()
    plt.subplot(131)
    plt.imshow(hidden[0, 0].cpu().numpy())
    plt.title(f"Conv time: {conv_time:.2f}")
    plt.subplot(132)
    plt.imshow(afout[0, 0].cpu().numpy())
    plt.title(f"Scan time: {scan_time:.2f}")
    plt.subplot(133)
    plt.imshow(np.abs(hidden[0, 0].cpu().numpy() - afout[0, 0].cpu().numpy()))
    plt.title(f"Diff: {diff:.2f}")
    plt.show()
    plt.close(f)

    import pdb;pdb.set_trace()
    a= 2