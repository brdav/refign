import os

import torch
from torch.utils import cpp_extension

cwd = os.path.dirname(os.path.realpath(__file__))

sources = []
sources.append(os.path.join(cwd, 'correlation.cpp'))

if torch.cuda.is_available():
    sources.append(os.path.join(cwd, 'correlation_sampler.cpp'))
    sources.append(os.path.join(cwd, 'correlation_cuda_kernel.cu'))
    correlation = cpp_extension.load('correlation',
                                     sources=sources,
                                     build_directory=cwd,
                                     extra_cflags=['-fopenmp'],
                                     extra_ldflags=['-lgomp'],
                                     with_cuda=True,
                                     verbose=False)
else:
    # CPU only version
    sources.append(os.path.join(cwd, 'correlation_sampler_cpu.cpp'))
    correlation = cpp_extension.load('correlation',
                                     sources=sources,
                                     build_directory=cwd,
                                     extra_cflags=['-fopenmp'],
                                     extra_ldflags=['-lgomp'],
                                     with_cuda=False,
                                     verbose=False)
