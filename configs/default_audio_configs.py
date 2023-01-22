import ml_collections
import torch


def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 256  # bs to calculate the gt field
    training.n_iters = 500000 # total number of iterations
    training.snapshot_freq = 10000 # save model every 10000 iterations
    training.log_freq = 100
    training.eval_freq = 10000 
    training.snapshot_freq_for_preemption = 1000 # save meta ckpt every 1000 iterations
    training.snapshot_sampling = True
    training.reduce_mean = True
    training.M = 280
    training.amp = False # automatic mixed precision
    training.accum_iter = 0 # gradient accumulation

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.N = 1 # number of sampling steps
    sampling.z_exp = 1 # start of substituting z prediction with gt
    sampling.rk_stepsize = 0.9 # stepsize for the rk4 integrator of torchdiffeq

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.batch_size = 8
    evaluate.enable_sampling = True
    evaluate.num_samples = 32
    evaluate.enable_loss = False # calculate loss on the evaluation set
    evaluate.save_images = True  # debugging
    evaluate.show_sampling = False # generate gif from sampling (only in euler methods)
    
    # data
    config.data = data = ml_collections.ConfigDict()
    data.centered = False # if data is in -1 to 1 range
    data.num_channels = 1 # number of image channels
    
    # model
    config.model = model = ml_collections.ConfigDict()

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'AdamW'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.
    optim.scheduler = 'none'  # 'none', 'OneCylce'
    optim.max_lr = 3e-4 # for OneCycle
    config.seed = 49
    config.device = "cuda" if torch.cuda.is_available() else torch.device('cpu')

    return config


# audio configs for 64x64 mels and 16kHz sampling rate
def get_mels_64():
    spec = ml_collections.ConfigDict()
    spec.num_mels = 64
    spec.nfft = 1024
    spec.hop_length = 256
    spec.sample_rate = 16_000
    spec.fmin = 20
    spec.audio_length = 1
    spec.image_size = spec.audio_length * spec.sample_rate // spec.hop_length + 2  # this is 64 which fits the num mels
    spec.spec_len_samples = spec.image_size
    return spec


# audio configs for 128x128 mels and 16kHz sampling rate
def get_mels_128():
    spec = ml_collections.ConfigDict()
    spec.num_mels = 128
    spec.nfft = 512
    spec.hop_length = 128
    spec.sample_rate = 16_000
    spec.fmin = 20
    spec.audio_length = 1
    spec.image_size = spec.audio_length * spec.sample_rate // spec.hop_length + 3  # this is 128 which fits the num mels
    spec.spec_len_samples = spec.image_size
    return spec
