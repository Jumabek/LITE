import os
import time
from lite_deepsort_app import run
from opts import opt
import warnings
from os.path import join
warnings.filterwarnings("ignore")


def process_sequence(seq, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    device = f'cuda:0'
    start_time = time.time()

    print(
        f'Processing video {seq} on {device} (process ID: {os.getpid()})...', flush=True)
    path_save = join(opt.dir_save, seq + '.txt')
    run(
        sequence_dir=join(opt.dir_dataset, seq),
        output_file=path_save,
        min_confidence=opt.min_confidence,
        nms_max_overlap=opt.nms_max_overlap,
        min_detection_height=opt.min_detection_height,
        nn_budget=opt.nn_budget,
        display=False,
        visualize=False,
        verbose=True,
        device=device
    )
    end_time = time.time()
    print(
        f'Finished processing video {seq} on {device} in {end_time - start_time:.2f} seconds', flush=True)


if __name__ == '__main__':
    start_time = time.time()

    gpu_id = 0 
    sequences = opt.sequences

    for seq in sequences:

        process_sequence(seq, gpu_id)

    end_time = time.time()
    total_time = end_time - start_time
    print(
        f'Total time taken for the run: {total_time:.2f} seconds', flush=True)
