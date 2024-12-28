import os
import time
from track import run
from opts import opt
import warnings
from os.path import join
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
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

    # Load the model
    gpu_id = 0
    sequences = opt.sequences

    with ThreadPoolExecutor() as executor:
        # Submit all sequences to run in parallel
        futures = [executor.submit(process_sequence, seq, gpu_id)
                   for seq in sequences]
        # Wait for all futures to complete
        for future in futures:
            future.result()

    end_time = time.time()
    total_time = end_time - start_time
    print(
        f'Total time taken for the run: {total_time:.2f} seconds', flush=True)
