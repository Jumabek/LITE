import os
import time
from multiprocessing import Pool, current_process
from track import run
from opts import opt
import warnings
from os.path import join
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")


def process_sequence(seq, gpu_id):
    # Explicitly set the CUDA_VISIBLE_DEVICES to the specified GPU only
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Since we are setting the visible devices to a single GPU, it is always cuda:0
    device = f'cuda:0'
    start_time = time.time()

    try:
        print(
            f'Processing video {seq} on {device} (process ID: {os.getpid()})...', flush=True)
        path_save = join(opt.dir_save, seq + '.txt')
        print(f'Saving results to {path_save}')
        run(
            sequence_dir=join(opt.dir_dataset, seq),
            output_file=path_save,
            min_confidence=opt.min_confidence,
            nms_max_overlap=opt.nms_max_overlap,
            min_detection_height=opt.min_detection_height,
            nn_budget=opt.nn_budget,
            display=True,
            verbose=True,
            device=device
        )
        end_time = time.time()
        print(
            f'Finished processing video {seq} on {device} in {end_time - start_time:.2f} seconds', flush=True)
    except Exception as e:
        print(
            f'Error processing video {seq} on {device} (process ID: {os.getpid()}): {str(e)}', flush=True)


def process_sequences_on_gpu(sequences, gpu_id):
    # Adjust max_workers as needed
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_sequence, seq, gpu_id)
                   for seq in sequences]
        for future in as_completed(futures):
            try:
                future.result()  # To raise any exceptions occurred
            except Exception as e:
                print(
                    f'Error in future for GPU {gpu_id} (process ID: {os.getpid()}): {str(e)}', flush=True)


if __name__ == '__main__':
    start_time = time.time()

    #gpu_ids = [0, 1, 2, 3]  # List of GPU indices to use
    gpu_ids = [0]  # List of GPU indices to use
    sequences = opt.sequences

    # Split sequences into chunks, one for each GPU
    chunk_size = len(sequences) // len(gpu_ids)
    sequence_chunks = [
        sequences[i * chunk_size: (i + 1) * chunk_size] for i in range(len(gpu_ids))]

    # Ensure all sequences are included in case of uneven division
    if len(sequences) % len(gpu_ids) != 0:
        sequence_chunks[-1].extend(sequences[len(gpu_ids) * chunk_size:])

    # Use multiprocessing Pool with the same number of processes as GPUs
    with Pool(processes=len(gpu_ids)) as pool:
        results = []
        for i, chunk in enumerate(sequence_chunks):
            gpu_id = gpu_ids[i]
            print(
                f'Assigning GPU {gpu_id} to process chunk {i+1}/{len(sequence_chunks)}', flush=True)
            result = pool.apply_async(
                process_sequences_on_gpu, args=(chunk, gpu_id))
            results.append(result)

        for result in results:
            result.wait()

        pool.close()
        pool.join()

    end_time = time.time()
    total_time = end_time - start_time
    print(
        f'Total time taken for the run: {total_time:.2f} seconds', flush=True)