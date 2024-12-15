import os
import time
from lite_deepsort_app import run
from opts import opt
import warnings
from os.path import join
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

warnings.filterwarnings("ignore")


def process_sequence(seq, gpu_id, model):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    device = f'cuda:0'
    start_time = time.time()

    print(
        f'Processing video {seq} on {device} (process ID: {os.getpid()})...', flush=True)
    path_save = join(opt.dir_save, seq + '.txt')



    run(
        model=model,
        sequence_dir=join(opt.dir_dataset, seq),
        output_file=path_save,
        min_confidence=opt.min_confidence,
        nms_max_overlap=opt.nms_max_overlap,
        min_detection_height=opt.min_detection_height,
        nn_budget=opt.nn_budget,
        display=True,
        visualize=False,
        verbose=True,
        device=device
    )
    end_time = time.time()
    print(
        f'Finished processing video {seq} on {device} in {end_time - start_time:.2f} seconds', flush=True)


if __name__ == '__main__':
    start_time = time.time()
    
    model_name = opt.yolo_model + '.pt'
    model = YOLO(model_name)

    print(f'Loaded YOLO model: {model_name}', flush=True)

    # Load the model
    gpu_id = 0
    sequences = opt.sequences

    with ThreadPoolExecutor() as executor:
        # Submit all sequences to run in parallel
        futures = [executor.submit(process_sequence, seq, gpu_id, model)
                   for seq in sequences]
        # Wait for all futures to complete
        for future in futures:
            future.result()

    end_time = time.time()
    total_time = end_time - start_time
    print(
        f'Total time taken for the run: {total_time:.2f} seconds', flush=True)
