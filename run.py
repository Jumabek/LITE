import os
import time
from track import run
from opts import opt
import warnings
from os.path import join
warnings.filterwarnings("ignore")
from pathlib import Path


def process_sequence(seq, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    device = f'cuda:0'
    start_time = time.time()

    print(
        f'Processing video {seq} on {device} (process ID: {os.getpid()})...', flush=True)
    path_save = join(opt.dir_save, 'data', seq + '.txt')
    os.makedirs(Path(path_save).parent, exist_ok=True)
    
    run(
        sequence_dir=join(opt.dir_dataset, seq),
        output_file=path_save,
        nn_budget=opt.nn_budget,
        visualize=opt.visualize,
        verbose=True,
        device=device
    )
    end_time = time.time()
    print(
        f'Finished processing video {seq} on {device} in {end_time - start_time:.2f} seconds', flush=True)

    if opt.fps_save:
        num_frames = len(os.listdir(join(opt.dir_dataset, seq, 'img1')))
        avg_time_per_frame = (end_time - start_time) / num_frames
        FPS = 1 / avg_time_per_frame
        path_to_fps_csv = join(opt.dir_save, 'fps.csv')
        if not os.path.exists(path_to_fps_csv):
            with open(path_to_fps_csv, 'w') as f:
                f.write('tracker_name,sequence_name,FPS,conf\n')
        with open(path_to_fps_csv, 'a') as f:
            f.write(f'{opt.tracker_name},{seq},{FPS:.1f},{opt.min_confidence}\n')


if __name__ == '__main__':
    start_time = time.time()

    gpu_id = 0
    sequences = opt.sequences
    sequences = sorted(sequences)

    for seq in sequences:
        process_sequence(seq, gpu_id)

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total time taken for the run: {total_time:.2f} seconds', flush=True)
