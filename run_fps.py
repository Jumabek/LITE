from multiprocessing import Pool
from lite_deepsort_app import run
from opts import opt
import time
from os.path import join
import os


def process_sequence(seq):
    print(f'Processing video {seq}...')
    tick = time.time()
    path_save = join(opt.dir_save, seq + '.txt')
    sequence_dir = join(opt.dir_dataset, seq)
    device = f'cuda:0'
    run(
        sequence_dir=sequence_dir,
        output_file=path_save,
        min_confidence=opt.min_confidence,
        nms_max_overlap=opt.nms_max_overlap,
        min_detection_height=opt.min_detection_height,
        nn_budget=opt.nn_budget,
        display=True,
        device=device,
        verbose=True
    )
    tock = time.time()

    num_frames = len(os.listdir(join(sequence_dir, 'img1')))
    print(f'Number of frames: {num_frames}')

    time_spent_for_the_sequence = tock - tick

    avg_time_per_frame = (time_spent_for_the_sequence) / num_frames

    FPS = 1/avg_time_per_frame

    path_to_fps_csv = f'results/{sequence_dir.split("/")[1]}-FPS/{opt.sequence}/fps.csv'

    if not os.path.exists(path_to_fps_csv):
        with open(path_to_fps_csv, 'w') as f:
            f.write('tracker_name,sequence_name,FPS,conf\n')

    with open(path_to_fps_csv, 'a') as f:
        f.write(f'{opt.tracker_name},{seq},{FPS:.1f},{opt.min_confidence}\n')
    print(f'Finished video {seq}')


if __name__ == '__main__':

    process_sequence(opt.sequence)
