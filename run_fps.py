from multiprocessing import Pool
from lite_deepsort_app import run
from opts import opt
import time
import warnings
from os.path import join
warnings.filterwarnings("ignore")


def process_sequence(seq):
    print(f'Processing video {seq}...')
    path_save = join(opt.dir_save, seq + '.txt')
    device = f'cuda:0'
    run(
        sequence_dir=join(opt.dir_dataset, seq),
        output_file=path_save,
        min_confidence=opt.min_confidence,
        nms_max_overlap=opt.nms_max_overlap,
        min_detection_height=opt.min_detection_height,
        nn_budget=opt.nn_budget,
        display=True,
        device=device,
        verbose=True
    )
    print(f'Finished video {seq}')


if __name__ == '__main__':
    for sequence in opt.sequences:
        process_sequence(sequence)
        break

    # process_sequence('dancetrack0001')

# if __name__ == '__main__':
#     with Pool(processes=4) as pool:  # Adjust number of processes based on your system's capability
#         pool.map(process_sequence, opt.sequences)
